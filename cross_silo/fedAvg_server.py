'''Basic server for federated learning with Flower. Implements FedAvg strategy.'''
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict
import time

import flwr as fl
from flwr.common import Metrics
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.NeuralNetwork import ModularNet
from src.utils import create_dataloaders, create_dataloader,load_test_partitions, test_acc_f1
from src.utils import save_history, build_history_datastructure
from src.dataset_utils import (load_named_dataset, one_hot_encoding, ordinal_encoding,
                               cap_dataset, get_hus)
from src.fl_parser import fl_server_parser, compose_paths
from src.plotting_utils import plot_history, save_fig
from src.shap_explanations import get_shap_explanations
from src.dalex_explanations import get_dalex_explanation

# Define a global list to store the history of metrics
# at each round of federated learning, save the accuracy, the f1 score and the loss
history = []

def fit_config(server_round: int):
    '''Return training configuration dict for each round.
    Keep batch size constant at 32, but increase the number of local epochs
    '''
    config = {
        'batch_size': 4,
        'local_epochs': 1 if server_round < 5 else 
                        (2 if server_round < 15 else 3),
        'lr': 0.01,
        'momentum': 0.9,
        'class_weight': False

    }
    return config

def evaluate_config(server_round: int):
    '''Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    '''
    val_steps = 5 if server_round < 4 else 10
    return {'val_steps': val_steps, 'server_round': server_round}

def get_evaluate_fn(model: torch.nn.Module, val_loader: DataLoader):
    '''Return an evaluation function for server-side evaluation.'''
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, # pylint: disable=unused-argument
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar], # pylint: disable=unused-argument
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy, f1_macro = test_acc_f1(model,val_loader)
        print("--"*9,"adding server dict")
        server_dict = {"name":"server",
                       "accuracy":accuracy,
                       "f1_score":f1_macro,
                       "loss":loss,
                       "round":server_round}
        history.append(server_dict)
        return loss, {'accuracy': accuracy, 'f1_score': f1_macro}

    return evaluate

def create_dictionary_from_metrics(metrics):
    '''Create a dictionary from the metrics'''
    partial_dict = {}
    for _ , metric in metrics:
        cl_name = metric["name"]
        partial_dict[cl_name] = [metric["loss"],metric["accuracy"],metric["f1_score"]]
    return partial_dict,len(metrics[1])

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    '''Aggregate metrics by weighted average, weighted by number of examples.'''
    # Calculate weighted average of metrics
    # if the metrics are empty return a empty dict, check on the first 
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    # save the history of metrics
    client_dict, clients_evauated = create_dictionary_from_metrics(metrics)
    print("--"*9,clients_evauated," client dictionary added to history")
    history.append(client_dict)

    # do the same for the trainset
    tr_accuracies = [num_examples * m["accuracy_train"] for num_examples, m in metrics]
    tr_f1_scores = [num_examples * m["f1_score_train"] for num_examples, m in metrics]
    # Aggregate and return weighted average over the samples of the metrics
    old_result = {"accuracy": sum(accuracies) / sum(examples),"f1_score":sum(f1_scores)/sum(examples),
            "accuracy_train":sum(tr_accuracies)/sum(examples),
            "f1_score_train":sum(tr_f1_scores)/sum(examples)}
    # the new result it the simple average of the metrics
    new_result = {"accuracy": sum([m["accuracy"] for _, m in metrics])/clients_evauated,
                  "f1_score":sum([m["f1_score"] for _, m in metrics])/clients_evauated,
                  "accuracy_train":sum([m["accuracy_train"] for _, m in metrics])/clients_evauated,
                  "f1_score_train":sum([m["f1_score_train"] for _, m in metrics])/clients_evauated}
    return new_result

def start_server(dataset_name: str,num_rounds = 20,num_clients=4):
    '''Function to start the server and perform the federated learning.'''   
    # Load the dataset as a TensorDataset
    tr_set, te_set = load_named_dataset(dataset_name)
    # then
    tr_set = one_hot_encoding(tr_set)
    te_set = one_hot_encoding(te_set)
    _, validation_loader , _ = create_dataloaders(trainset=tr_set,testset=te_set)

    # Initialize model and get initial weights
    num_features = validation_loader.dataset.tensors[0].shape[1]
    num_classes = validation_loader.dataset.tensors[1].shape[1]
    print("Dataset:", dataset_name)
    print("Number of features:", num_features)
    print("Number of classes:", num_classes)

    # SAME MODEL AS IN THE CLIENTS
    #hu = get_hus(dataset_name)
    #neural_model = Net(num_features, hidden_units = hu, num_classes = num_classes)
    neural_model = ModularNet(input_size=num_features,
                   layers=[64,64],
                   num_targets=num_classes,
                   dropout=0.05)
    # print type of instanciated model

    model_parameters = [val.cpu().numpy() for _, val in neural_model.state_dict().items()]
    # federated average as strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit = 1.0, # Percentage client used for training (1 indicates all)
        fraction_evaluate = 1,
        #min_fit_clients = 4, # min number of clients used for training
        min_evaluate_clients = num_clients,
        min_available_clients = num_clients, # Minimum number of clients to make the round advance
        evaluate_metrics_aggregation_fn = weighted_average, # Aggregate the metrics 
        #fl.server.strategy.aggregate.weighted_loss_avg
        
        evaluate_fn = get_evaluate_fn(neural_model, validation_loader),
        on_fit_config_fn = fit_config,
        on_evaluate_config_fn = evaluate_config,
        initial_parameters= fl.common.ndarrays_to_parameters(model_parameters)
    )

    # start timer
    start_fl = time.time()
    # Start Flower server (SSL-enabled) for four rounds of federated learning

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1, #percebtuale di client usati per training (1 indica tutti)
        fraction_evaluate=1,
        min_fit_clients=8, #min numero client usdati per train (< n_client indica che primo round non li usa tutti)
        min_evaluate_clients=1,
        min_available_clients=8, #numero clienti che server aspetta prima di iniziare tutto
        evaluate_fn=get_evaluate_fn(neural_model,val_loader=validation_loader),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address = '0.0.0.0:8080',
        config = fl.server.ServerConfig(num_rounds=num_rounds),
        strategy = strategy,
    )
    # elapsed time
    fl_elapsed = time.time() - start_fl
    return neural_model, tr_set, te_set, fl_elapsed

if __name__ == '__main__':

    parser = fl_server_parser()
    args = parser.parse_args()
    # Set the name for the distribution of the dataset
    distribution = 'iid' if args.alpha > 1.0 else 'imb_'+str(args.alpha)


    # Create the paths based on the arguments
    model_path, explanation_path, history_path = compose_paths(args, client_id = -1)

    # Start the server and perform the federated learning
    nn, whole_tr_set, whole_te_set, elapsed = start_server(dataset_name=args.dataset_name,
                                                          num_rounds = args.num_rounds,
                                                          num_clients = args.num_clients)
    time_path = history_path.replace(".json","_server_time.txt")
    with open(time_path,'w+',encoding='utf-8') as file_name:
        file_name.write(str(elapsed))
    # log the history of metrics
    save_history(history,history_path)
    # Plot the history of metrics including the clients metrics
    # then save the figure
    metrics_centralized,clients_metrics = build_history_datastructure(history,args.num_clients)
    fig = plot_history(metrics_centralized,clients_metrics,
                       dt_name=args.dataset_name,
                       experiment_settings=str(args.num_clients)+"_"+distribution)
    # replace the .json with "" as postponig .png is done in the function
    save_fig(fig,history_path.replace(".json","_plot"))

    # FL ended - Save model and explanation server side
    torch.save(nn, model_path)

    exit(0)

    # whole_test_partition, _ = create_dataloader(one_hot_encoding(whole_te_set))
    # whole_train_loader, _ = create_dataloader(ordinal_encoding(whole_tr_set))
    # The representative samples used as background data
    # are the mean of the training set
    background_data = whole_tr_set.tensors[0].mean(axis=0).reshape(1,-1)
    print(background_data.shape)
    background = TensorDataset(torch.tensor(background_data),
                               torch.tensor([1]))
    # generated during the partitioning phase
    # and create the background data loader
    # background_path = history_path.replace("history.json","background_data.npz")
    # background = np.load(background_path)
    # background = TensorDataset(torch.tensor(background["features"]),
    #  torch.tensor(background["labels"]))
    bg_loader, _ = create_dataloader(background, batch_size=32, shuffle=False)
    # load the concatenated test set of all clients
    partitions_te_set = load_test_partitions(args.dataset_name,
                                             args.num_clients,
                                             dataset_distribution=distribution)
    # cap the dataset to 2000 samples and create the dataloader
    loader_explain, _ = create_dataloader(ordinal_encoding(
                                            cap_dataset((partitions_te_set.tensors[0],
                                                        partitions_te_set.tensors[1]),
                                                        2000)
                                            ),
                                             batch_size=32, shuffle=False)

    ##################################################
    ##### DALEX EXPLANATIONS SIDE ####################
    ##################################################
    # Get the dalex values
    # start timer
    start = time.time()
    dalex_values = get_dalex_explanation(black_box = nn,
                                         background_data_loader = bg_loader,
                                         explain_partition = loader_explain)
    # elapsed time
    elapsed = time.time() - start

    time_path = explanation_path.replace("shap","dalex")+"_server_time.txt"
    with open(time_path,'w+',encoding='utf-8') as file_name:
        file_name.write(str(elapsed))
    print("Dalex explanation on server took",elapsed,"seconds")
    # Save the dalex values
    np.savez(explanation_path.replace("shap","dalex"), dalex_values = dalex_values)
    print("Saved dalex explanation matrix of Server")
    ##################################################
    ##### SHAP EXPLANATIONS SIDE #####################
    ##################################################
    # Get the shap values
    # start timer
    start = time.time()
    shap_values =  get_shap_explanations(black_box = nn,
        # the original code uses the median of the whole training set as background data
        # I use a proportion of fixed samples from the training set as background data
                                         background_data_loader = bg_loader,
                                         explain_partition = loader_explain)
    # elapsed time
    elapsed = time.time() - start
    time_path = explanation_path+"_server_time.txt"
    with open(time_path,'w+',encoding='utf-8') as file_name:
        file_name.write(str(elapsed))
    print("Shap explanation on server took",elapsed,"seconds")

    # Save the shap values
    np.savez(explanation_path, shap_values=shap_values)
    print('Saved explanation of Server')
