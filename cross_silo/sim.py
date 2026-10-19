'''This script is used to simulate federated learning process on 
a given dataset with the arguments for the HP search.'''
import os
from collections import OrderedDict
from typing import Dict, Tuple, List, Optional
import torch
from torch.utils.data import DataLoader,TensorDataset

import flwr as fl
from flwr.common import Metrics
import wandb
# import the run from wandb


from src.NeuralNetwork import ModularNet
from src.dataset_utils import one_hot_encoding, split_data
from src.utils import (create_dataloaders,load_partition, load_test_partitions, loss_acc_f1_M_w,create_dataloader)
from src.fl_parser import simulation_parser, compose_paths

# Server and Client imports
from basic_client import BasicClient
from fedAvg_server import evaluate_config



def get_distribution_name(alpha= None ):
    '''Return description of distribution'''
    if alpha < 110:
        return "imb_"+str(alpha)
    else:
        return "iid"
def get_client_fn(arguments,num_classes=3,num_features=4,model=None):
                  #server_run=None,
                  #runs_id=None):
    """Return a function to construct a client.

    The VirtualClientEngine will exectue this function whenever a client is sampled by
    the strategy to participate.
    """
    dataset_name = arguments.dataset_name
    num_clients = arguments.num_clients
    distribution = get_distribution_name(alpha=arguments.alpha)
    def client_fn(cid: str) -> fl.client.Client:
        def partition_fn(cid: int):
            devset, testset = load_partition(dataset_folder=dataset_name,
                                            partition_id=cid+1,
                                            total_partitions=num_clients,
                                            dataset_distribution=distribution)
            if arguments.sweep:
                # split the trainset into train and validation with 80/20 ratio
                trainset80, valset20 = split_data((devset.tensors[0],
                                            devset.tensors[1]), 0.8)
                # now transform the trainset80 into a TensorDataset
                trainset80 = TensorDataset(*trainset80)
                valset20 = TensorDataset(*valset20)
                train_loader, val_loader, _ = create_dataloaders(trainset=trainset80,
                                                            testset=valset20)
                test_loader = DataLoader(testset, batch_size=32, shuffle=False)
            else:
                val_loader = None
                train_loader, test_loader, _ = create_dataloaders(trainset=devset,
                                                            testset=testset)
            
            return train_loader, val_loader, test_loader, model
        
        """Construct a FlowerClient with its own dataset partition."""
        client_name = "client_"+str(int(cid)+1).zfill(3)
        train_loader,val_loader, test_loader, local_model = partition_fn(int(cid))
        print("Client",client_name,"has been sampled!")
        # Create and return client
        return BasicClient(model=local_model,
                           train_loader=train_loader,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           client_name=client_name)
    return client_fn

def main():
    parser = simulation_parser()

    # Parse input arguments
    args = parser.parse_args()

    distribution = 'iid' if args.alpha > 110.0 else 'imb_'+str(args.alpha)
    # Load the dataset as a TensorDataset
    partitions_te_set = load_test_partitions(args.dataset_name,
                                             args.num_clients,
                                             dataset_distribution=distribution)

    test_loader = DataLoader(partitions_te_set, batch_size=args.batch_size, shuffle=False)


    # Initialize model and get initial weights
    num_features = test_loader.dataset.tensors[0].shape[1]
    num_classes = test_loader.dataset.tensors[1].shape[1]
    print("Dataset:", args.dataset_name)
    print("Number of features:", num_features)
    print("Number of classes:", num_classes)

    run = wandb.init()
    if not args.sweep:
        # log the hyperparameters
        run.config.update(args)
    #runs_id = [wandb.util.generate_id() for _ in range(args.num_clients)]
    neural_model = ModularNet(num_features, layers = [args.hidden_units]*args.layers,
                       num_targets = num_classes, dropout = args.dropout)
    # print type of instanciated model
    print("Model type:",type(neural_model))
    model_parameters = [val.cpu().numpy() for _, val in neural_model.state_dict().items()]
    args.explainer_name = "NoExplainer"
    model_path, _, _ = compose_paths(args, client_id = -1)
    if not args.sweep:
        # remove last part of the model_path using the split
        model_path = os.path.sep.join(model_path.split(os.path.sep)[:-1])
        model_path = os.path.join(model_path, run.name+"_model.pth")
    def fit_config(server_round: int):
        '''Return training configuration dict for each round.
        '''
        config = {
            'batch_size': args.batch_size,
            'local_epochs': args.local_epochs,
            'lr': args.learning_rate,
            'momentum': args.momentum,
            'class_weight': args.class_weights,
            'server_round': server_round
        }
        return config
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        '''Aggregate metrics by weighted average, weighted by number of examples.'''
        # Calculate weighted average of metrics
        #input("Press Enter to continue...")
        server_round = metrics[0][1]["server_round"]
        examples = [num_examples for num_examples, _ in metrics]

        accuracies = [num_examples * m["accuracy_test"] for num_examples, m in metrics]
        f1_scores = [num_examples * m["f1_score_test"] for num_examples, m in metrics]

        accuracies_train = [num_examples * m["accuracy_train"] for num_examples, m in metrics]
        f1_train = [num_examples * m["f1_score_train"] for num_examples, m in metrics]
        if args.sweep:
            accuracies_val = [num_examples * m["accuracy_val"] for num_examples, m in metrics]
            f1_w_val = [num_examples * m["f1_w_val"] for num_examples, m in metrics]
            f1_macro_val = [num_examples * m["f1_score_val"] for num_examples, m in metrics]
            # log the metrics with wandb if sweep is running
            wandb.log({"global/val_accuracy": sum(accuracies_val) / sum(examples),
                    "global/val_f1_macro": sum(f1_macro_val) / sum(examples),
                    "global/val_f1_w": sum(f1_w_val) / sum(examples),
                    "round": server_round,})
        # always log the train metrics
        wandb.log({
                "global/train_accuracy": sum(accuracies_train) / sum(examples),
                "global/train_f1_macro": sum(f1_train) / sum(examples),
                "round": server_round,
                })
        # log the metrics of every client with a name for each client
        for i, (num_examples, m) in enumerate(metrics):
                wandb.log({
                        "client_"+str(i+1).zfill(3)+"/train_accuracy": m["accuracy_train"],
                        "client_"+str(i+1).zfill(3)+"/train_f1_macro": m["f1_score_train"],
                        "client_"+str(i+1).zfill(3)+"/train_loss": m["loss_train"],
                        "client_"+str(i+1).zfill(3)+"/train_f1_weighted": m["f1_w_train"],

                        "client_"+str(i+1).zfill(3)+"/test_accuracy": m["accuracy_test"],
                        "client_"+str(i+1).zfill(3)+"/test_f1_macro": m["f1_score_test"],
                        "client_"+str(i+1).zfill(3)+"/test_loss": m["loss_test"],
                        "client_"+str(i+1).zfill(3)+"/test_f1_weighted": m["f1_w_test"],
                        "round": server_round,
                    })
                # now log precision and recall for each client, for each class
                for j in range(num_classes):
                    wandb.log({
                        "client_"+str(i+1).zfill(3)+"/train_precision_class_"+str(j).zfill(2): m["pr_train"][j],
                        "client_"+str(i+1).zfill(3)+"/train_recall_class_"   +str(j).zfill(2): m["rc_train"][j],
                        "client_"+str(i+1).zfill(3)+"/test_precision_class_" +str(j).zfill(2): m["pr_test"][j],
                        "client_"+str(i+1).zfill(3)+"/test_recall_class_"    +str(j).zfill(2): m["rc_test"][j],
                        "round": server_round,
                        })
                # log the validation metrics if sweep is running
                if args.sweep:
                    wandb.log({
                        "client_"+str(i+1).zfill(3)+"/val_f1_macro": m["f1_score_val"],
                        "client_"+str(i+1).zfill(3)+"/val_f1_weighted": m["f1_w_val"],
                        "client_"+str(i+1).zfill(3)+"/val_accuracy": m["accuracy_val"],
                        "client_"+str(i+1).zfill(3)+"/val_loss": m["loss_val"],
                        "round": server_round,
                        })
                    for j in range(num_classes):
                        wandb.log({
                        "client_"+str(i+1).zfill(3)+"/val_precision_class_"+str(j).zfill(2): m["pr_val"][j],
                        "client_"+str(i+1).zfill(3)+"/val_recall_class_"   +str(j).zfill(2): m["rc_val"][j],
                        "round": server_round,
                        })
                    

                    
        # Aggregate and return weighted average over the samples of the metrics
        return {"accuracy": sum(accuracies) / sum(examples),
                "f1_score":sum(f1_scores)/sum(examples)}

    def get_evaluate_fn(model, test_loader: DataLoader):
        '''Return an evaluation function for server-side evaluation.'''
        # The `evaluate` function will be called after every round
        def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar], # pylint: disable=unused-argument
            ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            # Update model with the latest parameters
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            
            loss, accuracy, f1_macro, f1_w,pr,rc = loss_acc_f1_M_w(model,test_loader)
            
            wandb.log({"test_accuracy": accuracy,
                       "test_f1_macro": f1_macro,
                       "test_loss": loss,
                       "test_f1_weighted": f1_w,
                       "round": server_round,
                       })
            # now log precision and recall for each class
            for j in range(num_classes):
                wandb.log({
                    "test_precision_class_"+str(j).zfill(2): pr[j],
                    "test_recall_class_"   +str(j).zfill(2): rc[j],
                    "round": server_round
                    })
            # for sake of completeness save the model on disk, the round is used to name the model and filled with 2 digits
            if not args.sweep:

                model_path_round = model_path.replace(".pth","_round_"+str(server_round).zfill(2)+".pth")
            else:
                model_path_round = model_path
            print("Saving model on disk:",model_path_round)
            torch.save(model, model_path_round)

            return loss, {'test_accuracy': accuracy, 'test_f1_M': f1_macro,'test_f1_weighted': f1_w}

        return evaluate
    # federated average as strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit = 1.0, # Percentage client used for training (1 indicates all)
        fraction_evaluate = 1,
        #min_fit_clients = 4, # min number of clients used for training
        min_evaluate_clients = args.num_clients,
        min_available_clients = args.num_clients,
        evaluate_metrics_aggregation_fn = weighted_average,
        evaluate_fn = get_evaluate_fn(neural_model,test_loader),
        on_fit_config_fn = fit_config,
        on_evaluate_config_fn = evaluate_config,
        initial_parameters= fl.common.ndarrays_to_parameters(model_parameters)
    )

    ray_num_cpus = 20
    ray_num_gpus = 1
    ram_memory = 16_000 * 1024 * 1024 * 2

    ray_init_args = {
        "include_dashboard": False,
        "num_cpus": ray_num_cpus,
        "num_gpus": ray_num_gpus,
        "_memory": ram_memory,
        "_redis_max_memory": 100000000,
        "object_store_memory": 100000000,
        "log_to_driver": True,
    }
    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(arguments=args,
                                num_classes=num_classes,
                                num_features=num_features,
                                model=neural_model),
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        #client_resources=client_resources,
        ray_init_args=ray_init_args,
    )
    run.finish()
    # Save model with weights and dictionary of parameters on disk within the name
    # of the run but only if the sweep is not running
    if not args.sweep:
        torch.save(neural_model, model_path)
        print(model_path)
if __name__ == "__main__":
    main()
