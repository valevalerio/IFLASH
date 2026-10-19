''' Client Implementation for Federated Learning with PyTorch and Flower'''
import time
from collections import OrderedDict
import flwr as fl

import torch
from torch.utils.data import TensorDataset
import numpy as np
# Custom module for nn definition
from src.NeuralNetwork import ModularNet
from src.utils import (create_dataloaders,train, load_partition, create_dataloader,
                       load_test_partitions,loss_acc_f1_M_w)
from src.dataset_utils import cap_dataset, ordinal_encoding, get_hus
from src.fl_parser import fl_client_parser, compose_paths
from src.shap_explanations import get_shap_explanations
from src.dalex_explanations import get_dalex_explanation
import wandb
class BasicClient(fl.client.NumPyClient):
    '''Basic client implementation'''
    def __init__(self, model, train_loader,val_loader,test_loader=None,client_name="client_0"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.client_name = client_name           
        
    def get_parameters(self, config : dict):
        '''Return the local model parameters'''
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    def set_parameters(self, parameters):
        '''Set model parameters'''
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # Load parameters into model
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config : dict):
        ''' Train and the model locally '''
        # Update local model parameters
        self.set_parameters(parameters)
        # Get hyperparameters for this round
        batch_size = config["batch_size"]
        epochs = config["local_epochs"]
        lr = config["lr"]
        momentum = config["momentum"]

        print("training for",epochs,"epochs, with batch size",batch_size)
        # Train the model using hyperparameters from config
        metrics = train(self.model, self.train_loader, epochs=epochs,class_weights=None,
                        lr=lr,momentum=momentum)
        # Return the updated model parameters and results
        return self.get_parameters(config={}), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config : dict):
        '''Evaluate parameters on the locally held test set'''
        # Update local model with global parameters
        self.set_parameters(parameters)
        # Get config values
        steps: int = config["val_steps"]

        server_round = config["server_round"]
        # Evaluate the model using the local validation set
        metrics = {"name":self.client_name}
        num_samples = len(self.train_loader.dataset)
        if self.val_loader is not None:
            loss_val, accuracy_val, f1_macro_val,f1_w_val,pr_val,rc_val = loss_acc_f1_M_w(self.model, self.val_loader, steps=steps)
            metrics["accuracy_val"] = accuracy_val
            metrics["f1_score_val"] = f1_macro_val
            metrics["loss_val"] = loss_val
            metrics["f1_w_val"] = f1_w_val
            metrics["pr_val"] = pr_val
            metrics["rc_val"] = rc_val
            print("LOSS:",str(loss_val)[:4])

        # same for the trainset
        loss_train, accuracy_train, f1_macro_train,f1_w_train,pr_train,rc_train = loss_acc_f1_M_w(self.model, self.train_loader)
        metrics["accuracy_train"] = accuracy_train
        metrics["f1_score_train"] = f1_macro_train
        metrics["loss_train"] = loss_train
        metrics["f1_w_train"] = f1_w_train
        metrics["pr_train"] = pr_train
        metrics["rc_train"] = rc_train
        metrics["server_round"] = server_round
        # same for the testset
        if self.test_loader is not None:
            loss_test, accuracy_test, f1_macro_test,f1_w_test,pr_test,rc_test = loss_acc_f1_M_w(self.model, self.test_loader)
            metrics["accuracy_test"] = accuracy_test
            metrics["f1_score_test"] = f1_macro_test
            metrics["loss_test"] = loss_test
            metrics["f1_w_test"] = f1_w_test
            metrics["pr_test"] = pr_test
            metrics["rc_test"] = rc_test
        
        return loss_train, num_samples, metrics
    def ret_model(self):
        '''Return the model'''
        return self.model

def run_client_and_explain(arguments):
    '''Run the client, save the model and the shap values'''
    # Load Dataset and dataloaders
    # dataset loadining is conditioned on the client_id
    #tr_set, te_set = load_named_dataset(args.dataset_name)
    #def load_partition(dataset_folder,partition_id,total_partitions,dataset_distribution='iid'):
    args = arguments
    distribution = 'iid' if args.alpha > 1.0 else 'imb_'+str(args.alpha)
    #dataset_folder = os.path.join('Experiments',args.dataset_name)
    tr_set,te_set = load_partition(args.dataset_name, args.client_id, args.num_clients,distribution)
    train_loader, test_loader, _ = create_dataloaders(trainset=tr_set,testset=te_set,
                                                      batch_size=48)

    # Get basic info on dataset
    num_features = train_loader.dataset.tensors[0].shape[1]
    num_classes = test_loader.dataset.tensors[1].shape[1]
    print("Number of features:", num_features)
    print("Number of classes:", num_classes)

    # Instanciate Network
    # The number of input features is calculated in advance as the nuber of classes
    # the model is different when the deataset is adult or covertype
    if arguments.dataset_name == "covertype":
        # model is a 3 layer NN with
        # layers with #units [feature_inputs, 512, 256, num_classes]
        # SELU hidden activation function, output activation function is log_softmax
        # the dropout is low (0.05)
        neural_model = CoverNet(num_features, num_classes = num_classes)
    else:
        #hu = get_hus(arguments.dataset_name)
        #neural_model = Net(num_features, hidden_units = hu, num_classes = num_classes)
        neural_model = ModularNet(input_size=num_features,
                   layers = [args.hidden_units]*args.layers,
                   num_targets=num_classes,
                   dropout=args.dropout)
    # print type of instanciated model
    print("Model type:",type(neural_model))

    # Create client
    client = BasicClient(neural_model,
                         train_loader=train_loader,
                         val_loader=test_loader,
                         client_name="client_"+str(arguments.client_id).zfill(3))
    # Start client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
    
    # FL client has finished, save the model and compute the explanations
    model_path,explanation_path,_ = compose_paths(args=arguments,client_id=arguments.client_id)

    # Save Model of client
    torch.save(neural_model, model_path)
    ##################################################
    ####    EXPLAINABILITY SIDE ######################
    ####    https://slundberg.github.io/shap/notebooks/Iris%20classification%20with%20scikit-learn.html # pylint: disable=line-too-long
    ##################################################
    neural_model = client.ret_model()

    # Load representative samples as background data
    # create the background data loader
    background_data = train_loader.dataset.tensors[0].mean(axis=0).reshape(1,-1)
    print(background_data.shape)
    background = TensorDataset(torch.tensor(background_data),
                               torch.tensor([1]))
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
    dalex_values = get_dalex_explanation(black_box = neural_model,
                                         background_data_loader = bg_loader,
                                         explain_partition = loader_explain)
    # elapsed time
    elapsed = time.time() - start
    # log time to file adding args.client_id.zfill(3)+"_time.txt" to the explanation path
    time_path=explanation_path.replace("shap","dalex")+"_"+str(args.client_id).zfill(3)+"_time.txt"
    with open(time_path,'w+',encoding='utf-8') as file_name:
        file_name.write(str(elapsed))
    print("Dalex explanation took",elapsed,"seconds")
    # Save the dalex values
    np.savez(explanation_path.replace("shap","dalex"), dalex_values = dalex_values)
    print("Saved dalex explanation matrix for client",args.client_id)

    ##################################################
    ##### SHAP EXPLANATIONS SIDE #####################
    ##################################################
    # Get the shap values
    # start timer
    start = time.time()
    shap_values = get_shap_explanations(black_box = neural_model,
                                         background_data_loader = bg_loader,
                                         explain_partition = loader_explain)
    # elapsed time
    elapsed = time.time() - start
    time_path = explanation_path+"_"+str(args.client_id).zfill(3)+"_time.txt"
    with open(time_path,'w+',encoding='utf-8') as file_name:
        file_name.write(str(elapsed))
    print("Shap explanation took",elapsed,"seconds")
    # Save the shap values
    np.savez(explanation_path, shap_values = shap_values)
    print("Saved shap explanation matrix for client",args.client_id)

if __name__ == '__main__':
    parser = fl_client_parser()
    args = parser.parse_args()
    run_client_and_explain(args)
    