''' Argument parsers for the server and the clients. '''
import argparse
import os
import pandas as pd
# pylint: disable=line-too-long

def simulation_parser():
    ''' Argument parser for the simulation. 
    all the default values are None, so that they can be set in the parser'''
    parser = argparse.ArgumentParser()
    # fixed parameters
    parser.add_argument('--dataset_name', type=str, default="one_hot_Monks-3", help='Name of the dataset to use')
    parser.add_argument('--num_clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--alpha', type=float, default=100000, help='Alpha value for the Dirichlet distribution')
    parser.add_argument('--num_rounds', type=int, default=150, help='Number of rounds of federated learning')

    # hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_units', type=int, default=64, help='Number of hidden units')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout value')
    parser.add_argument('--class_weights', type=bool, default=True, help='Use class weights')
    parser.add_argument('--momentum', type=float, default=0.5, help='Momentum value')
    parser.add_argument('--local_epochs', type=int, default=2, help='Number of epochs to use')
    parser.add_argument('--no-sweep', dest='sweep', action='store_false', help='Whether it is a sweep')
    return parser

def fl_server_parser():
    ''' Argument parser for the server. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='one_hot_Monks-3', help='Name of the dataset to use')
    parser.add_argument('--explainer_name', type=str, default='shap', help='Name of the explainer to use')
    parser.add_argument('--model_name', type=str, default='model_S.pth', help='Name of the model to save')
    parser.add_argument('--num_rounds', type=int, default=20, help='Number of rounds of federated learning')
    parser.add_argument('--num_clients', type=int, default=2, help='Number of clients to use')
    parser.add_argument('--alpha', type=float, default=0.99, help='Alpha value for the Dirichlet distribution')
    parser.add_argument('--prototype_features', type=bool, default=True, help='Whether to use prototype features')
    parser.add_argument('--target_encoding', type=bool, default=True, help='Whether to use target encoding for categorical features')
    parser.add_argument('--metric', type=str, default='Delta', help='Metric to evaluate the explanation distance')
    parser.add_argument('--hyperparameters', type=int, default=0, help='Hyperparameter combination to use, if 0, use hyperparameters')
    return parser

def fl_client_parser():
    ''' Argument parser for the client. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, required=True, help='ID of the client')
    parser.add_argument('--dataset_name', type=str, default='Iris', help='Name of the dataset to use')
    parser.add_argument('--explainer_name', type=str, default='shap', help='Name of the explainer to use')
    parser.add_argument('--model_name', type=str, default='model_C_', help='Name of the model to save')
    parser.add_argument('--hidden_units', type=int, default=64, help='Number of hidden units')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout value')
    parser.add_argument('--num_rounds', type=int, default=20, help='Number of rounds of federated learning')
    parser.add_argument('--num_clients', type=int, default=4, help='Number of clients to use')
    parser.add_argument('--alpha', type=float, default=1e10, help='Alpha value for the Dirichlet distribution')
    parser.add_argument('--prototype_features', type=bool, default=True, help='Whether to use prototype features')
    parser.add_argument('--target_encoding', type=bool, default=True, help='Whether to use target encoding for categorical features')
    
    parser.add_argument('--hyperparameters', type=int, default=0, help='Hyperparameter combination to use, if 0, don\'t use hyperparameters')
    return parser

def compose_paths(args: argparse.Namespace, client_id: int):
    ''' Compose the paths for the model and the explanation '''
    pool_size = args.num_clients
    alpha = args.alpha
    experiment_folder = 'Experiments'
    # get the dataset name from the argparser
    dataset_folder = args.dataset_name
    models_folder = 'models'
    explanation_method = args.explainer_name
    experiment_settings = str(pool_size).zfill(3)+'_'+('iid' if alpha>110 else 'imb_'+str(alpha))
    model_file_name = 'model_S.pth' if client_id == -1 else args.model_name+f"{client_id:03}"+".pth"
    explanation_name = ("E_S_"+explanation_method if client_id == -1 else explanation_method+"_C_"+f"{client_id:03}")
    # Join the folders using the OS independent operator from os.path for models and explanations
    model_path = os.path.join('.',
                              experiment_folder,
                              dataset_folder,
                              experiment_settings,
                              models_folder,
                              model_file_name)
    explanation_path = os.path.join('.',
                                    experiment_folder,
                                    dataset_folder,
                                    experiment_settings,
                                    explanation_method,
                                    explanation_name)
    print("model will be in",model_path)
    print("Explanations in",explanation_path)
    history_path = os.path.join('.',
                                experiment_folder,
                                dataset_folder,
                                experiment_settings,
                                'history.json')
    return model_path, explanation_path, history_path

def get_hyperparam(combination : int, dataset_name : str,):
    ''' Get the hyperparameters for the experiment '''
    # load the hyperparameters csv using pandas
    hyperparameters = pd.read_csv(os.path.join('Experiments',dataset_name,'hyperparameters.csv'))
    # get the hyperparameters for the combination
    hp = hyperparameters.iloc[combination]
    # return the hyperparameters as a dictionary
    print(hp.to_dict())
    return hp.to_dict()
