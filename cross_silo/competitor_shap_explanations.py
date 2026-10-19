''' This file is used to create the shap explanations for the clients
it is not used directly in the federated learning process, but it is used coherentely 
using the same dataset and the same model as the one used in the federated learning process,
all the previous results have been saved in the folder Experiments, so this file is used only
to create the shap explanations for the clients and the server, and save them in the folder'''
import os
import time

import numpy as np
from sympy import comp
import torch
import torch.utils
# utilities for not having included the shap explainer before
from src.utils import (load_partition, create_dataloaders,
                        create_dataloader, load_test_partitions, load_train_partitions)
from src.fl_parser import fl_client_parser, compose_paths
from src.dataset_utils import ordinal_encoding, cap_dataset
from src.NeuralNetwork import ModularNet
from src.shap_explanations import get_shap_explanations
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_centroids(dt_name, num_clients, distribution='iid'):
    """Load the centroids from the file"""
    # Load the centroids from the file
    dt_name = "dataset_"+("0" if dt_name == "adult_TargEncoded" else
                          "1" if dt_name == "diamond_1_hot" else
                          "2" if dt_name == "dry_bean" else "")
    alpha = "iid" if distribution == 'iid' else str(distribution.split('_')[-1])
    centroids_path = os.path.join("flexai_clusters_results",
                                  dt_name+"_dist_"+alpha+"_clients_"+str(num_clients).zfill(3),
                                  "results",
                                  "global_centroids_round_30.npz")
    if not os.path.exists(centroids_path):
        raise FileNotFoundError(f"Centroids file not found: {centroids_path}")
    
    centroids = np.load(centroids_path)
    feats = centroids['V_global']
    print("Loaded centroids from", centroids_path)
    print("Centroids shape:", feats.shape)
    # create random labels for the features
    labs = np.zeros((feats.shape[0], 1))
    return feats, labs

def load_sampled_points(dt_name_str,num_clients,distribution='iid'):
    """Load the sampled points from the file"""
    # Load the centroids from the file
    dt_name = "dataset_"+("0" if dt_name_str == "adult_TargEncoded" else
                          "1" if dt_name_str == "diamond_1_hot" else
                          "2" if dt_name_str == "dry_bean" else "")
    alpha = "iid" if distribution == 'iid' else str(distribution.split('_')[-1])
    sampled_path = os.path.join("flexai_all_hist_counts_results",
                                  dt_name+"_dist_"+alpha+"_clients_"+str(num_clients).zfill(3),
                                  "results_fed_hist_counts",
                                  "_".join(["global_common",
                                            dt_name_str,
                                            str(num_clients).zfill(3),
                                            distribution,
                                            "040"]
                                           )+".npz"
                                  )
    if not os.path.exists(sampled_path):
        raise FileNotFoundError(f"Sampled points file not found: {sampled_path}")
    
    sampled = np.load(sampled_path)
    print(sampled.files)
    feats = sampled['X']
    print("Loaded sampled points from", sampled_path)
    print("Sampled points shape:", feats.shape)
    # create random labels for the features
    labs = np.zeros((feats.shape[0], 1))
    return feats, labs


if __name__=='__main__':
    # This is an example of how to use the functions in this file
    # parse the arguments and find the name of the dataset, the client_id and the subfolder
    # experiment like 008_iid
    # python create_fast_shap_explanation.py --dataset_name Iris --client_id 3 --num_client 6 --alpha 0.5
    parser = fl_client_parser()
    args = parser.parse_args()
    print("parsed arguments")
    dt_name = args.dataset_name
    client_id = args.client_id
    client_id = -1 if client_id <= 0 else client_id
    args.explainer_name = "shap"
    # the distribution should be changed with the value of alpha
    # maybe I can keep it this way and if alpha is >.5 use the tag noniid
    # and if alpha is <=.5 use the tag imb
    distribution = 'iid' if args.alpha > 110.0 else 'imb_'+str(args.alpha)
    num_clients = args.num_clients

    subfolder_experiment = str(num_clients).zfill(3)+"_"+distribution
    # define the path of the explanations
    exp_path = os.path.join("Experiments",dt_name,subfolder_experiment)
    # define the path of the black box model using the client_id (if client ID is <=0 is the server)

    # Load the background data loader
    print("Loading data")
        
    tr_set = load_train_partitions(args.dataset_name,
                                        args.num_clients,
                                        dataset_distribution=distribution)
    num_classes = tr_set.tensors[1].shape[1]
    # Get the V_global features from the file that contains the centroids
    # or in case of sampled points, use the SHAP_FL approach
    comp = "Sampled"  # "Sampled" "Cmeans"
    if comp == 'Cmeans':
        comp_name = "COMP_shap"
        feats, labs = load_centroids(dt_name, num_clients, distribution=distribution)
    if comp == 'Sampled':
        comp_name = "competitor_shapfl"
        feats, labs = load_sampled_points(dt_name, num_clients, distribution=distribution)

    tr_set = torch.utils.data.TensorDataset(torch.Tensor(feats),torch.Tensor(labs))
    train_loader,_ = create_dataloader(cap_dataset((tr_set.tensors[0],
                                                        tr_set.tensors[1]),
                                                        6000),
                                            batch_size=128, shuffle=True)
    # Get basic info on dataset
    num_features = train_loader.dataset.tensors[0].shape[1]
    num_classes = train_loader.dataset.tensors[1].shape[1]

    print("Number of features:", num_features)
    print("Number of classes:", num_classes)
    print(client_id)
    model_path,explanation_path,_ = compose_paths(args=args,
                                                  client_id=args.client_id)
    # from the model_path string get the folder
    # then seek inside the folder for the most recent file with .ph extension
    # and use it as the model path. The model has the name of the wandb run
    explanation_path = explanation_path.replace("E_S_shap",comp_name)
    print("explanation path:",explanation_path)

    dir_path = os.path.dirname(model_path)
    models = [model_n for model_n in os.listdir(dir_path) if model_n.endswith("model.pth")]
    print("models found:",models)
    model_path = os.path.join(dir_path,models[-1])
    print("Loading model:",model_path)
    # Load the black box model
    neural_model = torch.load(model_path)

    # Load representative samples as background data
    bg_loader = train_loader
    '''THE samples are now shared across the experiments so are in the same folder'''
    explanation_set = np.load(os.path.join("Experiments",dt_name,"explanation_set.npz"))
    # create the dataloader transforming the dataset in a tensor dataset
    explan_dataset = torch.utils.data.TensorDataset(torch.Tensor(explanation_set['features']),
                                                    torch.Tensor(explanation_set['labels']))
    loader_explain, _ = create_dataloader(explan_dataset,batch_size=32, shuffle=False)
    print("Loaded explanation set")
    # Check if the explanation has already been computed
    if os.path.exists(explanation_path):
        print("Shap explanation already computed for competitor")
    else:
        print("Shap explanation not computed for competitor, computing it now")
    
    # Get the shap explanations
    print("Getting shap explanations")
    

    # this time is not only the time of the explanation, but also the time of the training
    # of the surrogate model and the explainer model (which is a one time cost)
    start = time.time()
    shap_values = get_shap_explanations(black_box = neural_model,
                                               background_data_loader = train_loader,
                                                explain_partition = loader_explain)
    elapsed = time.time() - start

    # save the explanations
    time_path = explanation_path+"_time.txt"
    print("saving in",time_path,"the time")
    with open(time_path,'w+',encoding='utf-8') as file_name:
        for line in [elapsed]:
            file_name.write(str(line)+"\n")
    # Save the shap values
    np.savez(explanation_path, shap_values = shap_values)
    print("Saved shap explanation matrix for competitor in",explanation_path)
    print("shap explanation took",elapsed,"seconds (minutes:",np.round(elapsed/60),")")
