''' This file is used to create the shap explanations for the clients
it is not used directly in the federated learning process, but it is used coherentely 
using the same dataset and the same model as the one used in the federated learning process,
all the previous results have been saved in the folder Experiments, so this file is used only
to create the shap explanations for the clients and the server, and save them in the folder'''
import os
import time

import numpy as np
import torch
import torch.utils
# utilities for not having included the shap explainer before
from src.utils import (load_partition, create_dataloaders,
                        create_dataloader, load_train_partitions)
from src.fl_parser import fl_client_parser, compose_paths
from src.dataset_utils import cap_dataset
from src.shap_explanations import get_shap_explanations
from SHAP_FL import HistSampler,pad_bins,unpad_bins

def get_edges_bins(all_samples):
    minimums = np.min(all_samples, axis=0)
    maximums = np.max(all_samples, axis=0)
    # the number of bins should be a learnable parameters imho,
    # the paper does not specify how to choose it (serious shortcoming, easly fixable in future works)
    # So we set it to a fancy formula, unfairly as it consider all the global data but anyway it is not a big deal
    # for each feature the formula should be (max-min )/ (2 * (IQR) * n^(1/3))
    # and IQR is approximated from the data
    q75, q25 = np.percentile(all_samples, [75, 25], axis=0)
    iqr = q75 - q25
    n = all_samples.shape[0]
        
    # Step 1: bin width (Freedman–Diaconis)
    bin_width = 2 * iqr * n ** (-1/3)
    # Step 2: data range per dimension
    data_range = maximums - minimums
    # Step 3: number of bins
    nbins = np.zeros_like(bin_width, dtype=int)
    # Avoid division by zero for constant or near-constant features and place a 2-bin histogram
    for j,bw in enumerate(bin_width):
        if bw < 1e-10:
            # in this case a degree of diversity preserved
            nbins[j] = -1
        else:
            nbins[j] = int(np.ceil(data_range[j] / bw))
            # uniq = np.unique(all_samples[:,j])
            #if uniq.shape[0] < nbins[j]:
            #    nbins[j] = uniq.shape[0]
    nbins[nbins==-1]=max(nbins)
    # now that we have the number of bins for each feature, we can create the initial parameters dividing the range in equal parts
    bins = []
    for j in range(all_samples.shape[1]):
        edges = np.linspace(minimums[j], maximums[j], nbins[j]+1)
        bins.append(edges)
    return bins

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
    # here we try to load the binns of the whole dataset to use them in the HistSampler
    # if there are not saved bins we compute them from the training data of all clients
    # the file is in the main experiment folder, under the name dt_name+"_bins.csv"

    bins_folder = os.path.join(exp_path,"sampled_background")
    bins_path = os.path.join(exp_path,dt_name+"_bins.npz")
    if os.path.exists(bins_path):
        print("Loading bins from",bins_path)
        bins_v = np.load(bins_path)['padded_bins']
        assert bins_v.shape[0] == 2, "Bins file corrupted shape is:"+str(bins_v.shape)
        
        bins, counts = unpad_bins(padded_array=bins_v)
    else:
        # load all training partitions to compute the bins
        print("Computing bins from training partitions")
        tr_set = load_train_partitions(args.dataset_name,
                                        args.num_clients,
                                        dataset_distribution=distribution)
        num_classes = tr_set.tensors[1].shape[1]
        all_samples = tr_set.tensors[0].numpy()
        print("All samples shape:",all_samples.shape)
        bins = get_edges_bins(all_samples)
        # bins is now a list of integer arrays, one for each feature
        print("bins have shape:",[b.shape for b in bins])
        # pad the bins to save them in a csv file
        padded_bins = pad_bins(bins)
        print("Padded bins shape:", padded_bins.shape)
        assert padded_bins.shape[0] == 2, "Padded bins corrupted shape is:"+str(padded_bins.shape)
        
        print("Saving bins in",bins_path)

        np.savez(bins_path, padded_bins=padded_bins)
        bins, counts = unpad_bins(padded_array=padded_bins)


    sampler = HistSampler(bin_edges=bins,
                          counts= counts
                          )
    # the sampler is now "fixed" for all clients and server
    # we can use it to get representative samples for each client and server
    # but before we have to "train" on the local/audit data to get the histograms 
    # and the sampling probabilities
    
    if client_id > 0:
        tr_set,te_set = load_partition(args.dataset_name, args.client_id,
                                       args.num_clients,distribution)
        num_classes = tr_set.tensors[1].shape[1]
    else: # this is the server
        tr_set = load_train_partitions(args.dataset_name,
                                        args.num_clients,
                                        dataset_distribution=distribution)
        num_classes = tr_set.tensors[1].shape[1]
    
    data = tr_set.tensors[0].numpy()
    local_count = []
    for feat_idx,feature_bins in zip(range(data.shape[1]),
                                sampler.bin_edges_list):
            local_hist, _ = np.histogram(data[:, feat_idx],
                                                       bins=feature_bins,
                                                       density=False)
            local_count.append(local_hist)
    sampler.counts_list = local_count
    reference_background = sampler.sample(n_samples=40)# fixed number of samples for all clients and server

    # Get basic info on dataset
    num_features = tr_set.tensors[0].shape[1]
    num_classes = tr_set.tensors[1].shape[1]

    print("Number of features:", num_features)
    print("Number of classes:", num_classes)
    print(client_id)
    model_path,explanation_path,_ = compose_paths(args=args,
                                                  client_id=args.client_id)
    
    # the explanation_path as it is now, it would rewrite the other explanations
    print(explanation_path, "was the initial explanation path")
    res_path = os.path.dirname(explanation_path) # here we get the folder of the explainer
    res_path = os.path.dirname(res_path) # here we get the exp folder ()
    # assert exp_path==res_path, "Experiment path mismatch:"+exp_path+" vs "+res_path
    res_path = os.path.join(res_path,"sampled_shap")
    # check if the folder exists, if not create it
    os.makedirs(res_path,exist_ok=True)
    file_name = explanation_path.split(os.sep)[-1]
    # here we'll save the explanations
    explanation_path = os.path.join(res_path,file_name)
    print(explanation_path, "is the final explanation path")
    
        
    # from the model_path string get the folder
    # then seek inside the folder for the most recent file with .ph extension
    # and use it as the model path. The model has the name of the wandb run

    dir_path = os.path.dirname(model_path)
    models = [model_n for model_n in os.listdir(dir_path) if model_n.endswith("model.pth")]
    print("models found:",models)
    model_path = os.path.join(dir_path,models[-1])
    print("Loading model:",model_path)
    # Load the black box model
    neural_model = torch.load(model_path)

    # Load representative samples as background data
    bg_loader = create_dataloader(
                        torch.utils.data.TensorDataset(torch.Tensor(reference_background),
                                                       torch.zeros(reference_background.shape[0])),
                        batch_size=32,
                        shuffle=False)[0]
    print(bg_loader)
    # Load the explanation set
    explanation_set = np.load(os.path.join("Experiments",dt_name,"explanation_set.npz"))
    # create the dataloader transforming the dataset in a tensor dataset
    explan_dataset = torch.utils.data.TensorDataset(torch.Tensor(explanation_set['features']),
                                                    torch.Tensor(explanation_set['labels']))
    loader_explain, _ = create_dataloader(explan_dataset,batch_size=32, shuffle=False)
    print("Loaded explanation set")
    # Check if the explanation has already been computed
    if os.path.exists(explanation_path):
        print("Shap explanation already computed for ",("client "+str(client_id) if client_id > 0 else "server"))
    else:
        print("Shap explanation not computed for ",("client "+str(client_id) if client_id > 0 else "server"))
    
    # Get the shap explanations
    print("Getting shap explanations")
    
    # this time is not only the time of the explanation, but also the time of the training
    # of the surrogate model and the explainer model (which is a one time cost)
    start = time.time()
    shap_values = get_shap_explanations(black_box = neural_model,
                                               background_data_loader = bg_loader,
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
    print("Saved shap explanation matrix for ",("client "+str(client_id) if client_id > 0 else "server"),"in",explanation_path)
    print("shap explanation took",elapsed,"seconds (minutes:",np.round(elapsed/60),")")
