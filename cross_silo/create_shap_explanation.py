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
                        create_dataloader, load_test_partitions, load_train_partitions)
from src.fl_parser import fl_client_parser, compose_paths
from src.dataset_utils import ordinal_encoding, cap_dataset
from src.NeuralNetwork import ModularNet
from src.shap_explanations import get_shap_explanations
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def get_representative_samples_france(dataset,samples_per_class=5):
    '''this function is used to get the representative samples of the dataset. but in a different way.
    First there is a clustering of all the data. The best k is retrieved using the elbow method.
    Then, for each cluster we order the samples by the distance from the centroid in 4 bins.
    Then we take some samples of each bin as the representative sample of the cluster, (in order to have the label)'''
    print(dataset.tensors[1].shape,"shape of the labels")
    num_classes = dataset.tensors[1].shape[1]
    representative_samples = np.empty((0,dataset.tensors[0].shape[1]))
    labels = np.empty((0,num_classes))
    # create the kmeans object
    inertias = []
    max_K = min(12,dataset.tensors[0].shape[0])
    for i in range(2,max_K):
        kmeans = KMeans(n_clusters=i, random_state=42,n_init=3)
        kmeans.fit(dataset.tensors[0])
        # get the centroids of the clusters
        inertias.append(kmeans.inertia_)

    # coupute the best k
    inertias_diff = np.diff(inertias)
    # find the last highest difference
    k = inertias_diff[::-1].argmax()
    k = len(inertias) - k
    print("best k:",k)
    # plot the elbow plot
    '''plt.plot(range(2,max_K),inertias)
    plt.title("Elbow plot")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    # plot the best k
    plt.scatter(k,inertias[k-2],c='r')
    # plot the diff of the inertias
    plt.plot(range(2,max_K-1),inertias_diff,label="diff of inertias")
    plt.grid()
    plt.legend()
    # put the x in log scale
    plt.yscale('log')

    # save the plot
    plt.savefig("elbow_plot.png")
    input("press enter to continue, elbow_plot.png")
    '''
    kmeans = KMeans(n_clusters=k, random_state=42,n_init=3)
    kmeans.fit(dataset.tensors[0])
    # get the centroids of the clusters
    centroids = kmeans.cluster_centers_
    # get the samples that are mapped to each cluster
    labels = kmeans.labels_
    representative_labels = np.empty((0,num_classes))
    for cluster in range(k):
        # get the samples of the cluster
        cluster_samples = dataset.tensors[0][labels==cluster,:]
        # get the distances of the samples from the centroid
        distances = np.linalg.norm(cluster_samples-centroids[cluster],axis=1)
        # order the samples in 4 bins using again KMeans, but on the distances
        kkkk = min(4,cluster_samples.shape[0])
        kmeans = KMeans(n_clusters=kkkk, random_state=42,n_init=3)
        kmeans.fit(distances.reshape(-1,1))
        ordered_samples = cluster_samples[np.argsort(kmeans.labels_)]

        
        # divide the samples in 4 bins
        bins = [ordered_samples[kmeans.labels_==i] for i in range(4)]
        # plot the number of samples in each bin
        '''plt.clf()
        plt.bar(range(4),[len(b) for b in bins])
        plt.title("Number of samples in each bin")
        plt.xlabel("Bins")
        plt.ylabel("Number of samples")
        plt.grid()
        # for the xticks use the mean of the distance of the samples in each bin
        plt.xticks(range(4),["D"+str(jj)+"\n"+str(np.mean(np.linalg.norm(b-centroids[cluster],axis=1)).round(2)) for jj,b in enumerate(bins)])
        # save the figure
        plt.savefig("Bins_"+str(cluster).zfill(2)+".png")
        plt.close()
        input("press enter to continue")
        '''
        original_labels_of_cluster = dataset.tensors[1][labels==cluster,:]
        
                                            
        # take the first sample of each bin, and its relative label
        for d_i in range(4):
            # if there are no samples in the bin, skip it
            if len(bins[d_i]) == 0:
                continue
            # choose a random sample from the bin
            rnd_sample = 0#np.random.choice(bins[d_i])
            representative_samples = np.vstack((representative_samples,bins[d_i][rnd_sample]))
            representative_labels = np.vstack([representative_labels,
                                                original_labels_of_cluster[rnd_sample,:]])
                                    
    print(representative_samples.shape,representative_labels.shape)
    return representative_samples,representative_labels
    
def get_representative_samples(dataset,samples_per_class=5):
    print(dataset.tensors[1].shape,)
    num_classes = dataset.tensors[1].shape[1]
    representative_samples = np.empty((0,dataset.tensors[0].shape[1]))
    labels = np.empty((0,num_classes))
    for class_value in range(num_classes):
        sub_set = dataset.tensors[0][torch.max(dataset.tensors[1],axis=1)[1]==class_value,:]
        if len(sub_set) < samples_per_class:
            print("not enough samples for class",class_value)
            continue
        print(sub_set.shape)
        clusterer = KMeans(n_clusters=samples_per_class,random_state=42,n_init=3)
        clusterer.fit(sub_set)
        representative_samples = np.vstack((representative_samples,clusterer.cluster_centers_))
        labels = np.vstack([labels,
                                np.tile(np.eye(num_classes)[class_value],
                                        samples_per_class).
                                        reshape(samples_per_class,num_classes)])
    return representative_samples,labels
#rep_samp = get_representative_samples(tr_set, samples_per_class=2)
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
    
    

    if client_id > 0:
        tr_set,te_set = load_partition(args.dataset_name, args.client_id,
                                       args.num_clients,distribution)
        num_classes = tr_set.tensors[1].shape[1]
        # on average we want a total of 10 exemplars
        # so we want 2 exemplars per class if we have 5 classes
        # and 1 exemplar per class if we have 10 classes
        # so we want 10/num_classes exemplars per class
        feats, labs = get_representative_samples_france(tr_set, samples_per_class=30//num_classes)
        tr_set = torch.utils.data.TensorDataset(torch.Tensor(feats),torch.Tensor(labs))
        train_loader, test_loader, _ = create_dataloaders(trainset=tr_set,testset=te_set)
    else: # this is the server
        tr_set = load_train_partitions(args.dataset_name,
                                        args.num_clients,
                                        dataset_distribution=distribution)
        num_classes = tr_set.tensors[1].shape[1]
        feats, labs = get_representative_samples_france(tr_set, samples_per_class=30//num_classes)
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
    # load the concatenated test set of all clients
    #partitions_te_set = load_test_partitions(args.dataset_name,
    #                                         args.num_clients,
    #                                         dataset_distribution=distribution)
    # cap the dataset to 2000 samples and create the dataloader
    #loader_explain, _ = create_dataloader(ordinal_encoding(
    #                                        cap_dataset((partitions_te_set.tensors[0],
    #                                                    partitions_te_set.tensors[1]),
    #                                                    2000)
    #                                        ),batch_size=32, shuffle=False)
    
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
    print("Saved shap explanation matrix for ",("client "+str(client_id) if client_id > 0 else "server"),"in",explanation_path)
    print("shap explanation took",elapsed,"seconds (minutes:",np.round(elapsed/60),")")
