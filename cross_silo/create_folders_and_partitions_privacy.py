''' This module contains:
    the subroutines to create the subdirectories for the experiments.
    the wrapper to create the partitions of the dataset their relative plots
        (ensuring that the partitions are computed only once and saved).
    Starts simulation of the federated learning.
    Computes the explanations of the models.
    '''
import os
import argparse
import numpy as np
from sklearn.cluster import KMeans
# from src.utils import create_partitions
from src.dataset_utils import load_named_dataset, join_datasets, split_data
from src.fl_parser import simulation_parser
from src.wandb_analysis import plot_partitions
# pylint: disable=wrong-import-position
from FederatedDataset.FederatedDataset.PartitionTypes.iid_partition import IIDPartition
from FederatedDataset.FederatedDataset.PartitionTypes.non_iid_partition_nodes import NonIIDPartitionNodes # pylint: disable=line-too-long



# pylint: disable=line-too-long
# create the subdirectories for the experiments
def check_create_dir(path):
    '''Check if a directory exists, if not create it'''
    if not os.path.exists(path):
        # log the creation of the directory
        print("Creating directory: ",path)
        os.makedirs(path)

def check_if_file_exists(path):
    '''Check if a file exists'''
    return os.path.exists(path)
# pylint: disable=consider-using-f-string
def exactly_k_digits(list_of_nums):
    '''return the list of formatted floats with k digits as string'''
    return ", ".join("{:.3f}".format(float_number) for float_number in list_of_nums)

def create_partitions_from(dataset,num_partitions,alpha,num_class):
    '''Create the partitions of the dataset'''
    # both the functions returns 2 dictionaries:
    # the first one contains the partition name as key and the indexes of the samples as values
    # the second one contains the partition name as key and the labels as values
    if alpha > 110:
        idxs_data,labels = IIDPartition.do_partitioning(dataset,
                            num_partitions=num_partitions,
                            total_num_classes=num_class)
    else:
        idxs_data,labels = NonIIDPartitionNodes.do_partitioning(dataset,
                            num_partitions=num_partitions,
                            total_num_classes=num_class,
                            alpha=alpha)
    # create the partitions
    partitions = []
    partitions_distr = []
    for name in idxs_data.keys():
        indices = idxs_data[name]
        classes = labels[name]
        partitions.append(((dataset.tensors[0][indices,:],classes)))
        # compute the distributions of the partitions
        partitions_distr.append(np.eye(num_class)[classes].sum(0)/classes.shape[0])
    # return the partitions and the distributions
    return partitions, partitions_distr

# function to calculate and save the representative samples of a dataset
def representative_samples_save_and_compute(folder_name,
                                            dataset,
                                            num_samples=50,
                                            num_classes=3):
    '''Compute the representative samples of the dataset and save them in the dataset folder
    as representative_samples.npz'''
    samples_per_class = num_samples//num_classes

    file_name = os.path.join(folder_name,'background_data')
    if check_if_file_exists(file_name+'.npz'):
        representative_samples = np.load(file_name+'.npz')['features']
    else:
        # compute the representative samples fixing the random seed, using kmeans
        representative_samples = np.empty((0,dataset.tensors[0].shape[1]))
        labels = np.empty((0,1))
        for class_value in range(num_classes):
            sub_set = dataset.tensors[0][dataset.tensors[1]==class_value,:]
            clusterer = KMeans(n_clusters=samples_per_class,random_state=42,n_init=3)
            clusterer.fit(sub_set)
            representative_samples = np.vstack((representative_samples,clusterer.cluster_centers_))
            labels = np.vstack((labels,class_value*np.ones((samples_per_class,1))))
            # save the representative samples
        np.savez(file_name,
                 features=representative_samples.astype(np.float32),
                 labels=labels.astype(np.float32))
    return representative_samples

# function to check and create the subdirectories for the experiments
def check_create_dir_tree(dataset_folder,num_partitions,alpha=100000):
    '''Check if the subdirectories for the experiments exist, if not create them.'''
    # check if create the subdirectories for the experiments
    experiment_folder = 'Experiments'
    check_create_dir(experiment_folder)
    exp_dt_folder = os.path.join(experiment_folder,dataset_folder)
    check_create_dir(exp_dt_folder)
    # compose name of the experiment
    # first part is the number of partitions (the same as the number of clients) formatted on 3 digits
    # second part refers to the concentration parameter of the LDA: "iid" if alpha -gt 1 "imb"
    experiment_name = str(num_partitions).zfill(3)+'_'+('iid' if (float(alpha)>110) else 'imb_'+str(alpha))
    experiment_folder = os.path.join(exp_dt_folder,experiment_name)
    check_create_dir(experiment_folder)
    print("Experiment folder:",experiment_folder)
    # create the subdirectories for the models and the explanations
    models_folder = os.path.join(experiment_folder,'models')
    check_create_dir(models_folder)
    # create the subdirectories for the explanations
    # shap
    explanations_folder = os.path.join(experiment_folder,'shap')
    check_create_dir(explanations_folder)
    # dalex
    explanations_folder = os.path.join(experiment_folder,'dalex')
    check_create_dir(explanations_folder)
    # fastshap
    explanations_folder = os.path.join(experiment_folder,'fastshap')
    check_create_dir(explanations_folder)
    # return the name of the experiment folder
    return experiment_folder
    
def create_and_save_partitions(experiment_folder,num_partitions,alpha):
    '''Create the partitions of the dataset and save them in the experiment folder'''
    experiment_folder = os.path.normpath(experiment_folder)
    print(experiment_folder)
    # retrive the name of the experiment name from the experiment folder
    experiment_name = experiment_folder.split(os.path.sep)[-1]
    # retrive the name of the dataset name 
    dataset_name = experiment_folder.split(os.path.sep)[-2]

    # create the names of the partitions

    partition_names_train = [dataset_name+'_'+experiment_name+'_train_'+str(i+1).zfill(3) for i in range(num_partitions)]
    partition_names_test = [pnt.replace('train','test') for pnt in partition_names_train]
    # check if the last partitions of the dataset have already been computed and saved as npz
    partition_path_try = os.path.join(experiment_folder,partition_names_train[-1])+".npz"
    print(partition_path_try,": Exists?",os.path.exists(partition_path_try))
    if not os.path.exists(partition_path_try):
        # log creation of the partitions
        train_set, test_set, val_set = load_named_dataset(dataset_name,val=True)
        

        # retrive the number of classes from the training set
        num_class = np.unique(train_set.tensors[1]).shape[0]
        print("Creating partitions for the dataset:",dataset_name)
        print("Pre partitioning shapes",train_set.tensors[0].shape,test_set.tensors[0].shape
              ,train_set.tensors[1].shape,test_set.tensors[1].shape)
        # save the representative samples of the dataset for the background data of the explanations
        representative_samples_save_and_compute(folder_name=experiment_folder,dataset=train_set,
                                                num_samples=50, num_classes=num_class)
        # join the training and test set to create the partitions of equal distributions
        whole_set = join_datasets(train_set,test_set)
        all_partitions, parts_distrs = create_partitions_from(whole_set,
                                                            num_partitions,
                                                            alpha,
                                                            num_class)
        #test_partitions, te_distr = create_partitions_from(test_set,num_partitions,alpha,num_class)


        # Define Names
        log_file = os.path.join(experiment_folder,'partitions_log.txt')
        log_content = "Dataset: "+dataset_name+"; Alpha: "+str(alpha)+"\n"
        log_content +="#partitions: "+str(num_partitions).zfill(3)+"\n"
        log_i = 1

        # Saving the dataset partitions
        for pn_tr,pn_te,part,part_distr in zip(partition_names_train,
                                         partition_names_test,
                                         all_partitions,
                                         parts_distrs):
            p_tr,p_te = split_data(part, 0.8)
            
            # save the partitions using numpy savez storing the features
            # in the first element and the labels in the second
            train_targets = np.eye(num_class)[p_tr[1]]
            test_targets = np.eye(num_class)[p_te[1]]
            partition_distr_train = train_targets.sum(0)/p_tr[1].shape[0]
            partition_distr_test = test_targets.sum(0)/p_te[1].shape[0]
            
            print("Saving partitions: ",pn_tr,pn_te)
            print("Having as distributions: ",partition_distr_train,partition_distr_test)
            print("with shapes",[p.shape for p in p_tr],[p.shape for p in p_te])
            # save the partition distributions in the log file
            log_content += str(log_i).zfill(3)+": ["
            log_content += exactly_k_digits(partition_distr_train)+"]\t["
            log_content += exactly_k_digits(partition_distr_test)+"] "
            log_content += "n:"+ " ".join([str(p.shape[0]) for p in p_tr])+" "
            log_content += " ".join([str(p.shape[0]) for p in p_te])
            log_i += 1
            # save the partitions
            np.savez(os.path.join(experiment_folder,pn_tr),features=p_tr[0], labels=train_targets)
            np.savez(os.path.join(experiment_folder,pn_te),features=p_te[0], labels=test_targets)
        # val_set is not used in this case but we dump it anyway
        np.savez(os.path.join(experiment_folder,'val_set'),features=val_set.tensors[0], labels=np.eye(num_class)[val_set.tensors[1]])
        # save the log file
        with open(log_file,'w+',encoding='utf-8') as log_file:
            log_file.write(log_content)
        return True

    print(num_partitions," partitions for the dataset:",dataset_name,"have already been created!")
    return False

if __name__ == '__main__':
    parser = simulation_parser()
    args = parser.parse_args()

    # check and create if the log folder exists
    check_create_dir('logs')
    # check if the subdirectories for the experiments exist, if not create them
    # after that the partitions of the dataset are created and saved in the experiment folder
    experiment_folder_name = check_create_dir_tree(args.dataset_name,args.num_clients,args.alpha)
    # load the partitions of the dataset
    create_and_save_partitions(experiment_folder=experiment_folder_name,
                               num_partitions=args.num_clients,
                               alpha=args.alpha)

    distr = "iid" if (args.alpha is None or args.alpha>110) else "imb_"+str(args.alpha)

