'''This file contains the functions to load the datasets.
   The datasets are supposed to be returned asa TensorDataset object.
   The first element of the tuple are the training patterns, the second element are the test.'''
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_covtype
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np

# Import for the cleaning of the adult dataset
from src.discretize import EntropyDiscretizer
from src.clean_diamond import clean_diamond
from src.clean_adult import clean_adult
from src.clean_compas import clean_compas
def load_iris_dataset():
    '''Load the iris dataset and split it into train and test sets,
        return the datasets as TensorDataset objects.'''
    iris = load_iris(return_X_y=True)
    data, targets = iris
    # Split the data into train and test sets
    train_data, test_data, train_targets, test_targets = train_test_split(
        data, targets,stratify=targets, test_size=0.5, random_state=42)
    # Convert to tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_targets = torch.tensor(test_targets, dtype=torch.long)
    # Convert targets to one-hot encoding
    #train_targets = F.one_hot(train_targets).float()
    #test_targets = F.one_hot(test_targets).float()
    # Create datasets
    training_set = TensorDataset(train_data, train_targets)
    test_set = TensorDataset(test_data, test_targets)

    return training_set, test_set

def load_openfl_xai_dataset(selected_devices,dataset_name='OpenFL-XAI',
                            add_common_data = False):
    '''Load the OpenFL-XAI dataset and join it for central training,
        when the are not specified the selected devices are 0,1,2,3,4.
        return the datasets as TensorDataset objects.'''
    train_data = np.empty((0,15))
    test_data = np.empty((0,15))
    train_targets = np.empty((0,1))
    test_targets = np.empty((0,1))
    for i in selected_devices:
        train_data_file =os.path.join('Experiments',dataset_name,
                                        'col'+str(i)+ '_data',
                                        'X_train.npy')
        train_targts_file =os.path.join('Experiments',dataset_name,
                                        'col'+str(i)+'_data',
                                        'Y_train.npy')
        test_data_file = os.path.join('Experiments',dataset_name,
                                        'col'+str(i)+ '_data',
                                        'X_test.npy')
        test_targt_file = os.path.join('Experiments',dataset_name,
                                        'col'+str(i)+'_data',
                                        'Y_test.npy')
        # load the data using np.load and stack it in the final numpy array
        train_data = np.vstack((train_data, np.load(train_data_file)))
        print(np.load(train_targts_file).shape)
        train_targets = np.vstack((train_targets, np.load(train_targts_file).reshape(1,-1).T))
        test_data = np.vstack((test_data, np.load(test_data_file)))
        test_targets = np.vstack((test_targets, np.load(test_targt_file).reshape(1,-1).T))
    if add_common_data:
        # load the common data
        test_data_file = os.path.join('Experiments',dataset_name,
                                        'X_test.npy')
        test_targt_file = os.path.join('Experiments',dataset_name,
                                        'Y_test.npy')
        test_data = np.vstack((test_data, np.load(test_data_file)))
        test_targets = np.vstack((test_targets, np.load(test_targt_file).reshape(1,-1).T))
    # create the datasets
    training_set = TensorDataset(train_data, train_targets)
    test_set = TensorDataset(test_data, test_targets)

    return training_set, test_set

def load_monks3():
    ''' Load the monks-3 from csv with pandas and return the datasets as TensorDataset couples'''
    train_file =os.path.join('Experiments','Monks-3', 'monks-3.train')
    test_file = os.path.join('Experiments','Monks-3', 'monks-3.test')
    # Read the csv files
    train_data = pd.read_csv(train_file,sep=' ', header=None).iloc[: , 1:-1]
    test_data =  pd.read_csv(test_file, sep=' ', header=None).iloc[: , 1:-1]
    # Split the data into features and labels
    y_tr = train_data.iloc[:,0]
    y_te = test_data.iloc[:,0]
    train_data = train_data.iloc[:,1:]
    test_data = test_data.iloc[:,1:]
    # Convert pandas DataFrame to PyTorch Tensor
    train_data_tensor = torch.tensor(train_data.values).float()
    test_data_tensor = torch.tensor(test_data.values).float()
    #y_tr_tensor = torch.tensor([np.eye(2)[i] for i in y_tr.values]).long()
    #y_te_tensor = torch.tensor([np.eye(2)[i] for i in y_te.values]).long()
    # Crete the labels as tensor of shape (-1,1) of type long
    y_tr_tensor = torch.tensor(y_tr.values).long()
    y_te_tensor = torch.tensor(y_te.values).long()
    # Create TensorDataset
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor, y_tr_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_data_tensor, y_te_tensor)

    return train_dataset,test_dataset
def load_monks3_1hot():
    ''' Load the monks-3 from csv with pandas and return the datasets as TensorDataset couples.
    Differs from load_monks3 because it performs one-hot encoding on the features'''
    train_file =os.path.join('Experiments','Monks-3', 'monks-3.train')
    test_file = os.path.join('Experiments','Monks-3', 'monks-3.test')
    # Read the csv files
    train_data = pd.read_csv(train_file,sep=' ', header=None).iloc[: , 1:-1]
    test_data =  pd.read_csv(test_file, sep=' ', header=None).iloc[: , 1:-1]
    # Split the data into features and labels
    y_tr = train_data.iloc[:,0]
    y_te = test_data.iloc[:,0]
    train_data = train_data.iloc[:,1:]
    test_data = test_data.iloc[:,1:]
    # Convert pandas DataFrame to PyTorch Tensor
    train_data_tensor = torch.tensor(train_data.values).float()-1
    test_data_tensor = torch.tensor(test_data.values).float()-1
    # the minimum value of the categorical features should be 0

    #y_tr_tensor = torch.tensor([np.eye(2)[i] for i in y_tr.values]).long()
    #y_te_tensor = torch.tensor([np.eye(2)[i] for i in y_te.values]).long()
    # Crete the labels as tensor of shape (-1,1) of type long
    y_tr_tensor = torch.tensor(y_tr.values).long()
    y_te_tensor = torch.tensor(y_te.values).long()
    # Create TensorDataset
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor, y_tr_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_data_tensor, y_te_tensor)
    train_dataset = one_hot_encoding_features(train_dataset)
    test_dataset = one_hot_encoding_features(test_dataset)

    return train_dataset,test_dataset
def load_glass(prototipy = False):
    '''Load the glass dataset and split it into train and test sets,
    perform preprocessing and return the datasets as TensorDataset objects.'''
    glass_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
    glass_data_set = pd.read_csv(glass_url,header=None)

    whole_x = np.array(glass_data_set.iloc[:,1:-1])
    whole_y = np.array(glass_data_set.iloc[:,-1])
    whole_x = (whole_x - np.mean(whole_x,axis=0)) / np.std(whole_x,axis=0)
    whole_y[whole_y==5]=4
    whole_y[whole_y==6]=5
    whole_y[whole_y==7]=6
    #split the dataset
    x_train, x_test, y_train, y_test = train_test_split(whole_x, whole_y, test_size=0.2,
                                                        random_state=42,stratify=whole_y)
    if prototipy:
        discretizer = EntropyDiscretizer(x_train,labels = y_train,
                                         feature_names=glass_data_set.columns[1:-1],
                                         categorical_features=[])
        x_train = discretizer.discretize(x_train)
        x_test = discretizer.discretize(x_test)
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train).long()
    y_test = torch.tensor(y_test).long()
    print("trainShapes")
    print(x_train.shape,y_train.shape)
    print("testShapes")
    print(x_test.shape,y_test.shape)
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    return train_dataset,test_dataset

def load_adult(return_x_y=True,target_encoded=True, prototipy=True,val=False):
    '''Load the adult dataset and split it into train and test sets,
    perform preprocessing and return the datasets as TensorDataset objects.
       | t | proto | dataset
       | 0 |   0   | adult_1_hot - no pls
       | 1 |   0   | adult_TargEncoded - ok
       | 0 |   1   | adult_1_hot - pls no
       | 1 |   1   | adult_TargEncoded_Prototyped - yes '''
    # check if cleaning has already happened
    options =("TargEncoded" if target_encoded else "1_hot")
    options = (options+"_") if prototipy else options
    options +=("Prototyped" if prototipy else "")
    options = ("_"+options) if options else ""
    dataset_folder = "adult" + options
    adult_path = os.path.join("Experiments",dataset_folder,"adult"+options+".npz")
    # if yes skip the cleaning:
    print("loadining",adult_path)
    if not os.path.exists(adult_path):
        print("cleaning Dataset")
        clean_adult(adult_path,target_encode=target_encoded, prototipy=prototipy)
    # load the cleaned dataset using numpy
    adult_npz = np.load(adult_path)
    x_data = adult_npz['X']
    target_data = adult_npz['y']
    # convert to torch tensors
    x_train, x_test, y_train, y_test = train_test_split(x_data, target_data, random_state=42,
                                                        test_size=0.15,stratify=target_data)
    if val:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=42,
                                                        test_size=0.16,stratify=y_train)
        val_data_tensor = torch.tensor(x_val).float()
        y_val_tensor = torch.tensor(y_val).long()
        val_dataset = torch.utils.data.TensorDataset(val_data_tensor, y_val_tensor)
    train_data_tensor = torch.tensor(x_train).float()
    test_data_tensor = torch.tensor(x_test).float()
    y_tr_tensor = torch.tensor(y_train).long()
    y_te_tensor = torch.tensor(y_test).long()
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor, y_tr_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_data_tensor, y_te_tensor)
    if return_x_y:
        if val:
            return train_dataset,test_dataset,val_dataset
        return train_dataset,test_dataset
    # load the names of the features and the categorical values
    names_path = adult_path.replace(".npz","_names.txt")
    cat_values_path = adult_path.replace(".npz","_cat_values.txt")
    with open(names_path,'r',encoding='utf-8') as file_name:
        names = file_name.read().split(", ")
    with open(cat_values_path,'r',encoding='utf-8') as file_name:
        cat_values = file_name.read()
    if not val:
        return train_dataset,test_dataset,names,cat_values
    return train_dataset,test_dataset,val_dataset,names,cat_values

def load_dry_bean():
    ''' Load the dry_bean dataset and split it into train and test sets'''
    dry_bean_path = os.path.join("Experiments","dry_bean","dry_bean.npz")
    if not os.path.exists(dry_bean_path):
        dry_bean_path = os.path.join("Experiments","dry_bean","Dry_Bean_Dataset.xlsx")
        df = pd.read_excel(dry_bean_path)
        names_path = dry_bean_path[:-5]+"_names.txt"

        with open(names_path,'w',encoding='utf-8') as file_name:
            file_name.write(", ".join(list(df.columns[1:])))
        data = df[[n for n in df.columns if n !="Class"]].values.astype(float)
        labels = df["Class"].replace({class_name:v 
                                    for v,class_name in enumerate(sorted(df["Class"].unique()))}
                                    ).values.astype(int)
        dry_bean_path = os.path.join("Experiments","dry_bean","dry_bean")
        np.savez(dry_bean_path,data=data,labels=labels)
    # save the names
    else: 
        #load the npz
        dataset = np.load(dry_bean_path)
        data = dataset['data']
        labels = dataset['labels']
    data = (data - np.mean(data,axis=0)) / np.std(data,axis=0)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                            random_state=42,stratify=labels)
    train_data_tensor = torch.tensor(x_train).float()
    test_data_tensor = torch.tensor(x_test).float()
    y_tr_tensor = torch.tensor(y_train).long()
    y_te_tensor = torch.tensor(y_test).long()
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor, y_tr_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_data_tensor, y_te_tensor)
    return train_dataset,test_dataset

def load_compas(return_x_y=True,target_encoded=True, prototipy=False):
    '''Load the comaps dataset and split it into train and test sets,
    perform preprocessing and return the datasets as TensorDataset objects.
       | t | proto | dataset
       | 0 |   0   | comaps_1_hot - no pls
       | 1 |   0   | comaps_TargEncoded - ok
       | 0 |   1   | comaps_1_hot - pls no
       | 1 |   1   | comaps_TargEncoded_Prototyped - yes '''
    # check if cleaning has already happened
    options =("TargEncoded" if target_encoded else "1_hot")
    options = (options+"_") if prototipy else options
    options +=("Prototyped" if prototipy else "")
    options = ("_"+options) if options else ""
    dataset_folder = "compas" + options
    compas_path = os.path.join("Experiments",dataset_folder,"compas"+options+".npz")
    # if yes skip the cleaning:
    print("loadining",compas_path)
    if not os.path.exists(compas_path):
        print("cleaning Dataset")
        clean_compas(compas_path,targetEnc=target_encoded, prototipy=prototipy)
    # load the cleaned dataset using numpy
    compas_npz = np.load(compas_path)
    x_data = compas_npz['X']
    target_data = compas_npz['y']
    # convert to torch tensors
    x_train, x_test, y_train, y_test = train_test_split(x_data, target_data, random_state=42,
                                                        test_size=0.15,stratify=target_data)
    train_data_tensor = torch.tensor(x_train).float()
    test_data_tensor = torch.tensor(x_test).float()
    y_tr_tensor = torch.tensor(y_train).long()
    y_te_tensor = torch.tensor(y_test).long()
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor, y_tr_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_data_tensor, y_te_tensor)
    if return_x_y:
        return train_dataset,test_dataset
    # load the names of the features and the categorical values
    names_path = compas_path.replace(".npz","_names.txt")
    cat_values_path = compas_path.replace(".csv","_cat_values.txt")
    with open(names_path,'r',encoding='utf-8') as file_name:
        names = file_name.read().split(", ")
    with open(cat_values_path,'r',encoding='utf-8') as file_name:
        cat_values = file_name.read()
    return train_dataset,test_dataset,names,cat_values

def load_diamond(return_x_y=True,target_encoded=False,prototipy=False,val=False):
    '''Load the diamond dataset and split it into train and test sets'''
    options =("TargEncoded" if target_encoded else "1_hot")
    options = (options+"_") if prototipy else options
    options +=("Prototyped" if prototipy else "")
    options = ("_"+options) if options else ""
    dataset_folder = "diamond" + options
    diamond_path = os.path.join("Experiments",dataset_folder,"diamond"+options+".npz")
    print("loadining",diamond_path)
    if not os.path.exists(diamond_path):
        print("cleaning Dataset")
        clean_diamond(diamond_path,targetEnc=target_encoded, prototipy=prototipy)
    
    diamond_npz = np.load(diamond_path)
    x_data = diamond_npz['X']
    target_data = diamond_npz['y']
    # convert to torch tensors
    x_train, x_test, y_train, y_test = train_test_split(x_data, target_data, random_state=42,
                                                        test_size=0.2,stratify=target_data)
    if val:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=42,
                                                        test_size=0.16,stratify=y_train)
        val_data_tensor = torch.tensor(x_val).float()
        y_val_tensor = torch.tensor(y_val).long()
        val_dataset = torch.utils.data.TensorDataset(val_data_tensor, y_val_tensor)
    train_data_tensor = torch.tensor(x_train).float()
    test_data_tensor = torch.tensor(x_test).float()
    y_tr_tensor = torch.tensor(y_train).long()
    y_te_tensor = torch.tensor(y_test).long()
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor, y_tr_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_data_tensor, y_te_tensor)
    if return_x_y:
        if val:
            return train_dataset,test_dataset,val_dataset
        return train_dataset,test_dataset
    # load the names of the features and the categorical values
    names_path = diamond_path.replace(".npz","_names.txt")
    cat_values_path = diamond_path.replace(".csv","_cat_values.txt")
    with open(names_path,'r',encoding='utf-8') as file_name:
        names = file_name.read().split(", ")
    with open(cat_values_path,'r',encoding='utf-8') as file_name:
        cat_values = file_name.read()
    if not val:
        return train_dataset,test_dataset,names,cat_values
    return train_dataset,test_dataset,val_dataset,names,cat_values

def load_fico():
    ''' Load the fico dataset and split it into train and test sets,'''
    fico_path = os.path.join("Experiments","fico","fico.npz")

    if not os.path.exists(fico_path):
        print("First Time")
        fico_path = fico_path.replace(".npz",".csv")
        df = pd.read_csv(fico_path)


        names_path = fico_path[:-4]+"_names.txt"

        with open(names_path,'w',encoding='utf-8') as file_name:
            file_name.write(", ".join(list(df.columns[1:])))
        data = df[[n for n in df.columns if n !="RiskPerformance"]].values.astype(float)
        labels = df["RiskPerformance"].replace({'Bad': 1, 'Good': 0}).values.astype(int)
        np.savez(fico_path.replace(".csv",""),data=data,labels=labels)
    else:
        print("Load preproc")
        dataset = np.load(fico_path)
        data = dataset['data']
        labels = dataset['labels']
    data = (data - np.mean(data,axis=0)) / np.std(data,axis=0)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                            random_state=42,stratify=labels)
    train_data_tensor = torch.tensor(x_train).float()
    test_data_tensor = torch.tensor(x_test).float()
    y_tr_tensor = torch.tensor(y_train).long()
    y_te_tensor = torch.tensor(y_test).long()
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor, y_tr_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_data_tensor, y_te_tensor)
    return train_dataset,test_dataset

def load_covertype():
    ''' Load the covertype dataset and split it into train and test sets,
    perform preprocessing and return the datasets as TensorDataset objects.
    the labels are shifted by 1 to make them start from 0'''
    covertype_path = os.path.join("Experiments","covertype","covertype.npz")
    if not os.path.exists(covertype_path):
        print("Downloadining")
        dataset = fetch_covtype(data_home=os.path.join("Experiments","covertype"))
        data = dataset.data
        labels = dataset.target
        covertype_path = covertype_path[:-4]
        np.savez(os.path.join("Experiments","covertype","covertype"),data=data,labels=labels)
        names_path = covertype_path+"_names.txt"
        with open(names_path,'w',encoding='utf-8') as file_name:
            file_name.write(", ".join(dataset.feature_names))
    else:
        dataset = np.load(covertype_path)
        data = dataset['data']
        labels = dataset['labels']
    data = (data - np.mean(data,axis=0)) / np.std(data,axis=0)
    labels = labels-1
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                        random_state=42,stratify=labels)
    train_data_tensor = torch.tensor(x_train).float()
    test_data_tensor = torch.tensor(x_test).float()
    y_tr_tensor = torch.tensor(y_train).long()
    y_te_tensor = torch.tensor(y_test).long()
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor, y_tr_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_data_tensor, y_te_tensor)
    return train_dataset,test_dataset

def load_named_dataset(dataset_name,val=False):
    '''Load the dataset with the given name and return it as a TensorDataset object!
        Very important: the dataset is supposed to be returned as a TensorDataset object.'''
    # Iris
    if dataset_name == 'Iris':
        return load_iris_dataset()
    # Monks-3
    if dataset_name == 'Monks-3':
        return load_monks3()
    if dataset_name == 'one_hot_Monks-3':
        return load_monks3_1hot()

    # diamond
    if dataset_name == 'diamond_TargEncoded_Prototyped':
        return load_diamond(prototipy=True,target_encoded=True)
    if dataset_name == 'diamond_TargEncoded':
        return load_diamond(prototipy=False)
    if dataset_name == 'diamond_1_hot':
        return load_diamond(target_encoded=False,val=val)
    # compas
    if dataset_name == 'compas_TargEncoded_Prototyped':
        return load_compas(prototipy=True,target_encoded=True)
    if dataset_name == 'compas_TargEncoded':
        return load_compas(prototipy=False)
    if dataset_name == 'compas_1_hot':
        return load_compas(target_encoded=False)
    # Adult
    if dataset_name == 'adult_TargEncoded_Prototyped':
        return load_adult(prototipy=True,target_encoded=True)
    if dataset_name == 'adult_TargEncoded':
        return load_adult(prototipy=False,val=val)
    if dataset_name == 'adult_1_hot':
        return load_adult(target_encoded=False)
    # FICO
    if dataset_name == 'fico':
        return load_fico()
    if dataset_name == 'dry_bean':
        return load_dry_bean()
    # Glass
    if dataset_name == 'Glass':
        return load_glass()
    # Covertype
    if dataset_name == 'covertype':
        return load_covertype()
    raise NameError('Unknown dataset, implement the loader in dataset_utils.py')

def get_hus(dataset_name):
    '''return the number of hidden units to define the network'''
    dataset_name = dataset_name.lower()
    if 'adult' in dataset_name:
        return 16
    if "fico" in dataset_name:
        return 16
    if "diamond" in dataset_name:
        return 64
    if "compas" in dataset_name:
        return 64
    if "dry_bean" in dataset_name:
        return 64
    return 64

# transform the targets of a TensorDataset in one-hot encoding using the F.one_hot function
def one_hot_encoding(dataset):
    '''Transform the targets of a TensorDataset in one-hot encoding using the F.one_hot function'''
    # get the targets
    targets = dataset.tensors[1]
    # transform the targets in one-hot encoding
    targets = F.one_hot(targets.long()).float()
    # create the new dataset
    new_dataset = TensorDataset(dataset.tensors[0],targets)
    return new_dataset
def one_hot_encoding_features(dataset):
    '''Transform the targets of a TensorDataset in one-hot encoding using the F.one_hot function'''
    # get the targets
    features = dataset.tensors[0]
    # apply the F.one_hot to each feature column of the dataset
    one_hot_features = torch.empty((features.shape[0],0))
    for i in range(features.shape[1]):
        one_hot_features = torch.hstack([one_hot_features,F.one_hot(features[:,i].long()).float()])
    # create the new dataset
    new_dataset = TensorDataset(one_hot_features,dataset.tensors[1])
    return new_dataset
# trasform the targets of a TensorDataset in ordinal encoding
def ordinal_encoding(dataset):
    '''Transform the targets of a TensorDataset in ordinal encoding'''
    # get the targets
    targets = dataset.tensors[1]
    # transform the targets in ordinal encoding using torch.max
    targets = torch.max(targets,1)[1]
    # create the new dataset
    new_dataset = TensorDataset(dataset.tensors[0],targets)
    return new_dataset

# numpy array unique values
def unique_values(array):
    '''Return the unique values of a numpy array'''
    return np.unique(array)

# function to join train and test tensor datasets
def join_datasets(train_dataset,test_dataset):
    '''Join train and test tensor datasets'''
    # get the data and the targets of the train dataset
    train_data = train_dataset.tensors[0]
    train_targets = train_dataset.tensors[1]
    # get the data and the targets of the test dataset
    test_data = test_dataset.tensors[0]
    test_targets = test_dataset.tensors[1]
    # join the data and the targets
    data = torch.vstack((train_data,test_data))
    targets = torch.concatenate((train_targets,test_targets))
    # create the new dataset
    new_dataset = TensorDataset(data,targets)
    return new_dataset

# function to split a dataset in train and test
def split_data(dataset,train_size=0.8):
    '''Split a dataset in train and test'''
    # get the data and the targets of the dataset
    data = dataset[0]
    y = dataset[1]
    # split the dataset

    # check if the least represented class is more than 1 sample
    # if yes use stratify

    train_data, test_data, train_targets, test_targets = train_test_split(
        data, y, test_size=1-train_size, random_state=42)

    return (train_data, train_targets), (test_data, test_targets)

# function to cap the number of samples in a dataset
def cap_dataset(dataset,max_samples=2000):
    '''Cap the number of samples in a dataset'''
    # get the data and the targets of the dataset
    # shuffle the dataset before capping
    torch.manual_seed(42)
    indices = torch.randperm(dataset[0].shape[0])[:max_samples]

    data = dataset[0]
    targets = dataset[1]
    # cap the number of samples
    data    = dataset[0][indices]
    targets = dataset[1][indices]
    # create the new dataset
    new_dataset = TensorDataset(data,targets)
    return new_dataset
    