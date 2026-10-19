'''Utility functions for the framework.'''
import os
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, r2_score
import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb
from src.common import create_lda_partitions

print("CUDA available:","Yes" if torch.cuda.is_available() else "No")
# Locate Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("using:", DEVICE)




#############################
## DEEP LEARNING UTILITIES ##
#############################
##### FOR CLASSIFICATION ####

def train_wb(net, trainloader : DataLoader, testloader : DataLoader, lr : float,
            momentum : float = 0.5, partition_id = 0,
             class_weights=None, model_name = "model",device=DEVICE,epochs=100):
    '''Train the network net.
    Arguments:
        net: the network to train
        trainloader: the dataloader for the training set
        testloader: the dataloader for the test set
        lr: the learning rate
        batch_size: the batch size
    '''
    wandb.init(project="FL-"+model_name,
        config={"lr":lr,
            "partition":partition_id,
            "batch_size":trainloader.batch_size,
            "hidden_units":net.hidden_units,
            "layers":len(net.hidden_units),
            "dropout":net.dropout,
            "epochs":epochs,
            "class_weights":class_weights if class_weights is not None else False,
            "momentum":momentum})
    net.train()
    net.to(device)
    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            features, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                #print('[%d, %5d] loss: %.4f' %
                #    (epoch + 1, i + 1, running_loss / 10))
                wandb.log({"loss": running_loss / 10})
                running_loss = 0.0
    print('Finished Training')
    net.to("cpu")
    # test the model
    test_loss, test_acc, test_f1 = test_acc_f1(net, testloader)
    _, train_acc, train_f1 = test_acc_f1(net, trainloader)
    wandb.log({"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1,
                                       "train_acc": train_acc, "train_f1": train_f1})
    wandb.finish()
    return {}


# function to train the classifier
def train(net, trainloader : DataLoader, epochs : int, class_weights=None, lr=0.001,momentum=0.5):
    '''Train the network net.
    Arguments:
        net: the network to train
        trainloader: the dataloader for the training set
        epochs: the number of epochs to train
        class_weights: the weights for the loss function, if None, no weights are used
    '''
    net.train()
    net.to(DEVICE)
    # if class weights are provided, use them in the loss function
    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(features)
            # include the class weights in the loss function
            loss = criterion(outputs, labels)
            #loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    net.to("cpu")

    return {}

# compute the accuracy and the f1 score on a 1-output node classification problem
def test(net, testloader : DataLoader,steps=None):
    '''Validate the network on the entire test set.'''
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    evaluated_batches = 0
    # If steps is None, evaluate on the entire test set
    if steps is None:
        steps = len(testloader)
    # Disable gradient calculation
    with torch.no_grad():
        net.eval()
        net.to(DEVICE)
        for data in testloader:
            features, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            output_prob = net(features)
            loss += criterion(output_prob, labels).item()
            predicted = output_prob
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            evaluated_batches+=1
            if steps-evaluated_batches == 0:
                break
    accuracy = correct / total
    net.to("cpu")
    return loss, accuracy

def loss_acc_f1_M_w(net, testloader : DataLoader,steps=None):
    """Measure the model's performances on the a set.
    Returns the loss, accuracy, f1_macro, f1_weighted, precisions, recalls"""
    criterion = torch.nn.CrossEntropyLoss()
    all_labels, all_predictions = [], []
    loss = 0.0
    # If steps is None, evaluate on the entire test set
    evaluated_batches = 0
    if steps is None:
        steps = len(testloader)
    with torch.no_grad():
        net.eval()
        net.to(DEVICE)
        for features, labels in testloader:
            outputs = net(features.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            # Convert logits to predicted labels
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            evaluated_batches+=1
            if steps-evaluated_batches == 0:
                break
    net.to("cpu")
    num_classes = range(outputs.data.shape[1])

    all_labels = np.argmax(all_labels, 1)
    accuracy = accuracy_score(all_labels,all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro',zero_division=0,labels=num_classes)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted',zero_division=0,labels=num_classes)
    precisions = precision_score(all_labels, all_predictions, average=None,zero_division=0,labels = num_classes)
    recalls = recall_score(all_labels, all_predictions, average=None,zero_division=0,labels = num_classes)
    return float(loss), float(accuracy), float(f1_macro),float(f1_weighted),precisions,recalls

def test_acc_f1(net, testloader : DataLoader,steps=None):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    all_labels, all_predictions = [], []
    loss = 0.0
    # If steps is None, evaluate on the entire test set
    evaluated_batches = 0
    if steps is None:
        steps = len(testloader)
    with torch.no_grad():
        net.eval()
        net.to(DEVICE)
        for features, labels in testloader:
            outputs = net(features.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            # Convert logits to predicted labels
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            evaluated_batches+=1
            if steps-evaluated_batches == 0:
                break
    net.to("cpu")                
    all_labels = np.argmax(all_labels, 1)
    accuracy = accuracy_score(all_labels,all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')

    return float(loss), float(accuracy), float(f1_macro)

##### FOR REGRESSION ####
def train_regressor(net, trainloader : DataLoader, epochs : int):
    '''Train the network on the training set for a number of epochs.
    using as criterion the MSE loss function.
    and as optimizer the Adam optimizer.'''
    net.train()
    net.to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    net.to("cpu")
    return {}
# function to test the regressor
def test_mae_r2(net, testloader : DataLoader,steps=None):
    """Validate the model on the test set."""
    criterion = torch.nn.MSELoss()
    all_targets, all_predictions = [], []
    loss = 0.0
    # If steps is None, evaluate on the entire test set
    evaluated_batches = 0
    if steps is None:
        steps = len(testloader)
    with torch.no_grad():
        net.eval()
        net.to(DEVICE)
        for features, labels in testloader:
            outputs = net(features.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

            evaluated_batches+=1
            if steps-evaluated_batches == 0:
                break
        net.to("cpu")
    return (float(loss),
             float(mean_absolute_error(all_targets,all_predictions)),
               float(r2_score(all_targets,all_predictions)))


############################
#### DATASETS UTILITIES ####
############################
def create_dataloader(dataset,batch_size=32,shuffle=False):
    '''Create a single dataloader for a TensorDataset'''
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    num_examples = len(dataset)
    return dataloader, num_examples
def create_dataloaders(trainset,testset,batch_size=32):
    '''Create dataloaders for train and test sets'''
    trainloader,tr_examples = create_dataloader(trainset, batch_size=batch_size, shuffle=True)
    testloader,te_examples = create_dataloader(testset, batch_size=batch_size)
    num_examples = {"trainset" : tr_examples, "testset" : te_examples}
    return trainloader, testloader, num_examples
# function to create a partitioned dataset
def create_partitions(dt_set : tuple, num_partitions=4,alpha=1e10):
    '''Create partitioned datasets for train and test sets.
    The partitioning is done using the LDA algorithm. dt_set is a list of two tensors, the first
    containing the features and the second containing the labels.
    dt_set[0] is a tensor of shape (-1,num_features) and dt_set[1] is a tensor of shape (-1,1).
    Labels are assumed to be shaped (-1,1) and features (-1,num_features)).'''
    #otherwise the folling line will should be computed
    #y_tr = torch.max(y_tr,1)[1]

    x_tr,y_tr = dt_set[0],dt_set[1]
    #pylint: disable=line-too-long
    assert x_tr.shape[0] % num_partitions == 0, \
        "Number of partitions must be a factor of the number of examples: "+str(x_tr.shape[0])+" !% "+str(num_partitions)+"\n\tdataset has shape:"+str(x_tr.shape)
    #pylint: enable=line-too-long
    partitions, distribution = create_lda_partitions(dataset=[np.array(x_tr),
                                                              np.array(y_tr)],
                                                 accept_imbalanced = True,
                                                 concentration = alpha,
                                                 num_partitions = num_partitions)
    return partitions,distribution

# function to load a partition of the dataset
def load_partition(dataset_folder,
                   partition_id,total_partitions,
                   dataset_distribution='iid'):
    '''Load a partition of a dataset. The partitions are saved as npz files.'''
    # create the name of the train partition
    experiment_name = str(total_partitions).zfill(3)+'_'+dataset_distribution
    partition_name_tr = dataset_folder+'_'+str(total_partitions).zfill(3)+'_'+\
                    dataset_distribution+'_train_'+str(partition_id).zfill(3)+".npz"
    partition_name_te = partition_name_tr.replace('train','test')
    # load the partition
    tr_partition = np.load(os.path.join("Experiments",dataset_folder,experiment_name,
                                        partition_name_tr))
    te_partition = np.load(os.path.join("Experiments",dataset_folder,experiment_name,
                                        partition_name_te))
    # load the features and the labels
    x_tr,y_tr = tr_partition['features'],tr_partition['labels']
    x_te,y_te = te_partition['features'],te_partition['labels']
    # create the tensor dataset
    trainset = torch.utils.data.TensorDataset(torch.tensor(x_tr).float(),
                                              torch.tensor(y_tr).float())
    testset = torch.utils.data.TensorDataset(torch.tensor(x_te).float(),
                                              torch.tensor(y_te).float())
    return trainset,testset
def load_test_partitions(dataset_folder,
                   total_partitions,
                   dataset_distribution='iid'):
    '''Load all the test set of the partitions using load_partition and concatenate only the 
    test set in a single tensor dataset.'''
    # initialize the test set as empty array
    x_te,y_te = np.empty((0,0)),np.empty((0,0))
    for idx in range(total_partitions):
        _, testset = load_partition(dataset_folder,idx+1,total_partitions,dataset_distribution)
        if x_te.size:
            x_te = np.vstack((x_te,testset[:][0].numpy()))
            y_te = np.vstack((y_te,testset[:][1].numpy()))
        else:
            x_te = testset[:][0]
            y_te = testset[:][1]
    # create the tensor dataset
    testset = torch.utils.data.TensorDataset(torch.tensor(x_te).float(),
                                                torch.tensor(y_te).float())
    return testset
def load_train_partitions(dataset_folder,
                        total_partitions,
                        dataset_distribution='iid'):
    '''Load all the train set of the partitions using load_partition and concatenate only the
    train set in a single tensor dataset.'''
    # initialize the test set as empty array
    x_tr,y_tr = np.empty((0,0)),np.empty((0,0))
    for idx in range(total_partitions):
        trainset,_ = load_partition(dataset_folder,idx+1,total_partitions,dataset_distribution)
        if x_tr.size:
            x_tr = np.vstack((x_tr,trainset[:][0].numpy()))
            y_tr = np.vstack((y_tr,trainset[:][1].numpy()))
        else:
            x_tr = trainset[:][0]
            y_tr = trainset[:][1]
    # create the tensor dataset
    trainset = torch.utils.data.TensorDataset(torch.tensor(x_tr).float(),
                                                torch.tensor(y_tr).float())
    return trainset


# save the history of metrics
def save_history(history,file_path):
    '''Save the history of metrics, 
    history is a list of dictionaries.
    each element of the list is a dictionary with keys the client names
    and as value the tripet [loss,accuracy,f1_score]'''
    with open(file_path, 'w+', encoding='utf-8') as file_pointer:
        json.dump(history, file_pointer)
# load the history of metrics
def load_history(experiment_folder = 'Experiments',
                 dataset_folder = 'Iris',
                 experiment_settings = '012_iid',
                 num_clients = None):
    '''Load the history of metrics,
    history is a list of dictionaries.
    each element of the list is a dictionary with keys the client names
    and as value the tripet [loss,accuracy,f1_score]
    num_clients is the number of clients in the experiment'''
    if num_clients is None:
        num_clients = int(experiment_settings.split('_')[0])
    history_path = os.path.join(experiment_folder,
                            dataset_folder,
                            experiment_settings,
                            'history.json')
    with open(history_path,'r',encoding='utf-8') as json_file:
        history = json.load(json_file)
    return history,num_clients
# build the history datastructure for the plot
def build_history_datastructure(history,num_clients):
    '''Build the history datastructure from the history list of dictionaries.
    returns a tuple of two lists, the first is the list of centralized metrics
    the second is the list of clients metrics'''
    acc,f1_macro,loss = [],[],[]
    rounds_number = (len(history)-1)//2
    null_value = -100
    clients_acc = np.ones((rounds_number,num_clients))*null_value
    clients_f1 = np.ones((rounds_number,num_clients))*null_value
    clients_loss = np.ones((rounds_number,num_clients))*null_value

    round_idx = 0
    for federated_round in history:
        try:
            # centralized metrics
            if federated_round["round"]>0:
                acc.append(federated_round["accuracy"])
                f1_macro.append(federated_round["f1_score"])
                loss.append(federated_round["loss"])
            else:
                continue
        except KeyError:
            # client metrics
            for client_name in federated_round.keys():
                client_metrics = federated_round[client_name]
                client_id = int(client_name.split('_')[1]) -1
                clients_acc [round_idx,client_id] = float(client_metrics[1])
                clients_f1  [round_idx,client_id] = float(client_metrics[2])
                clients_loss[round_idx,client_id] = float(client_metrics[0])
                #print(client_id,end=', ')
            #print()
            #print(clients_acc[round_idx].round(2))
            round_idx+=1
    metrics_centralized = [acc,f1_macro,loss]
    metrics_clients = [clients_acc,clients_f1,clients_loss]
    return metrics_centralized,metrics_clients
# retrive the name of the experiment from the dataset name and the experiment settings
def name_by_folders(dt_name, exp_sett):
    '''Create a name for the experiment using the dataset name and the experiment settings'''
    splitted =  exp_sett.split("_")
    n_c = splitted[0]
    dis_alpha =splitted[1:]
    if len(dis_alpha) > 1:
        dis = "imb"
        alpha = dis_alpha[-1]
    else:
        dis = "iid"
        alpha = "100000"
    return dt_name+ " - "+" clients ".join([str(int(n_c)),dis])+ (" - "+alpha if dis == "imb" else "")
