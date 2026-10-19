# this module is for the hyperparameters management
# it will work altogheter with wandb to store the results
import argparse
from src.utils import (create_dataloaders, train_wb, load_partition, create_dataloader, load_test_partitions)
from src.NeuralNetwork import ModularNet

import wandb

def get_class_weights(dataloader):
    '''compute the class weights'''
    # get the labels
    labels = dataloader.dataset.tensors[1]
    # get the number of samples per class
    n_samples_per_class = labels.sum(axis=0)
    # return the class weights
    # as the ratio between the number of samples per class and the total number of samples
    return 1-(n_samples_per_class/labels.shape[0])

if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='adult_TargEncoded', help='Name of the dataset to use')
    # add the argument for the # of layers, #hidden units, learning rate, and batch size
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_units', type=int, default=64, help='Number of hidden units')
    parser.add_argument('--layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout value')
    parser.add_argument('--class_weights', type=bool, default=True, help='Use class weights')
    parser.add_argument('--client_id', type=int, default=3, help='Client ID')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum value')
    parser.add_argument('--num_clients', type=int, default=8, help='Number of clients')
    parser.add_argument('--alpha', type=float, default=0.99, help='Alpha value for the Dirichlet distribution')
    # parse the arguments
    args = parser.parse_args()
    # get the dataset
    distribution = 'iid' if args.alpha > 1.0 else 'imb_'+str(args.alpha)
    #dataset_folder = os.path.join('Experiments',args.dataset_name)
    tr_set,te_set = load_partition(args.dataset_name, args.client_id, args.num_clients,distribution)
    train_loader, test_loader, _ = create_dataloaders(trainset=tr_set,testset=te_set, batch_size = args.batch_size)

    # compose name for wandb
    wandb_name = args.dataset_name + "_"+str(args.num_clients).zfill(3)+"_"+str(args.alpha)+"_partition_"+str(args.client_id).zfill(3)
    class_weights = get_class_weights(train_loader) if args.class_weights else None
    # Get basic info on dataset
    num_features = train_loader.dataset.tensors[0].shape[1]
    num_classes = test_loader.dataset.tensors[1].shape[1]
    print("Number of features:", num_features)
    print("Number of classes:", num_classes)
    wandb.login()
    # Instanciate Network
    model = ModularNet(num_features, layers = [args.hidden_units]*args.layers, num_targets = num_classes, dropout = args.dropout)
    # Train the model

    train_wb(model, train_loader, test_loader, lr=args.learning_rate,model_name=wandb_name,
        momentum = args.momentum,
        epochs=30,
        partition_id=args.client_id,
        device="cpu", class_weights=class_weights)
