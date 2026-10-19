#!/bin/bash

# Activate the virtual environment
#source "../FLXAI/bin/activate"
# Define the dataset names
dataset_names=("adult_TargEncoded") #"one_hot_Monks-3") 
# Iterate over the dataset names
for dataset_name in "${dataset_names[@]}"; do
    # Define the set of integers
    clients=(12 16)

    # Iterate over the clients
    for num_clients in "${clients[@]}"; do
        # START SCRIPT
        # args: dataset_name, num_clients, alpha, num_rounds
        echo "./StartFederated.sh $dataset_name $num_clients 0.99 30"
        eval "./StartFederated.sh $dataset_name $num_clients 0.99 30"
        wait
        #wait
    done
done
