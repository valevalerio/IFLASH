#!/bin/bash

# Activate the virtual environment
source "../FLXAI/bin/activate"
cat banner.txt
# Check if the number of clients argument is provided and set the variable
# Set default value for the dataset name
# alternative for multilabel dataset 
datasetName="one_hot_Monks-3"
datasetName="adult_TargEncoded"
datasetName="dry_bean"
if [ ! -z "$1" ]; then
  datasetName="$1"
fi
# Set default value for the number of clients
num_clients=2
if [ ! -z "$2" ]; then
  num_clients="$2"
fi
# Set default value for alpha else alpha_arg="--alpha $3"
# Set default value for alpha and alpha_arg
alpha=0.01
#100000
alpha_arg="--alpha $alpha"
if [ ! -z "$3" ]; then
  alpha_arg="--alpha $3"
  alpha="$3"
fi


# Set default value for the number of rounds
num_rounds=20
if [ ! -z "$4" ]; then
  num_rounds="$4"
fi

######################
# Create the folders #
######################
echo "Creating folders for the experiment with "$num_clients" clients and partitions IID and non-IID (alpha=$alpha) for the dataset $datasetName"
create_folder_command="python create_folders_and_partitions.py --dataset "$datasetName" --num_partitions "$num_clients" "$alpha_arg""
# python create_folders_and_partitions.py --dataset Monks-3 --num_partitions 8 --alpha 100
echo "Create folder command: $create_folder_command"

eval $create_folder_command

num_clients_padded=$(printf "%03d" $num_clients)
 # the -n flag is used to check if the string is not empty
  if [[ "$alpha" -ge 1 ]]; then
    distribution_name="iid"
  # else append the string "_iid"
  else
    distribution_name="imb_"$alpha
  fi
  

######################
# Execute the server #
######################
log_server_filename="logs/${datasetName}_${num_clients_padded}_svr_${distribution_name}.log"
# redirect the output of the server to a log file
server_command="python fedAvg_server.py --num_clients $num_clients --dataset_name $datasetName --num_rounds $num_rounds $alpha_arg > $log_server_filename 2>&1 &"
echo "Server command: $server_command"
# Execute the server command
eval $server_command
# Delay for some time to allow the server to start
sleep 5

# Log file name format: datasetName_numClients_clientId.log


# Execute the clients in the background and redirect output to separate log files
for ((i=1; i<=num_clients; i++)); do

  # Only for the log name
  client_id_padded=$(printf "%03d" $i)  # Pad client ID with zeros to three digits
  log_file_name="logs/${datasetName}_${num_clients_padded}_${client_id_padded}_${distribution_name}.log"

  ##############################
  # Execute the client command #
  ##############################
  client_command="python basic_client.py --client_id $i --num_clients $num_clients --dataset_name $datasetName $alpha_arg > $log_file_name 2>&1 &"
  echo "Client $i command: $client_command"
  eval $client_command
done

# Wait for all the clients and the server to finish
wait
echo "All clients and server finished execution"


