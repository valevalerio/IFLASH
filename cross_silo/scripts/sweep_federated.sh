# this script is used to run the sweep for the best federated model
# activate the venv
source ../FLXAI/bin/activate
# add the script folder to the path


name_of_dataset="adult_TargEncoded"
alpha=10000
num_partitions=12
# if there are no arguments passed to the script, then use the default values
if [ $# -eq 0 ]; then
  echo "No arguments supplied, using default values"
  echo "name_of_dataset: $name_of_dataset num_partitions: $num_partitions alpha: $alpha"
else
  name_of_dataset=$1
  num_partitions=$2
  alpha=$3
fi

run_sweep_and_agent () {
  # Set the SWEEP_NAME variable
  SWEEP_NAME="$1"

  echo "Running sweep $SWEEP_NAME.yaml"
  
  # Run the wandb sweep command and store the output in a temporary file
  python3 -m wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "$SWEEP_NAME.yaml" > temp_output.txt 2>&1
  # Extract the sweep ID using awk
  SWEEP_ID=$(awk '/wandb agent/{ sub(/wandb agent /, ""); print $0; }' temp_output.txt)
  # Convert the string to an array using the space character as the separator
  echo "Sweep ID: $SWEEP_ID"
  arr=($SWEEP_ID)
  SWEEP_ID=${arr[-1]}
  echo "Sweep ID: $SWEEP_ID"
  # Remove the temporary output file
  #rm temp_output.txt
  
  # Run the wandb agent command
  python3 -m wandb agent $SWEEP_ID --project "$PROJECT_NAME" --count 50
}



distribution_name="imb_"$alpha


create_folder_command="python3 create_folders_and_partitions.py --dataset "$name_of_dataset" --num_clients "$num_partitions" --alpha "$alpha" "
# python create_folders_and_partitions.py --dataset Monks-3 --num_partitions 8 --alpha 100
echo "Create folder command: $create_folder_command"

eval $create_folder_command

padded_num_partitions=$(printf "%02d" $num_partitions)
PROJECT_NAME=Sweep_${name_of_dataset}_${padded_num_partitions}
#PROJECT_NAME="FL_Sweep_$name_of_dataset_$num_partitions" # swap out globally
#project_name contain the distribution name the dataset name and the number of partitions
PROJECT_NAME=${PROJECT_NAME}_$distribution_name
#PROJECT_NAME="$PROJECT_NAME"_"$distribution_name" # swap out globally
# repeat run_sweep_and_agent "bestlocal" for i in (1..num_partitions)):
echo $PROJECT_NAME

# concatenate datasetname, num_partition and alpha to create a unique name 
i=_"$name_of_dataset"_"$num_partitions"

# copy yaml file to new file with partition number and distribution name
cp best_federated.yaml best_federated"$i".yaml
# replace the total_clients string in yaml file with num_partitions
sed -i "s/total_clients/$num_partitions/g" best_federated"$i".yaml
# replace the alpha_distribution with alpha
sed -i "s/alpha_distribution/$alpha/g" best_federated"$i".yaml
# replace name_of_dataset with dataset name
sed -i "s/name_of_dataset/$name_of_dataset/g" best_federated"$i".yaml
# replace python with python3
sed -i "s/python/python3/g" best_federated"$i".yaml
# run sweep and agent
run_sweep_and_agent "best_federated$i"
rm best_federated"$i".yaml
