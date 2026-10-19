run_sweep_and_agent () {
  # Set the SWEEP_NAME variable
  SWEEP_NAME="$1"
  PARTITION="$2"

  echo "Running sweep $SWEEP_NAME.yaml"
  
  # Run the wandb sweep command and store the output in a temporary file
  wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "$SWEEP_NAME.yaml" >temp_output.txt 2>&1
  # Extract the sweep ID using awk
  #SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' temp_output.txt)
  SWEEP_ID=$(awk '/wandb agent/{ sub(/wandb agent /, ""); print $0; }' temp_output.txt)
  # Convert the string to an array using the space character as the separator
  echo "Sweep ID: $SWEEP_ID"
  arr=($SWEEP_ID)
  SWEEP_ID=${arr[-1]}
  echo "Sweep ID: $SWEEP_ID"
  # Remove the temporary output file
  rm temp_output.txt
  
  # Run the wandb agent command
  wandb agent $SWEEP_ID --project "$PROJECT_NAME" --count 50
}


# define num_partitions=8
num_partitions=2
# pad num_partitions with 0s as prefix


#iterate over alpha values 100000 and 0.99 and print the value
alphas=(100000 0.99)

for alpha in "${alphas[@]}"
do
  
  # if alpha >= 1: distribution_name="iid"
  # else distribution_name="imb_" + $aplha
  if [[ "$alpha" -ge 110 ]]; then
    distribution_name="iid"
  else
    distribution_name="imb_"$alpha
  fi

  num_partitions=$(printf "%03d" $num_partitions)
  name_of_dataset="one_hot_Monks-3"

  create_folder_command="python create_folders_and_partitions.py --dataset "$name_of_dataset" --num_clients "$num_partitions" --alpha "$alpha""
  # python create_folders_and_partitions.py --dataset Monks-3 --num_partitions 8 --alpha 100
  echo "Create folder command: $create_folder_command"

  eval $create_folder_command

  PROJECT_NAME=LOCAL_Sweep_${name_of_dataset}_${num_partitions}
  #PROJECT_NAME="FL_Sweep_$name_of_dataset_$num_partitions" # swap out globally
  #project_name contain the distribution name the dataset name and the number of partitions
  PROJECT_NAME=${PROJECT_NAME}_$distribution_name
  #PROJECT_NAME="$PROJECT_NAME"_"$distribution_name" # swap out globally
  # repeat run_sweep_and_agent "bestlocal" for i in (1..num_partitions)):
  echo $PROJECT_NAME
  for i in $(seq 1 $num_partitions)
  do
    # pad i with 0s as prefix
    i=$(printf "%03d" $i)
    # copy yaml file to new file with partition number
    cp bestlocal.yaml bestlocal"$i".yaml
    # replace "client_identifier" string in yaml file with partition number
    sed -i "s/client_identifier/$i/g" bestlocal"$i".yaml
    # replace the total_clients string in yaml file with num_partitions
    sed -i "s/total_clients/$num_partitions/g" bestlocal"$i".yaml
    # replace the alpha_distribution with alpha
    sed -i "s/alpha_distribution/$alpha/g" bestlocal"$i".yaml
    # replace name_of_dataset with dataset name
    sed -i "s/name_of_dataset/$name_of_dataset/g" bestlocal"$i".yaml
    # run sweep and agent
    run_sweep_and_agent "bestlocal$i" "$i"
    rm bestlocal"$i".yaml
    wait
  done
done