"this script is used to run the experiments for the paper"
"It iterates over 12 and 16 partitions, with alpha 100000, 5.0 and 0.99 for the adult_TargEncoded dataset"

# Define the dataset names
dataset_names=("dry_bean" "diamond_1_hot")
# Iterate over the dataset names
for name_of_dataset in "${dataset_names[@]}"; do

    # Define the set of integers
    clients=(12 16)
    alphas=(100000 5.0 0.99)
    for num_partitions in "${clients[@]}"; do
        for alpha in "${alphas[@]}"; do
            
            create_folder_command="python3 create_folders_and_partitions.py --dataset "$name_of_dataset" --num_clients "$num_partitions" --alpha "$alpha" "
            
            echo "Create folder command: $create_folder_command"
            eval $create_folder_command
            # check if the explanation set is created and the partitions are created
            ls -l Experiments/$name_of_dataset/explanation_set.npz
            

            wait
        done
    done
done