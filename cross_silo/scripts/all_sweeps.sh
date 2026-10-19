# run a sweep by calling sweep_federated.sh
# define the alpha values
alphas=(0.99 100000)
# define the number of partitions
num_clients=(12)
# define the dataset names
dataset_names=("diamond_1_hot" "dry_bean")

# if exist an argument then its a test:
# alphas is only 1 element, num_clients is 4
if [ $# -eq 1 ]; then
  alphas=(0.4)
  num_clients=(2)
  dataset_names=("one_hot_Monks-3")
fi

# iterate over the dataset names
# iterate over the alpha values
# iterate over the number of partitions
# and run the sweep_federated.sh having as arguments the dataset name
# the number of partitions and the alpha value
for dataset_name in "${dataset_names[@]}"
do
  for alpha in "${alphas[@]}"
  do
    for num_client in "${num_clients[@]}"
    do
      # run sweep_federated.sh
      echo ./sweep_federated.sh $dataset_name $num_client $alpha
      eval ./sweep_federated.sh $dataset_name $num_client $alpha
      wait
    done
  done
done
