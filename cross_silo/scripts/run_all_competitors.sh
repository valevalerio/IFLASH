#!/bin/bash
# Activate the virtual environment
#source "../FLXAI/bin/activate"
# Define the dataset names
dataset_names=("diamond_1_hot" "dry_bean")
dataset_names=("adult_TargEncoded")
# done are compas_TargEncoded, adult_TargEncoded, fico,dry_bean, covertype, diamond_1_hot
# Iterate over the dataset names

# this function takes a string and retrive the alpha value: 
# for "iid" alpha is 100000, for "imb_0.99" alpha is 0.99 and so on. 
# Use the "_" to split, if the string is not "iid"
function get_alpha {
    IFS='_' read -ra ADDR <<< "$1"
    if [ "${ADDR[0]}" == "iid" ]; then
        echo "100000"
    else
        echo "${ADDR[1]}"
    fi
}

cat banner.txt
for dataset_name in "${dataset_names[@]}"; do
    # Define the set of integers
    clients=(12 16)
    distributions=("iid" "imb_0.99" imb_5.0) #  "0.5" "0.99" "100000")
    for distribution_name in "${distributions[@]}"; do
        alpha=$(get_alpha $distribution_name)
        # Iterate over the clients
        client_count=0
        for num_clients in "${clients[@]}"; do
            num_clients_padded=$(printf "%03d" $num_clients)
            log_server_filename="logs/${dataset_name}_${num_clients_padded}_svr_${distribution_name}.log"
            
            log_file_name="logs/${dataset_name}_${num_clients_padded}_competitor_${distribution_name}.log"
            # START SCRIPT
            # args: dataset_name, num_clients, alpha, num_rounds
            fscommand="python3 competitor_shap_explanations.py --dataset_name $dataset_name --client_id -1 --num_client $num_clients --alpha $alpha > $log_file_name 2>&1 &"
                # check if the output file exists, its path is within the $dataset_name/$num_clients(zero padded 3 digits)_$distribution_name/shap/shap_C_$i_time.txt if it is a client and i is zero for the server, the i has to be padded to 3 digits
                #echo "Log file name: $log_file_name"
                # check if the ouput exist
                
            output_file="Experiments/$dataset_name/${num_clients_padded}_${distribution_name}/shap/competitor_shapfl_time.txt"
            if [ -f $output_file ]; then
                echo "Output file exists: $output_file"
                continue
            else
                echo "Output file does not exist: $output_file"
                echo "Running: $fscommand"
                echo $fscommand
                eval $fscommand
            fi
                
                
            wait
            echo "Ended"
        done
        
    done
done