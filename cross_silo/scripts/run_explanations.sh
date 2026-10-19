#!/bin/bash
# Activate the virtual environment
#source "../FLXAI/bin/activate"
# Define the dataset names
dataset_names=("adult_TargEncoded")
dataset_names=("diamond_1_hot")
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
    clients=(4 8 16)
    distributions=("iid" "imb_5" "imb_0.99") #  "0.5" "0.99" "100000")
    for distribution_name in "${distributions[@]}"; do
        alpha=$(get_alpha $distribution_name)
        # Iterate over the clients
        client_count=0
        for num_clients in "${clients[@]}"; do
            num_clients_padded=$(printf "%03d" $num_clients)
            log_server_filename="logs/${dataset_name}_${num_clients_padded}_svr_${distribution_name}.log"
            for ((i=0; i<=num_clients; i++)); do
                if [ $i -ne 0 ]; then
                    client_id_padded=$(printf "%03d" $i) 
                else
                    client_id_padded="svr"
                fi
                
                log_file_name="logs/${dataset_name}_${num_clients_padded}_${client_id_padded}_${distribution_name}.log"
                # START SCRIPT
                # args: dataset_name, num_clients, alpha, num_rounds
                fscommand="python3 create_shap_explanation.py --dataset_name $dataset_name --client_id $i --num_client $num_clients --alpha $alpha > $log_file_name 2>&1 &"
                echo "          "$fscommand
                #eval $fscommand
                ((client_count++))
                wait
                ## when the number of clients is 32 or more, we need to wait for some clients to finish before starting new ones
                #if [ $num_clients -ge 32 ]; then
                #    if [ $client_count -ge 17 ]; then
                #        echo "waiting for next client to finish"
                #        wait # wait for all the clients to finish
                #        client_count=0
                #        echo "Ended"
                #    fi
                #fi
            done
            wait # wait for all the clients to finish
            echo "Ended"
        done
        
    done
done