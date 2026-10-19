import numpy as np
import json
import os
# function to iterate over all the files in the directoy_path directory and print the shape of the data

# map function to print an array of number as a char, the lowest is the number the lightest is the char
def print01(x, n=100):
    symbs = np.array([" ", "▁", "▂", "▃", "▅", "▆", "█", "▓"])
    x=x/x.sum(axis=0)
    x = x.reshape(-1)
    #print(x*100)
    bins = symbs[(x*(len(symbs)-1)).astype(int)]
    print(*bins,n,end = " ")
    #print((x*9).astype(int))
    #print(x.round(2)*100)
def check_partitions(directory_path):
        is_npz = lambda x: x.endswith('.npz')
        total_samples = 0
        for f in sorted(os.listdir(directory_path)):
            if is_npz(f) and "train" in f:
                data = np.load(os.path.join(directory_path, f))
                #print(f, data["features"].shape, data["labels"].shape)
                #print(*data["labels"].mean(axis=0).round(3))
                print01(data["labels"].mean(axis=0),len(data["labels"]))
                total_samples += len(data["labels"])
                # print the test data
                test_file = f.replace("train", "test")
                test_data = os.path.join(directory_path, test_file)
                data = np.load(test_data)
                print01(data["labels"].mean(axis=0),len(data["labels"]))
                total_samples += len(data["labels"])
                print()
            if is_npz(f) and "val" in f:
                data = np.load(os.path.join(directory_path, f))
                print("val:")
                print01(data["labels"].mean(axis=0),len(data["labels"]))
                total_samples += len(data["labels"])
                print()
            if is_npz(f) and "explanation" in f:
                data = np.load(os.path.join(directory_path, f))
                print("explanation:")
                print01(data["labels"].mean(axis=0),len(data["labels"]))
                total_samples += len(data["labels"])
                print()
            
        print("all samples:",total_samples)



#check_partitions('./Experiments/covertype/008_imb_0.99')
#check_partitions('./Experiments/adult_TargEncoded/008_imb_0.99')
#check_partitions('./Experiments/one_hot_Monks-3/008_imb_0.99')
#check_partitions('./Experiments/dry_bean/008_imb_0.99')
#check_partitions('./Experiments/dry_bean/008_iid')
names_of_datasets = ["adult_TargEncoded","dry_bean","diamond_1_hot"]
distrs  = ["iid","imb_5.0","imb_0.99"]
ncs = [12,16]
for name_of_dataset in names_of_datasets:
    for n in ncs:
        for distr in distrs:
            print("---------\nDataset:",name_of_dataset,"\nn_parts:",n,"\ndistr:",distr)
            check_partitions(os.path.join('./Experiments',name_of_dataset,str(n).zfill(3)+"_"+distr))
            fixed_expl_set = os.path.join('./Experiments',name_of_dataset)
            if os.path.exists(fixed_expl_set):
                check_partitions(fixed_expl_set)
            if os.path.exists(os.path.join('./Experiments',name_of_dataset,str(n).zfill(3)+"_"+distr,"models")):
                model_founds = os.listdir(os.path.join('./Experiments',name_of_dataset,str(n).zfill(3)+"_"+distr,"models"))
                model_founds = [x for x in model_founds if x.endswith("model.pth")]
                print("models found:",len(model_founds),model_founds[:3])
            else:
                print("models found: 0")
            print()