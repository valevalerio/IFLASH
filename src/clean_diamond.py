'''Module to clean the diamonds dataset and save it in a npz file'''
import os
import numpy as np
from src.discretize import EntropyDiscretizer
import pandas as pd
from datasets import load_dataset
import category_encoders as ce

def check_create_dir(path):
    '''Check if a directory exists, if not create it'''
    if not os.path.exists(path):
        # log the creation of the directory
        print("Creating directory: ",path)
        os.makedirs(path)
    print(path,"exists")

def clean_diamond(file_path , targetEnc=False, prototipy = False):
    '''Clean the diamonds dataset and save it in a npz file'''
    dt = load_dataset("mstz/diamonds","cut")["train"].to_pandas()
    for featu in ["observation_point_on_axis_y",
                  "observation_point_on_axis_x",
                  "observation_point_on_axis_z"]:
        dt = dt[dt[featu]<11]
    categorical_features_idx = np.argwhere(dt.columns == "color")[0]

    numeric_features_names = [f for f in dt.columns if f not in ["color","cut"]]
    X_float = dt[numeric_features_names].values
    x_float = (X_float - np.mean(X_float,axis=0)) / np.std(X_float,axis=0)    

    categorical_features = np.array([])
    cat_values = {}
    names = numeric_features_names
    y = dt["cut"]

    for cat_feat_idx in categorical_features_idx:
        x_c = dt.iloc[:,cat_feat_idx]
        x_c_values = sorted(set(x_c))

        if targetEnc:
            #print(names[catFeatIdx],len(x_c_values))
            str_data = [str(e) for e in x_c]
            target_encoder = ce.TargetEncoder()
            target_encoder.fit(str_data,y)
            xtargenc = target_encoder.transform([str(e) for e in x_c]).to_numpy()
            categorical_features = np.hstack([categorical_features,xtargenc]) if categorical_features.size else xtargenc
            feat_name = "P(t="+str(1)+"|"+names[cat_feat_idx]+")"
            cat_values[feat_name] = np.unique(xtargenc)
            #print(feat_name,cat_values[feat_name],)
            cat_values[feat_name] = {c_val:target_encoder.transform([c_val])[0].to_numpy().astype(float)[0] for c_val in sorted(dt.iloc[:,cat_feat_idx].unique())}
            #print(feat_name,cat_values[feat_name],)
        else:
            one_hot_feats_names = []
            onehot = np.zeros((x_c.shape[0],len(set(x_c))))
            cat_feat_2_idx = {k:i for i,k in enumerate(x_c_values)}
            for i,cat_val in enumerate(x_c):
                onehot[i,cat_feat_2_idx[cat_val]]=1
            for i,cat_val in enumerate(x_c_values):
                one_hot_feats_names.append("is "+cat_val)
            categorical_features = np.hstack([categorical_features ,onehot]) if categorical_features.size else onehot
    if prototipy:
        discretizer = EntropyDiscretizer(x_float,labels=y,
                                         feature_names=numeric_features_names,
                                         categorical_features=[])
        x_float = discretizer.discretize(x_float)
    if targetEnc:
        names = np.concatenate([numeric_features_names,["color"]])
    else:
        names = np.concatenate([numeric_features_names,one_hot_feats_names])

    X = np.hstack([x_float,categorical_features])
    # save the dataset as npz
    folder_path = os.path.sep.join(file_path.split(os.path.sep)[:-1])
    check_create_dir(folder_path)
    # exclude the extension, savez implicitly adds it
    print("Saving dataset to: ",file_path)
    print("X shape: ",X.shape)
    names = [n.replace("observation_point_on_axis","obs_on") for n in names]

    print("names are ",names)
    # save them in a npz file
    np.savez(file_path[:-4],X=X,y=y,allow_pickle=False)
    # save the names of the features in a txt file
    names_path = file_path.replace(".npz","_names.txt")
    with open(names_path,'w+',encoding='utf-8') as file_name:
        file_name.write(", ".join(names))
    # save the dictionary of the categorical values in a txt file dumping the dictionary as a string
    string_dict = str(cat_values).replace("\'","\"")
    # dumps the dictionary as a string
    # create file
    cat_values_path = file_path.replace(".npz","_cat_values.txt")
    # write the dictionary as a string
    with open(cat_values_path,'w+',encoding='utf-8') as file_name:
        file_name.write(string_dict)
    print("Dataset saved to: ",file_path)
    print("Names saved to: ",names_path)
    print("Categorical values saved to: ",cat_values_path)