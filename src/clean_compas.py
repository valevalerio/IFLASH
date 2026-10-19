'''Module to clean the adult_datasets and save them as npz files.'''
import os
import pandas as pd
import numpy as np
from src.discretize import EntropyDiscretizer

import category_encoders as ce

def check_create_dir(path):
    '''Check if a directory exists, if not create it'''
    if not os.path.exists(path):
        # log the creation of the directory
        print("Creating directory: ",path)
        os.makedirs(path)
    print(path,"exists")
    #file_path,target_encode=False,prototipy=False):
def clean_compas(file_path , targetEnc=True, prototipy = False):
    load_file_path = os.path.join("Experiments","compas","compas-scores-two-years.csv")
    df = pd.read_csv(load_file_path)
    columns = ['age', #'age_cat', 
            'sex', 'race',  'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
            'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']
    

    df = df[columns]
    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])

    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])

    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)

    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    df['class'] = df['score_text']

    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    df.replace("Male",0,inplace=True)
    df.replace("Female",1,inplace=True)

    df.replace("Low",0,inplace=True)
    df.replace("Medium",1,inplace=True)
    df.replace("High",2,inplace=True)

    df.replace("M",0,inplace=True)
    df.replace("F",1,inplace=True)
    
    y = np.array(df["class"])
    dt = df
    print(df.columns)


    categorical_features_idx = [2]#["race"]
    numeric_features_names = ["age","sex","priors_count","days_b_screening_arrest","c_charge_degree","is_recid","is_violent_recid","two_year_recid"]
    numeric_features_idx = [i for i in np.arange(len(columns)) if i not in categorical_features_idx]
    
    X_float = dt[numeric_features_names].values
    numeric_features_idx = numeric_features_names
    x_float = (X_float - np.mean(X_float,axis=0)) / np.std(X_float,axis=0)    
    xs = np.array([])
    categorical_features = np.array([])
    cat_values = {}
    names = columns#np.array([e.strip() for e in dt.iloc[0,:]])

    for cat_feat_idx in categorical_features_idx:
        
        x_c = dt.iloc[:,cat_feat_idx]
        x_c_values = set(x_c)
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
                one_hot_feats_names.append("is "+cat_val)
            categorical_features = np.hstack([categorical_features ,onehot]) if categorical_features.size else onehot
    if prototipy:
        discretizer = EntropyDiscretizer(x_float,labels=y,
                                         feature_names=numeric_features_names,
                                         categorical_features=[])
        x_float = discretizer.discretize(x_float)
    if targetEnc:
        names = np.concatenate([numeric_features_names,["race"]])
    else:
        names = np.concatenate([numeric_features_names,one_hot_feats_names])
        

    X = np.hstack([x_float,categorical_features])
    for x in range(X.shape[1]):
        print(type(X[:,x]),X[:,x].shape)
    # save the dataset as npz
    folder_path = os.path.sep.join(file_path.split(os.path.sep)[:-1])
    check_create_dir(folder_path)
    # exclude the extension, savez implicitly adds it
    print("Saving dataset to: ",file_path)
    print("X shape: ",X.shape)
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

#clean_compas(os.path.join("Experiments","compas_TargEncoded","compas_TargEncoded.npz")
#             ,targetEnc=True, prototipy=False)