'''Module to clean the adult_datasets and save them as npz files.'''
import os
import numpy as np
import pandas as pd
import category_encoders as ce
from src.discretize import EntropyDiscretizer

def check_create_dir(path):
    '''Check if a directory exists, if not create it'''
    if not os.path.exists(path):
        # log the creation of the directory
        print("Creating directory: ",path)
        os.makedirs(path)
    print(path,"exists")
#pylint: disable=too-many-locals
#pylint: disable=invalid-name
#pylint: disable=line-too-long
def clean_adult(file_path,target_encode=False,prototipy=False):
    '''Clean the adult dataset and save it as a npz file.'''
    adult_path = os.path.join("Experiments","adult","adult.csv") #
    dt = pd.read_csv(adult_path,header=None, sep=',')
    #print()
    dt.columns = [e.strip() for e in dt.iloc[0,:]]
    print(dt.columns)
    #print(dt.columns)
    names = np.array([e.strip() for e in dt.iloc[0,:]])
    # Map Ages, Education, Workclass, and Weekly-Hours to smaller category set.
    dt=dt.iloc[1:,:]
    #special_category = 'occupation'
    #print(special_category)
    #print(dt[special_category].values)
    dt.loc[dt['occupation']=='Armed-Forces','occupation'] = 'Protective-serv'
    dt.loc[dt['workclass'].isin(['State-gov', 'Federal-gov', 'Local-gov']),
        'employment-type'] = 'Government'
    dt.loc[dt['workclass'].isin(['Self-emp-not-inc', 'Self-emp-inc']),
        'employment-type'] = 'Self-Employed'
    dt.loc[dt['workclass'].isin(['Private']),
        'employment-type'] = 'Privately-Employed'
    #special_category = 'employment-type'
    #print(special_category)
    #print(dt[special_category].values)
    #print(names)
    dt['education-num'] = dt['education-num'].values.astype(int)
    #special_category = 'education-num'
    #print(special_category)
    #print(dt[special_category].values)
    #print(type(dt[special_category].values[0]))
    #print(dt['education-num'].isin([8]))
    dt.loc[dt['education-num'].isin([2,3,4,5,6,7,8]),'education'] = 'Less than High School'
    dt.loc[dt['education-num'].isin([ 9,10]), 'education'] = 'High School'
    dt.loc[dt['education-num'].isin([11,12]), 'education'] = 'Associates'
    dt.loc[dt['education-num'].isin([13]),    'education'] = 'Bachelors'
    dt.loc[dt['education-num'].isin([14]),    'education'] = 'Masters'
    dt.loc[dt['education-num'].isin([15,16]), 'education'] = 'Pro'
    #education is now categorical
    numeric_features_idx = [0,10,11,12]
    x_float = np.array(dt.iloc[:,numeric_features_idx]).astype(np.float32)
    y = np.array(dt.iloc[:,14])#np.array(dt.iloc[1:,1])
    #Normalization of data
    x_float = (x_float - np.mean(x_float,axis=0)) / np.std(x_float,axis=0)
    y[y==' >50K'] = 1
    y[y==' <=50K'] = 0
    y = y.astype(np.intc)
    # LA REGOLA E' LA NORMA DELLE ECCEZIONI
    categorical_features_idx = [1,3,5,6,7,8,9,13]
    categorical_features = np.array(dt.iloc[:,categorical_features_idx])

    categorical_features = np.array([])
    cat_values = {}
    for cat_feat_idx in categorical_features_idx:
        x_c = dt.iloc[:,cat_feat_idx]
        x_c_values = set(x_c)
        if target_encode:
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
                                         feature_names=names[numeric_features_idx],
                                         categorical_features=[])
        x_float = discretizer.discretize(x_float)
    if target_encode:
        names = np.concatenate([names[numeric_features_idx],names[categorical_features_idx]])
    else:
        names = np.concatenate([names[numeric_features_idx],one_hot_feats_names])
    X = np.hstack([x_float,categorical_features])
    # save the dataset as npz
    folder_path = os.path.sep.join(file_path.split(os.path.sep)[:-1])
    check_create_dir(folder_path)
    # exclude the extension, savez implicitly adds it
    print("Saving dataset to: ",file_path)
    print("X shape: ",X.shape)
    print("names are ",names)
    # save them in a npz file
    np.savez(file_path,X=X,y=y)
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
