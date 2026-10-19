'''Module to get the dalex explanations of the clients and the mean of the clients explanations.
    It also compares the explanations of the mean of the clients and the values of the server.
    The modules is analogous to shap_explanations.py
    The function to get the dalex explanations is get_dalex_explanation.
    '''
import os
import numpy as np
import dalex as dx
from torch.utils.data import DataLoader
from torch.nn import Module
import torch
# function to get the explanation with dalex, takes as input the model,
# the partition to explan and the ground partition, returns the dalex values
def get_dalex_explanation(black_box : Module,
                          background_data_loader : DataLoader,
                          explain_partition : DataLoader):
    '''Get the dalex explanations of the partition to explain, uses the background data.
    Returns the dalex values of the explained partition.
    type of problem: classification'''
    print(type(black_box),type(background_data_loader),type(explain_partition))
    print(background_data_loader.dataset.tensors[0].shape,
          background_data_loader.dataset.tensors[1].shape,
          explain_partition.dataset.tensors[0].shape,
          explain_partition.dataset.tensors[1].shape)

    black_box.eval()
    torch.set_grad_enabled(False)
    # Define function to wrap model to transform data to tensor
    # The explainer doesn't like tensors, hence the f function
    def predict_fun_f(torch_bb,df_samples):
        return torch.max(torch_bb( torch.Tensor(df_samples.values) ).detach(),1)[1].numpy()
    # Create the explainer
    print("creating explainer")
    explainer = dx.Explainer(model = black_box,
             data = background_data_loader.dataset.tensors[0].numpy(),
             predict_function=predict_fun_f,
             y = background_data_loader.dataset.tensors[1].numpy(),
            label='Target class',
            model_type='classification')
    num_features = background_data_loader.dataset.tensors[0].shape[1]
    dalex_values = np.empty((num_features, 0)).T
    te_data = explain_partition.dataset[:][0].numpy()
    print("Samples to explain",explain_partition.dataset[:][0].shape)
    for explain_features in te_data:
        # Get the break down values from data to explain
        expl = explainer.predict_parts(explain_features,type='break_down', keep_distributions=True)
        ordered_dalex_vals=expl.result.sort_values(by='variable_name').iloc[1:-1,:]["contribution"]
        local_feature_importance = ordered_dalex_vals.values.reshape(1,num_features)
        dalex_values=np.vstack((dalex_values, local_feature_importance))
    return dalex_values
# gather the explanations from the clients, compute the mean and save it
def gather_clients_dalex(folder_dataset=".",folder_explanations="explanations"):
    '''Get the explanations from the clients, compute the mean and save it.
     Return E_hat_dalex, the mean of the clients explanations.'''
    exp_path = os.path.join(folder_dataset,folder_explanations)
    file_names = os.listdir(exp_path)
    file_names = [f for f in file_names if (f[:5]=='dalex' and f[-3:]=='npz')]
    print("Found",len(file_names),"explanation matrices")
    matrices = []
    for file_name in file_names:
        file_name = os.path.join(exp_path,file_name)
        matrices.append(np.load(file_name)["dalex_values"])
    matrices = np.array(matrices)
    e_hat_dalex = matrices.mean(axis=0)
    np.savez(os.path.join(folder_dataset,"E_hat_dalex.npz"),dalex_values=e_hat_dalex)
    print("Saved E_hat_dalex")
    return e_hat_dalex

def compare_explanations(folder_dataset=".",folder_explanations="explanations"):
    ''' Compare the explanations of the mean of the clients and the values of the server.
    Returns the difference between the two: Delta.'''
    # define the path of the explanations
    exp_path = os.path.join(folder_dataset,folder_explanations)
    # Load the explanations
    e_hat_dalex = np.load(os.path.join(folder_dataset,"E_hat_dalex.npz"))["dalex_values"]
    e_s = np.load(os.path.join(exp_path,"E_S_dalex.npz"))["dalex_values"]
    # Compute the delta
    delta = e_s-e_hat_dalex
    print("std of Delta:",delta.std(axis=0).mean())
    return delta
