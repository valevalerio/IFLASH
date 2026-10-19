''' This file contains the functions to compute the shap values 
    both for the global and the local models.
    Used by the server and the clients.
'''
import os
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.autograd import Variable
# Ignoring warning relatively to Numba and to deprecation of functions from numpy
warnings.filterwarnings("ignore")
import shap # pylint: disable=wrong-import-position
warnings.filterwarnings("default")

# compute shap values of black box model on the partition to explain using the background data.
def get_shap_explanations(black_box : Module,
                          background_data_loader : DataLoader,
                          explain_partition : DataLoader):
    '''Get the shap values of the partition to explain, uses the background data.
    the function returns the shap values of the explained partition.'''
    black_box.eval()
    torch.set_grad_enabled(False)
    # Define function to wrap model to transform data to tensor
    def bb_wrapper(sample):
        return black_box( Variable( torch.from_numpy(sample) ) ).detach().numpy()
    # Convert TensorDataset to numpy
    backgroud_data = background_data_loader.dataset[:][0].numpy()
    # Convert TensorDataset to numpy
    te_data = explain_partition.dataset[:][0].numpy()
    # The explainer doesn't like tensors, hence the f function
    print("shapes of background and samples to explain",backgroud_data.shape,te_data.shape)
    explainer = shap.KernelExplainer(bb_wrapper, backgroud_data)
    shap_values = explainer.shap_values(te_data)

    #torch.set_grad_enabled(True)
    #backgroud_data = background_data_loader.dataset[:][0]
    #te_data = explain_partition.dataset[:][0]
    #explainer = shap.GradientExplainer(black_box, [backgroud_data])
    # Get the shap values from test data
    #shap_values = explainer.shap_values([te_data])



    shap_values = shap_values[0]
    return shap_values
# plot the shap values and save them there
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# the shapely values are in the variable shap_values
# plot them with a blue red color map, as in the paper.
# use imshow, with no interpolation, and with the x axis as the features
# and the y axis as the samples.
# save the plot in the current folder with the name shap_C_016.png
# set the boundaries of the color map to be symmetric and within the range -5, 5
'''
os.chdir("diamond_1_hot")
os.chdir("..")
os.chdir("016_imb_0.99/shap")
os.listdir()
# load the npz file
shap_values = np.load("shap_C_016.npz")["shap_values"]

plt.clf()
plt.imshow(shap_values,aspect="auto",interpolation='none',cmap="bwr",vmin=-5,vmax=5)
# the plot still have the interpolation, to remove it use interpolation=None



plt.colorbar()
plt.xlabel("Features")
plt.ylabel("Samples")
plt.title("Shap values of the client 016")


plt.savefig("shap_C_016.png")
# print the current directory and the name of the file
print(os.getcwd()+"/shap_C_016.png")
plt.close()

'''
# gather the explanations from the clients, compute the mean and save it
def gather_clients_shap(folder_dataset=".",folder_explanations="explanations"):
    '''Get the explanations from the clients, compute the mean and save it.
     Return E_hat, the mean of the clients explanations.'''
    exp_path = os.path.join(folder_dataset,folder_explanations)
    file_names = os.listdir(exp_path)
    file_names = [f for f in file_names if ('shap' in f and f[-3:]=='npz' and not "_S_" in f)]
    print("Found",len(file_names),"explanation matrices")
    matrices = []
    for file_name in file_names:
        file_name = os.path.join(exp_path,file_name)
        matrices.append(np.load(file_name)["shap_values"])
    matrices = np.array(matrices)
    e_hat = matrices.mean(axis=0)
    np.savez(os.path.join(folder_dataset,"E_hat.npz"),shap_values=e_hat)
    print("Saved E_hat")
    return e_hat
# compare the explanations of the mean of the clients and the values of the server
def compare_explanations(folder_dataset=".",folder_explanations="explanations"):
    ''' Compare the explanations of the mean of the clients and the values of the server.
    Returns the difference between the two: Delta.'''
    # define the path of the explanations
    exp_path = os.path.join(folder_dataset,folder_explanations)
    # Load the explanations
    e_hat = np.load(os.path.join(folder_dataset,"E_hat.npz"))["shap_values"]
    e_s = np.load(os.path.join(exp_path,"E_S.npz"))["shap_values"]
    # Compute the delta
    delta = e_s-e_hat
    print("std of Delta:",delta.std(axis=0).mean())
    return delta
if __name__ == '__main__':
    gather_clients_shap()
    compare_explanations()
