''' This module contains the subroutines to plot the results of the experiments and 
    to plot the distributions of the partitions of the dataset.'''

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import torch

from src.utils import create_partitions, name_by_folders
from src.dataset_utils import load_named_dataset

# from src.plotting_utils import whole_dataset_plot
# whole_dataset_plot(PCA(2).fit_transform(X))
# principal_comp_df, pcaDec2 = create_2d_projected_df(x_train = X, y_train = y)


# Function to save a figure, it gets the figure and the name of the figure
def save_fig(fig,name):
    '''Save a figure for the report and close it.'''
    print("saving figure as "+name)
    fig.savefig(name+".png",dpi=300)
    # None is the current figure
    plt.close(None)
# Function to plot the whole dataset as kde (kernel density estimation) and scatter
def whole_dataset_plot(pca_df):
    '''Plot the whole dataset.
    pca_df is a dataframe containing the PCA of the whole dataset and the labels.
    The PCA in the columns "PCA_1" and "PCA_2", the labels are in the column "class".'''
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    sns.set(rc={
            #'figure.facecolor':(1,1,1,0),
            #'axes.facecolor':(1,1,1,0),
            'axes.edgecolor':'#000000'}) 
    plt.rcParams['figure.figsize'] = (10.0, 10.0)
    graphic = sns.jointplot(data=pca_df,x="PCA_2", y="PCA_1",fill=True,kind="kde",palette='Set2')
    x_range=graphic.ax_joint.get_xlim()
    y_range=graphic.ax_joint.get_ylim()
    print("saving WHOLE")
    save_fig(graphic,"plots/4WholeDataset_KDE_filled_WHOLE")
    plt.clf()
    plt.close(graphic.fig)
    graphic = sns.jointplot(
            data=pca_df,
            x="PCA_2", y="PCA_1",
            hue="class",
            #fill=True,
            #kind="kde", #kind of 2dplot, try hex or kde, standard is scatter
            #space=0.2, # space between marginals and 2d plot
            palette='Set2')
    graphic.ax_joint.set_ylim([y_range[0],y_range[1]])
    graphic.ax_joint.set_xlim([x_range[0],x_range[1]])
    sns.move_legend(graphic.ax_joint, "upper right", title='Class')
    print("saving scatter")
    save_fig(graphic,"plots/1WholeDataset_Scatter")
    plt.close(graphic.fig)
    plt.clf()
    graphic = sns.jointplot(data=pca_df,x="PCA_2", y="PCA_1",hue="class",kind="kde",palette='Set2')
    graphic.ax_joint.set_ylim([y_range[0],y_range[1]])
    graphic.ax_joint.set_xlim([x_range[0],x_range[1]])
    sns.move_legend(graphic.ax_joint, "upper right", title='Class')
    print("saving KDE")
    save_fig(graphic,"plots/2WholeDataset_KDE")
    plt.close(graphic.fig)
    plt.clf()
    graphic = sns.jointplot(data=pca_df,x="PCA_2", y="PCA_1",hue="class",
                            fill=True,kind="kde",palette='Set2')
    graphic.ax_joint.set_ylim([y_range[0],y_range[1]])
    graphic.ax_joint.set_xlim([x_range[0],x_range[1]])
    sns.move_legend(graphic.ax_joint, "upper right", title='Class')
    print("saving KDE filled")
    save_fig(graphic,"plots/3WholeDataset_KDE_filled")
    plt.close(graphic.fig)
    return x_range,y_range

def plot_partitioned_dataset(partitions,
                             pca_decomposition_2:PCA,
                             x_range,
                             y_range,
                             alpha = 1e10
                             ):
    '''Plot the partitioned dataset.'''
    # define the type of partitioning
    partition_type = ("iid" if alpha>110.0 else 'imb_'+str(alpha))
    part = partitions[0][0]
    stacked_partitioned = np.hstack([pca_decomposition_2.transform(part),
                        np.zeros(part.shape[0]).reshape(part.shape[0],1)])
    print(stacked_partitioned.shape)
    partition_id = 0
    single_pca_df_partition = pd.DataFrame({"PCA_1":stacked_partitioned[:,0],
                                    "PCA_2":stacked_partitioned[:,1],
                                    "partition":1+stacked_partitioned[:,2].astype(int)})
    plt.clf()
    plt.rcParams['figure.figsize'] = (5.0, 5.0)
    # define the palette
    palette = sns.color_palette(palette="Set2")
    # plot the first partition
    graphic = sns.jointplot(
            data=single_pca_df_partition,
            x="PCA_2", y="PCA_1",
            #hue="partition",
            color=palette[partition_id],
            levels=12,
            fill=True,
            kind="kde", #kind of 2dplot, try hex or kde, standard is scatter
            #space=0.2, # space between marginals and 2d plot
            palette='Set2', # palette aka cmap
            #marginal_kws={'ls':'--'} # options for the marginals
        )
    graphic.ax_joint.set_ylim([y_range[0],y_range[1]])
    graphic.ax_joint.set_xlim([x_range[0],x_range[1]])
    save_fig(graphic,"plots/"+ partition_type +" partition_"+str(partition_id+1))
    plt.close(graphic.fig)
    for partition_id,sub_part in enumerate(partitions[1:]):
        partition_id+=1
        sub_part = sub_part[0]
        # compose the dataframe
        part = np.hstack([pca_decomposition_2.transform(sub_part),
                        partition_id*np.ones(sub_part.shape[0]).reshape(sub_part.shape[0],1)])
        stacked_partitioned = np.vstack([stacked_partitioned,part])
        single_pca_df_partition = pd.DataFrame({"PCA_1":part[:,0],
                                    "PCA_2":part[:,1],
                                    "partition":1+part[:,2].astype(int)})
        plt.clf()
        plt.rcParams['figure.figsize'] = (5.0, 5.0)
        graphic = sns.jointplot(
            data=single_pca_df_partition,
            x="PCA_2", y="PCA_1",
            #hue="partition",
            color=palette[partition_id],
            levels=12,
            fill=True,
            kind="kde", #kind of 2dplot, try hex or kde, standard is scatter
            #space=0.2, # space between marginals and 2d plot
            palette='Set2', # palette aka cmap
            #marginal_kws={'ls':'--'} # options for the marginals
        )
        graphic.ax_joint.set_ylim([y_range[0],y_range[1]])
        graphic.ax_joint.set_xlim([x_range[0],x_range[1]])
        save_fig(graphic,"plots/"+ partition_type +" partition_"+str(partition_id+1))
        plt.close(graphic.fig)
    # compose the dataframe of whole dataset projected on the 2D space
    pca_df_partitioned = pd.DataFrame({"PCA_1":stacked_partitioned[:,0],
                                    "PCA_2":stacked_partitioned[:,1],
                                    "partition":1+stacked_partitioned[:,2].astype(int)})
    plt.rcParams['figure.figsize'] = (10.0, 10.0)
    graphic = sns.jointplot(
        data=pca_df_partitioned,
        x="PCA_2", y="PCA_1",
        hue="partition",
        levels=12,
        fill=True,
        kind="kde", #kind of 2dplot, try hex or kde, standard is scatter
        #space=0.2, # space between marginals and 2d plot
        palette='Set2', # palette aka cmap
        #marginal_kws={'ls':'--'} # options for the marginals
    )
    graphic.ax_joint.set_ylim([y_range[0],y_range[1]])
    graphic.ax_joint.set_xlim([x_range[0],x_range[1]])
    save_fig(graphic,"plots/"+ partition_type +" partitions")
    plt.close(graphic.fig)

def compose_plots(num_partitions,partition_type):
    '''Compose the plots of the partitions and save the composed plot.'''
    fig, axarr = plt.subplots(2, 3, figsize=(21, 14))
    n_cols = int(num_partitions/2)
    for i in range(num_partitions):
        im_name = "plots/"+ partition_type + " partition_"+str(i+1)+".png"
        axarr[int(i/n_cols),(i%n_cols)].imshow(mpimg.imread(im_name))
    # turn off x and y axis
    _ = [ax.set_axis_off() for ax in axarr.ravel()]
    plt.tight_layout()
    plt.savefig("plots/Composed_"+partition_type+"_partitions.png")
    plt.close(fig)

def create_2d_projected_df(x_train,y_train):
    '''Create a dataframe containing the 2D projection of the dataset.
    returns the dataframe and the PCA object used to project the dataset.'''
    # Create the PCA object with 2 components as parameter
    pca = PCA(n_components=2)
    feat_projectet_2 = pca.fit_transform(x_train)
    x_1,x_2 = feat_projectet_2[:,0],feat_projectet_2[:,1]
    pca_df = pd.DataFrame({"PCA_1":x_1,"PCA_2":x_2,"class":y_train})
    return pca_df,pca

def plot_history(metrics_centralized,metrics, dt_name="DT name",experiment_settings="-1_default"):
    '''Plot the history of the metrics of the clients and the centralized model.'''
    #metrics = [clients_acc,clients_f1,clients_loss]
    #metrics_centralized = [acc,f1,loss]
    rounds_number,num_clients = metrics[0].shape
    print("found",num_clients,"clients trained for",rounds_number,"rounds")
    fig,axes = plt.subplots(1,3,figsize=(15,6))
    plot_idx = 0
    for metric_history,metric_history_centralized in zip(metrics,metrics_centralized):
        for cl_idx in range(num_clients):
            axes[plot_idx].plot(np.arange(rounds_number),metric_history.T[cl_idx,:rounds_number]
                                ,alpha=.3,ls='--')
        axes[plot_idx].plot(np.arange(rounds_number),metric_history.T[:,:rounds_number].mean(0),
                            color='k',label='Mean')
        axes[plot_idx].plot(np.arange(rounds_number),metric_history_centralized,
                            color='r',ls='-.',label='Centralized')
        axes[plot_idx].legend(fontsize=16)
        plot_idx+=1
    # turn on grid for all the subplots
    _ = [ax.grid(True) for ax in axes]
    # set the ticks
    ticks_vals = list(range(0,rounds_number+1,50))
    # set the plot details
    for axe,title in zip(axes,["Acc","F1 Macro","Loss"]):
        axe.set_title("test "+title,fontsize=22)
        axe.set_xticks(ticks_vals,[str(ticks_val) for ticks_val in ticks_vals],fontsize=15)
        axe.set_xlabel("Rounds",fontsize=16)
        if title!='Loss':
            axe.set_ylim(-.02,1.05)
        #else:
            #ax.plot(np.arange(rounds_number),clients_loss.sum(axis=1).T[:rounds_number])
    fig.suptitle(name_by_folders(dt_name=dt_name,exp_sett=experiment_settings),fontsize=40)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # load the dataset
    DATASET_FOLDER = 'Iris'
    NUM_PARTITIONS = 4
    ALPHA = 1e10
    #partition_type = ("iid" if alpha_param>1.0 else 'imb_'+str(ALPHA))
    # load the dataset
    print("Loading dataset: ",DATASET_FOLDER)
    patterns_train, patterns_test = load_named_dataset(dataset_name=DATASET_FOLDER)
    print("shapes:")
    x_tr,y_tr = patterns_train.tensors[0],patterns_train.tensors[1]
    y_tr = torch.max(y_tr,1)[1]
    print("\t",x_tr.shape,y_tr.shape)
    # create the partitions of the dataset
    print("Creating "+str(NUM_PARTITIONS)+" partitions of the dataset")
    parts, distribution = create_partitions((x_tr,y_tr),
                                                num_partitions=NUM_PARTITIONS,
                                                alpha=ALPHA)
    # plot the whole dataset
    print("Creating 2d projection of the dataset")
    principal_comp_df, pcaDec2 = create_2d_projected_df(x_train = x_tr,
                                                        y_train = y_tr)
    # plot the whole dataset and get the ranges of the joint plot
    print("Plotting the whole dataset")
    horiz_range, vertical_range = whole_dataset_plot(pca_df = principal_comp_df)
    # plot the partitioned dataset
    print("plotting the partitions")
    # plot the partitioned dataset
    plot_partitioned_dataset(partitions=parts,
                             pca_decomposition_2=pcaDec2,
                             x_range=horiz_range,
                             y_range=vertical_range,
                             alpha=ALPHA)
    # compose the plots of the partitions
    compose_plots(NUM_PARTITIONS, ("iid" if ALPHA>1.0 else 'imb_'+str(ALPHA)) )
