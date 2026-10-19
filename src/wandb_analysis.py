import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def get_f1_majority2(ratio,num_classes):
    return (2*ratio/(ratio+1))/num_classes # the average

def get_f1_majority_weighted(ratio,num_classes):
    return (2*ratio/(ratio+1))*ratio # the average



def get_dt_name_nc_d(hp_search):
    '''This function is used to extract the 
    dataset name, the number of clients
    and the distribution from the sweep name'''
    # last element is iid or a number
    # if it is a number and such number i > 110 distr is iid
    last = hp_search.split("_")[-1]
    hp_search = hp_search.split("_")
    # we use the bitwise or to avoid the exception if last is not a number
    if last == "iid":
        distr = "iid"
        nc = hp_search[-2]
        dt_name = "_".join(hp_search[1:-2])
    elif float(last) > 110:
        distr = "iid"
        nc = hp_search[-3]
        dt_name = "_".join(hp_search[1:-3])
    elif float(last) < 110:
        distr = "imb_"+hp_search[-1]
        if hp_search[-1]=="_5":
            distr+=".0"
        nc = hp_search[-3]
        dt_name = "_".join(hp_search[1:-3])
    
    return dt_name,int(nc),distr

# the function below is used to extract the worst performing client from the sweep
def get_nc(keys):
    # conf is the configuration of the run
    # for each client I log train and val of f1 and accuracy, 4 metrics
    # the global metrics are the same (but global, so 4 in total)
    # the metrics on the test are 3 (f1, accuracy, loss)
    # others metrics are 8
    # so the number of clients is (len(conf.keys())-4-4-3)/4
    return np.max([int(el.split("/")[0].split("_")[1]) for el in list(keys) if "client" in el.lower()])
def get_num_classes(keys):
    return np.max([int(el.split("/")[1].split("_")[-1]) for el in list(keys) if "class" in el and "client" in el.lower()])

# the function below is used to extract the worst, the best and all the clients metric from the run
def get_best_worst_all(run_dict, metric):
    ''' function to extract the best and worst client from the run_dict
    along with the metric value of the best and worst client
    '''
    nc = get_nc(run_dict.keys())
    best_client = 1
    try:
        best_metric_value = run_dict["/".join(["client_001", metric])]
    except: 
        best_metric_value = run_dict["/".join(["Client_001", metric])]
    worst_client = 1
    try:
        worst_metric_value = run_dict["/".join(["client_001", metric])]
    except:
        worst_metric_value = run_dict["/".join(["Client_001", metric])]
    all_metric_value = []
    for i in range(1,nc+1):
        try:
            metric_value = run_dict["/".join(["client_"+str(i).zfill(3), metric])]
        except:
            metric_value = run_dict["/".join(["Client_"+str(i).zfill(3), metric])]
        all_metric_value.append(metric_value)
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_client = i
        if metric_value < worst_metric_value:
            worst_metric_value = metric_value
            worst_client = i
    return best_client, best_metric_value, worst_client, worst_metric_value, all_metric_value


# get_array_metric(runs_df.values[0][1],"val_precision").shape, get_array_metric(runs_df.values[0][1],"val_recall").shape
def get_metric(run_dict,metric="precision"):
    num_classes = get_num_classes(run_dict.keys())+1
    metr = np.zeros(num_classes)
    for j in range(num_classes):
        # aggregated metric
        # metr[j] = run_dict["/".join(["global", metric+"_class_"+str(j).zfill(2)])]
        metr[j] = run_dict["test_"+metric+"_class_"+str(j).zfill(2)]
    return np.array(metr)


# find the array_metr of every class for every client
def get_array_metric(run_dict,metric="val_precision"):
    nc = get_nc(run_dict.keys())
    num_classes = get_num_classes(run_dict.keys())+1
    array_metr = np.zeros((nc,num_classes))
    # iterate over the classes
    for i in range(1,nc+1):
        for j in range(num_classes):
            try:
                array_metr[i-1,j] = run_dict["/".join(["client_"+str(i).zfill(3), metric+"_class_"+str(j).zfill(2)])]
            except:
                array_metr[i-1,j] = run_dict["/".join(["Client_"+str(i).zfill(3), metric+"_class_"+str(j).zfill(2)])]
            #print(metric,"of client",i,"class",j,"is",np.round(array_metr[i-1,j]))
    return np.array(array_metr)
def get_hp_vals(configuration):
    couples = [hp[2:].split("=") for hp in configuration.split(" ") if "=" in hp]
    return ", ".join([hp_name+" "+(str(hp_val)[:5] ) for hp_name, hp_val in couples])


import matplotlib.colors as m
# this function is used to plot the metric per class for each client (its a numpy matrix of shape (num_clients, num_classes))
def plot_table(ax, data, title="", vmin=0, vmax=1,add_description=True):
    # if the data is not a list the aspact ratio is auto
    # if the data is a list, the aspect ratio is 1
    base_fontsize = 18
    if data.shape[0] != 1:
        # reverse the order of the rows (clients)
        data = data[::-1]
        im = ax.imshow(data, cmap="Blues", vmin=vmin, vmax=vmax, interpolation=None, aspect="auto")
        scaled_fontsize = base_fontsize / (data.shape[0] / 4.0)
    else:
        im = ax.imshow(data, cmap="Blues", vmin=vmin, vmax=vmax, interpolation=None)
        scaled_fontsize = base_fontsize

    # no grid
    ax.grid(False)
    #if add_description:
    ax.set_title(title, fontsize=scaled_fontsize)
    ax.tick_params(axis='x', rotation=30)
    # write the values of the heatmap in each cell
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # if the color is dark, write the value in white, otherwise in black
            if data[i,j]>0.6:
                ax.text(j,i, np.round(data[i,j],2), ha="center", va="center", color="white")
            else:
                ax.text(j,i, np.round(data[i,j],2), ha="center", va="center", color="black")
    # the x labels are the classes, the y labels are the clients
    # ax.set_xlabel("Classes", fontsize=scaled_fontsize)
    #ax.set_ylabel("Clients", fontsize=scaled_fontsize)
    if data.shape[0] != 1:
        ax.set_title(title, fontsize=16)
        if "recall" not in title:
            ax.set_yticks(range(data.shape[0]),["Node "+str(i+1) for i in list(range(data.shape[0]))[::-1]], fontsize=scaled_fontsize)
        else:
            ax.set_yticks([])
    else:
            ax.set_yticks([])
    # the xticks are the classes class 0, class 1, ...
    if add_description:
        ax.set_xticks(range(data.shape[1]),["Class "+str(i) for i in range(data.shape[1])], fontsize=scaled_fontsize-2, rotation=30)
    else:
        ax.set_xticks(range(data.shape[1]),["Class "+str(i) for i in range(data.shape[1])], fontsize=10, rotation=30)
    return im

def additional_plot(s,axes,precisions, recalls):
    # create addtional plot for the precision and recall
    name_of_dataset,num_clients,distribution = get_dt_name_nc_d(s)
    if distribution[-1]=="0": distribution = distribution[:-1]
    fig_name = "_".join([name_of_dataset,str(num_clients).zfill(3),distribution])
    fig_name+="_precision_recall"
    fig_path = os.path.join("reports",fig_name+".pdf")
    print(fig_path)
    #print(precisions_dict[s].shape)
    #print(recalls_dict[s].shape) 

    #fig, axes = plt.subplots(figsize=(7,7),nrows=3,ncols=2)
    row_idx = 0
    col_idx = 0
    for tuple_array_metric,metric_name in zip([precisions, recalls],
                            ["precision", "recall"]):
        row_idx = 0
        for array_metric, set_name in zip(tuple_array_metric, ["train","test"]):
            fmetric_name = metric_name+" "+set_name
            plot_table(axes[row_idx][col_idx], array_metric, title=fmetric_name,add_description=False)
            row_idx+=1
            # plot the heatmap of the metric 

            # add a last raw similarly to the previous but on the whole set
        fmetric_name = metric_name+" global\non entire test set"
        plot_table(axes[-1][col_idx], tuple_array_metric[-1].reshape(1,-1), title=fmetric_name,add_description=False)
        col_idx+=1
    # space the plots manually using subplots_adjust
    #plt.tight_layout()
    #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.5)
    #plt.tight_layout() 
    #plt.savefig(fig_path, bbox_inches='tight')
    #plt.show()
    #plt.clf()
    #plt.close()
def check_partitions(directory_path,num_clients):
    is_npz = lambda x: x.endswith('.npz')
    total_samples = 0
    samples = [_ for _ in range(num_clients)]
    sorted_samples_file_names = [e for e in sorted(os.listdir(directory_path)) if is_npz(e) and "train" in e]
    print(sorted_samples_file_names)
    for j,f in enumerate(sorted_samples_file_names):
        data = np.load(os.path.join(directory_path, f))
        total_samples += len(data["labels"])
        part_samples = data["labels"].sum(axis=0)
        test_file = f.replace("train", "test")
        test_data = os.path.join(directory_path, test_file)
        data = np.load(test_data)
        total_samples += len(data["labels"])
        part_samples += data["labels"].sum(axis=0)
        samples[j]=part_samples
    return np.array(samples)


#for s in ordered_keys:
def plot_run(s, scores, precisions, recalls, configurations,prefix="",f_path=""):
    #s = "FdSwp_"+name_of_dataset+"_"+str(num_clients).zfill(2)+"_"+distribution
    #name_of_dataset = "adult_TargEncoded"
    #num_clients = 16
    #num_clients_s = str(num_clients).zfill(3)
    #distribution = "imb_0.99"
    
    name_of_dataset,num_clients,distribution = get_dt_name_nc_d(s)
    #if distribution[-1]=="0":
    #    distribution = distribution[:-1]
    num_clients_s = str(num_clients).zfill(3)

    fig_name = "_".join([name_of_dataset,num_clients_s,distribution])
    fig_title = " ".join(name_of_dataset.split("_"))+"\n"+str(num_clients)+" clients\n"+distribution
    if f_path=="":
        f_path = os.path.join("Experiments",name_of_dataset,num_clients_s+"_"+distribution)
    all_samples = check_partitions(f_path,num_clients)

    # draw the stacked bar chart
    # set the colormap of plt as Set2

    # fig, axes = plt.subplots(figsize=(14,7),ncols=2, nrows=1)
    fig = plt.figure(figsize=(18,9))
    gs = GridSpec(1, 3, figure=fig)

    plt.suptitle(fig_title, fontsize=20)
    plt.subplot(1,3,1)
    plt.ylabel("Clients", fontsize=18)
    plt.xlabel("Number of samples", fontsize=18)
    plt.yticks(range(all_samples.shape[0]),["node "+str(i+1) for i in range(all_samples.shape[0])], fontsize=16)
    plt.xticks(fontsize=16, rotation=30)
    plt.grid(True)
    bottoms = np.zeros(all_samples.shape[0])
    for class_idx in range(all_samples.shape[1]):
        plt.barh(np.arange(len(all_samples)), all_samples[:,class_idx], label="Class "+str(class_idx),left=bottoms)
        bottoms += all_samples[:,class_idx]
    # put the legend outside the plot on the right
    plt.legend(fontsize=16)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=16)
    # save the figure as a pdf
    #plt.savefig(f_path+".pdf", bbox_inches='tight')
    #plt.show()
    #print(f_path+".pdf")

    # get the y limits of the plot to make the next plot with the same limits
    y_min, y_max = plt.gca().get_ylim()


    plt.subplot(1,3,2)
    # move the subplots more to the left
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.55)
    # now plot the train score for each client putting in the x axis the metric
    # the colors are not relevant

    tr_index = 5
    val_index = 6
    test_index = 7
    print(s)

    # plot the boxplot of the train score
    # the x axis is the metric
    # the y axis the clients
    print(len(scores[tr_index]))
    print(len(scores[val_index]))
    # plot a vertical line for the global score
    plt.axvline(x=scores[test_index], color="gray", linestyle="--", label="global test" + " " + str(np.round(scores[test_index],2)))
    # plot the train scores
    plt.scatter(scores[val_index],range(num_clients), s=100, marker="o",label=("validation" if prefix=="" else "test"))
    plt.scatter(scores[tr_index],range(num_clients), color="k", s=100, marker="x",label="train")
    # plot the validation scores
    plt.yticks(range(num_clients),["Node "+str(i+1) for i in range(num_clients)], fontsize=16)

    # save the trivial f1 score, if the model always predict the majority class
    majority_f1_macro = []
    for e in all_samples:
        r = np.max(e)/np.sum(e)
        majority_f1_macro.append(
            get_f1_majority2(r,len(e))
        )
    plt.scatter(majority_f1_macro,range(num_clients), color="k", s=100, marker=".",label="majority classifier")


    plt.xticks(fontsize=16)
    plt.legend(fontsize=16)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=16)
    plt.xlabel("F1 score macro", fontsize=18)
    plt.ylabel("Clients", fontsize=18)

    # set the limits of the y axis
    plt.ylim(y_min,y_max)



    #nested_gs = GridSpec.from_subplotspec(3, 2, subplot_spec=gs[2])
    # Create axes for the nested grid
    nested_axes = [[plt.subplot2grid((3, 6), (i, j+4), fig=fig) for j in range(2)] for i in range(3)]
    # Call the additional_plot function with the nested axes
    additional_plot(s, nested_axes,precisions=precisions, recalls=recalls)
    # add the caption regarding the best hyperparameters
    # get the best hyperparameters
    best_hp_caption = get_hp_vals(configurations[s])
    # add the caption regarding the best configuration 
    fig.text(0.5, -0.05, best_hp_caption, ha='center',fontsize=16)
    os.makedirs("reports", exist_ok=True)
    if prefix!="":
        prefix+="_"
    fig_path = os.path.join("reports",prefix+fig_name+".pdf")
    plt.savefig(fig_path, bbox_inches='tight')
    if num_clients == 0:
        plt.show()
    plt.clf()
    plt.close()

    return fig_path




def plot_partitions(num_clients,name_of_dataset,distribution,save_fig=False,join_all = False):
    num_clients_s = str(num_clients).zfill(3)
    exp_folder = os.path.join("Experiments",name_of_dataset,num_clients_s+"_"+distribution)
    #all_samples = check_partitions("./Experiments/adult_TargEncoded/008_iid/")
    all_samples = check_partitions(exp_folder,num_clients)

    f_path = os.path.join("Experiments",name_of_dataset,num_clients_s+"_"+distribution)
    fig_name = "_".join([name_of_dataset,num_clients_s,distribution])
    fig_title = " ".join(name_of_dataset.split("_"))+"\n"+str(num_clients)+" clients\n"+distribution.split("_")[1]
    all_samples = check_partitions(f_path,num_clients)

    # draw the stacked bar chart
    # set the colormap of plt as Set2

    # fig, axes = plt.subplots(figsize=(14,7),ncols=2, nrows=1)
    fig = plt.figure(figsize=(18,9))

    plt.suptitle(fig_title, fontsize=20)
    plt.ylabel("Clients", fontsize=18)
    plt.xlabel("Number of samples", fontsize=18)
    plt.yticks(range(all_samples.shape[0]),["node "+str(i+1) for i in range(all_samples.shape[0])], fontsize=16)
    # use scientific notation for the x axis in order to have a better visualization, 
    # every tick is multiplied by 10^3
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks(fontsize=16, rotation=30)
    # use a sns color palette as default
    plt.set_cmap("Set2")

    plt.grid(True)
    bottoms = np.zeros(all_samples.shape[0])
    for class_idx in range(all_samples.shape[1]):
        plt.barh(np.arange(len(all_samples)), all_samples[:,class_idx], label="Class "+str(class_idx),left=bottoms)
        bottoms += all_samples[:,class_idx]
    # put the legend outside the plot on the right
    plt.legend(fontsize=16)
    if save_fig:
        print("saving in ",fig_name+".png")
        plt.show()
        fig_ppath = os.path.join("Experiments",name_of_dataset,fig_name+".pdf")
        plt.savefig(fig_ppath, bbox_inches='tight')
        plt.close()
    
    if join_all:
        # seek all the pdf in the folder and join them
        pdfs = [os.path.join("Experiments",name_of_dataset,f) 
                        for f in os.listdir(os.path.join("Experiments",name_of_dataset))
                 if f.endswith(".pdf")]
        # sort the pdfs by the last number in the name
        pdfs = sorted(pdfs, key=lambda x: float("".join(x.split("_")[-1][:-4])))
        print(pdfs)
        from PyPDF2 import PdfWriter, PdfReader
        merger = PdfWriter()
        for f in pdfs:
            if not num_clients_s in f:
                continue
            metric_page = PdfReader(open(f,"rb")).pages[0]
            merger.add_page(metric_page)
            # then remove the file
            os.remove(f)
            
        # write them all in a single pdf named after the dataset
        with open("reports/"+"alphas_"+name_of_dataset+"_"+num_clients_s+".pdf", "wb") as outputStream:
            merger.write(outputStream)
        #with open("reports/combined.pdf", "wb") as outputStream:

