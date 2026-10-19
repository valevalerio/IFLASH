''' Module to plot the times of the explanations of the experiments and print few statistics'''
import os
import matplotlib.pyplot as plt
import numpy as np
# functions to plot the times
def seek_times(dataset_folder):
    '''inside the dataset folder there are the folders of the experiments
    inside each experiment folder there are the folders of the explainers: shap and dalex
    inside each explainer folder there are the values of the explanations and files with the time
    all the files with the suffix _time.txt contain the time of the explanation of the client
    the files with the suffix server_time.txt are the time of the explanation of the server
    '''
    # find the folders of the experiments
    exp_folders = sorted([e for e in os.listdir(dataset_folder)
                          if os.path.isdir(os.path.join(dataset_folder,e))])
    print("found",len(exp_folders),"folders of experiments")
    explanation_times = {}
    for exp_folder in exp_folders:
        explanation_times[exp_folder] = {}
        train_t_files = [e for e in os.listdir(os.path.join(dataset_folder,exp_folder))
                   if e.endswith("_time.txt")]
        # read the only value contained in the file, is a float
        if len(train_t_files):
            with open(os.path.join(dataset_folder,exp_folder,train_t_files[0]),'r',
                      encoding='utf-8') as file_name:
                explanation_times[exp_folder]["train"] = float(file_name.read())
            for explainer in ["shap","dalex","fastshap"]:

                exp_times_files = os.path.join(dataset_folder,exp_folder,explainer)
                exp_times_files = os.listdir(exp_times_files)
                exp_times_files = [e for e in exp_times_files if e.endswith("_time.txt")]
                explanation_times[exp_folder][explainer] = {}
                # read the only value contained in the file, is a float
                client_times = []
                for time_f in exp_times_files:
                    if not (time_f.endswith("server_time.txt") or time_f.endswith("fastshap_time.txt")):
                        with open(os.path.join(dataset_folder,exp_folder,explainer,time_f),'r',
                                  encoding= 'utf-8') as file_name:
                            actual_time = float(file_name.readline())# if explainer!="fastshap" else float(file_name.readline()))
                            print(actual_time,os.path.join(dataset_folder,exp_folder,explainer,time_f))
                            client_times = client_times + [actual_time]
                    else:
                        
                        with open(os.path.join(dataset_folder,exp_folder,explainer,time_f),'r',
                                  encoding='utf-8') as file_name:
                            #server_time = float(file_name.read())
                            server_time = float(file_name.readline())# if explainer!="fastshap" else float(file_name.readline()))
                            print(server_time,os.path.join(dataset_folder,exp_folder,explainer,time_f))
                            explanation_times[exp_folder][explainer]["server"] = server_time
                explanation_times[exp_folder][explainer]["clients"] = client_times

                print("found",len(exp_times_files),
                      "files for the explainer",explainer, 
                      "in the experiment",exp_folder)
        else:
            continue
    return explanation_times

if __name__ == "__main__":
    dt_folder = os.path.join("Experiments","adult_TargEncoded")
    times_dict = seek_times(dt_folder)
    exp_names = sorted(list(times_dict.keys()))
    train_times = []

    dalex_server_times = []
    shap_server_times = []
    # the mean is over iid vs non-iid
    dalex_client_times = []
    shap_client_times = []
    for j,experiment_name in enumerate(exp_names):

        train_t = times_dict[experiment_name]["train"]/60
        train_times.append(train_t)

        for expl_name in ["shap","dalex","fastshap"]:
            #print(experiment_name,expl_name)
            server_expl_time = times_dict[experiment_name][expl_name]["server"]/60
            if expl_name=="dalex":
                dalex_server_times.append(server_expl_time)
                dalex_client_times.append(
                    np.array(times_dict[experiment_name][expl_name]["clients"])/60)
            else:
                shap_server_times.append(server_expl_time)
                shap_client_times.append(
                    np.array(times_dict[experiment_name][expl_name]["clients"])/60)

    dalex_iid =     np.array([e for i,e in enumerate(dalex_server_times) if i%2==0])
    dalex_non_iid = np.array([e for i,e in enumerate(dalex_server_times) if i%2==1])
    dalex_server_times = np.vstack([dalex_iid,dalex_non_iid]).mean(0)

    shap_iid =     np.array([e for i,e in enumerate(shap_server_times) if i%2==0])
    shap_non_iid = np.array([e for i,e in enumerate(shap_server_times) if i%2==1])
    shap_server_times = np.vstack([shap_iid,shap_non_iid]).mean(0)

    train_times_iid =     np.array([e for i,e in enumerate(train_times) if i%2==0])
    train_times_non_iid = np.array([e for i,e in enumerate(train_times) if i%2==1])
    train_times = np.vstack([train_times_iid,train_times_non_iid]).mean(0)


    shap_times_iid = np.array([e for i,e in enumerate(shap_client_times) if i%2==0])
    shap_times_non_iid = np.array([e for i,e in enumerate(shap_client_times) if i%2==1])
    shap_client_times = np.vstack([shap_times_iid,shap_times_non_iid]).mean(0)

    dalex_times_iid = np.array([e for i,e in enumerate(dalex_client_times) if i%2==0])
    dalex_times_non_iid = np.array([e for i,e in enumerate(dalex_client_times) if i%2==1])
    dalex_client_times = np.vstack([dalex_times_iid,dalex_times_non_iid]).mean(0)

    # plot the times
    plt.plot(np.arange(3),train_times,ls='--')
    plt.scatter(np.arange(3),train_times,label="Train Time",marker="d")
    plt.ylabel("Minutes")
    plt.xlabel("Clients")
    plt.legend()
    plt.title("Training 150 rounds")
    ticks = [0,1,2]
    plt.xticks(ticks,[str(8*(i+1)) for i in ticks])
    plt.grid(True)

    plt.savefig(os.path.join(dt_folder,"train_time.png"))
    plt.clf()

    for el in zip(["shap","dalex"],[shap_server_times,dalex_server_times]):
        plt.plot(np.arange(3),el[1],ls=('--' if el[0]=="shap" else "-"),label=el[0]+" Server Time")
        plt.scatter(np.arange(3),el[1],marker="d")
    plt.ylabel("Minutes")
    plt.xlabel("Clients")
    plt.legend()
    plt.title("Explaining 2000 samples")
    ticks = [0,1,2]
    plt.boxplot(dalex_client_times,positions=[0,1,2])
    plt.boxplot(shap_client_times,positions=[0,1,2])
    plt.xticks(ticks,[str(8*(i+1)) for i in ticks])
    plt.grid(True)
    plt.savefig(os.path.join(dt_folder,"explaining_times.png"))



    print(" ".join(["Expected time for adult is",
                    str(np.sum([train_times,shap_server_times,dalex_server_times])/60)[:4],
                    "HOURS"]))
    t = [train_times.sum(),shap_server_times.sum(),dalex_server_times.sum()]
    total_t = np.sum(t)
    print("time for:")
    print("train","\tshap","\tdalex")
    print(*[str(e.round(2))+"%, " for e in (t/total_t * 100)])
