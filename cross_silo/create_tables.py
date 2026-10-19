# %%
import os

# %%


# %%
#import numpy pandas json

import pandas as pd
import numpy as np
import json
import time

# %%
# print the current working directory
#os.chdir("..")
print(os.getcwd())   

# %%


dt_name = 'dutch'
dt_name = 'adult'
dt_name = 'dry_bean'
dt_name = "diamond_1_hot"
dt_name = "adult_TargEncoded"
#f_dt_name = os.path.join("FEDEXAI","crossdevice", dt_name)
dt_names = ['adult', 'dutch', 'dry_bean', 'diamond_1_hot', 'adult_TargEncoded']
for dt_name in dt_names:
    if dt_name not in ['adult', 'dutch']:
        f_dt_name = os.path.join(dt_name)
    else:
        f_dt_name = os.path.join("crossdevice", dt_name)
    #with open(f_dt_name+"_all_shap_values_dict_new.json", "r") as f:
    print(os.getcwd())
    dict_F_name = f_dt_name+"_all_shap_values_dict_new.json" if dt_name in ["dutch","adult"] else f_dt_name+"_all_shap_values_dict.json"
    with open(dict_F_name, "r") as f:
        loaded_dict = json.load(f)

    all_shap_values_dict_loaded = {k: {k2: np.array(v2) for k2, v2 in v.items()} for k, v in loaded_dict.items()}

    # %%
    dt_name

    # %%


    if dt_name=='adult':
        feat_names =['AGEP', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'COW_1.0', 'COW_2.0',
        'COW_3.0', 'COW_4.0', 'COW_5.0', 'COW_6.0', 'COW_7.0', 'COW_8.0',
        'SCHL_1.0', 'SCHL_2.0', 'SCHL_3.0', 'SCHL_4.0', 'SCHL_5.0', 'SCHL_6.0',
        'SCHL_7.0', 'SCHL_8.0', 'SCHL_9.0', 'SCHL_10.0', 'SCHL_11.0',
        'SCHL_12.0', 'SCHL_13.0', 'SCHL_14.0', 'SCHL_15.0', 'SCHL_16.0',
        'SCHL_17.0', 'SCHL_18.0', 'SCHL_19.0', 'SCHL_20.0', 'SCHL_21.0',
        'SCHL_22.0', 'SCHL_23.0', 'SCHL_24.0', 'MAR_1.0', 'MAR_2.0', 'MAR_3.0',
        'MAR_4.0', 'MAR_5.0', 'RAC1P_1.0', 'RAC1P_2.0', 'RAC1P_3.0',
        'RAC1P_4.0', 'RAC1P_5.0', 'RAC1P_6.0', 'RAC1P_7.0', 'RAC1P_8.0',
        'RAC1P_9.0','BIAS',""]
    elif dt_name=='dutch':
        long_feat_names = [
            "age","household_position","household_size","prev_residence_place","citizenship","country_birth","edu_level","economic_status","cur_eco_activity","Marital_status","sex_binary",'bias'
            ]
        feat_names = [
            "Age","HousePos","FamSize","PrevResPlc","Citz","BirthCtry","Edu","EcoSts","Job","MaritSts","Sex",''
            ]
    elif dt_name=='dry_bean':
        feat_names = ['A', #Area
            'P', #Perimeter
            'MAxL', #Major Axis Length
            'mAxL', # Minor Axis Length
            'AspctRt', #Aspect Ratio
            'Ecc', #Eccentricity
            'CnvxA', #Convex Area 
            'EqDiam', #Equivalent Diameter
            'Ext', #Extent
            'Sol', # Solidity 
            'Rnd', # Roundness 
            'Cmpct', # Compactness abbreviated as 
            'SF1', # Shape Factor 1
            'SF2', # Shape Factor 2
            'SF3', # Shape Factor 3
            'SF4' # Shape Factor 4
            ]
    elif dt_name=='diamond_1_hot':
        feat_names = ['Carat', # the weight of the diamond 
                    'Clrty', # the clarity of the diamond
                    'Dpt', # the depth of the diamond
                    'Tab', # the table of the diamond
                    'Price',# the price of the diamond
                    'Obs_x', # the x dimension of the diamond
                    'Obs_y', # the y dimension of the diamond
                    'Obs_z', # the z dimension of the diamond
                    'C_D', # the color of the diamond
                    'C_E',
                    'C_F',
                    'C_G',
                    'C_H',
                    'C_I',
                    'C_J']
    elif dt_name=='adult_TargEncoded':
        long_feat_names = ['Age', 'CapitalGain', 'CapitalLoss', 'HoursPerWeek', 'WorkClss', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race','Sex', 'NativeCountry']
        feat_names = ['Age', 'CGain', 'CLoss', 'WRKH', 'Job', 'Edu', 'MaritSts', 'Occ', 'Rel', 'Race', 'Sex', 'BirthCtry']


    else: #this else works for the long feature names of the cross-silo datasets (dry_bean, adult, diamond_1_hot)

        exp_path = os.path.join("Experiments",dt_name)

        feat_names_f = dt_name+"_names.txt"
        # inside the file there are the names of the features separated by a comma
        feat_names_f = os.path.join(exp_path,feat_names_f)
        with open(feat_names_f,"r") as f:
            feat_names = f.read().split(",")
        # using the map function to trim the names
        feat_names = list(map(lambda x: x.strip(),feat_names))
    print(feat_names)

    # %%

    props = [
            "E_hat",
            "E_weighted", # the explanation is the average of the explanations weighted by the f1 score
            "E_f1_macro", # the explanation is the average of the explanations weighted by the f1 score MACRO


            "E_fait_weighted", # the explanation is the average of the explanations weighted by the faithfulness
            "E_avg_t_0",       # the explanation is the average of the explanations filtered on the faithfulness > 0
                            # if there are not explanations that have 
            "E_mixed",         # 
            "E_mix_f1",
            "E_Q1",
            
            #"E_avg_t_06",
            #"E_france_W",
            #"E_harm",
            ]

    # %%
    if dt_name ==  'adult':
        f_dt_name+="_100_income"


    # %%
    f_dt_name,dt_name

    # %%
    import json

    with open(f_dt_name+"_indexes_per_calss.json", "r") as f:
        indexes_per_calss_list = json.load(f)

    indexes_per_calss = {k: np.array([np.array(v1) for v1 in v]) for k, v in indexes_per_calss_list.items()}
    with open(f_dt_name+"_activations_of_models.json", "r") as f:
        activations_of_models_list = json.load(f)



    activations_of_models = {k: {float(k1): np.array(v1) for k1, v1 in v.items()} for k, v in activations_of_models_list.items()}

    # sobstitute every key with the same but instead of % there is _
    # for the keys that are in the dictionaries activations_of_models, indexes_per_calss and all_shap_values_dict_loaded
    if dt_name.lower() == 'dutch':
        
        for k in list(indexes_per_calss.keys()):
            k1 = k.replace("5%","5.0%")
            k1 = k1.replace("099%","0.99%")
            k1 = "_".join(list(np.array(k1.replace("_nodi","").split("%"))[[1,0]]))
            if "iid" not in k1:
                k1 = k1.replace("_","_imb_")
            
            indexes_per_calss[k1] = indexes_per_calss.pop(k)
            activations_of_models[k1] = activations_of_models.pop(k)
            all_shap_values_dict_loaded[k1] = all_shap_values_dict_loaded.pop(k)

        



    # %%
    focus = list(all_shap_values_dict_loaded.keys())
    if dt_name.lower() == "dutch":
        focus = [
           '50_imb_0.99',
           '50_imb_5.0',
           '50_iid',
           '70_imb_0.99',
           '70_imb_5.0',
           '70_iid',
          '150_imb_0.99',
          '150_imb_5.0',
          '150_iid',

        ]
        print(focus)

    # %%
    focus

    # %%


    # %%


    # %%
    import importlib
    import iflash as flx
    #
    importlib.reload(flx)
    from iflash import *

    # %%
    props

    # %%
    import matplotlib.pyplot as plt

    # %%
    # set some parameters
    plt.rcParams['figure.dpi'] = 100  # Screen display
    plt.rcParams['savefig.dpi'] = 300  # Saved figure

    # %%
    # Set font sizes
    plt.rcParams['font.size'] = 10  # Base font size
    plt.rcParams['axes.titlesize'] = 10  # Subplot titles
    plt.rcParams['axes.labelsize'] = 10  # Axis labels
    plt.rcParams['xtick.labelsize'] = 9.5  # X tick labels
    plt.rcParams['ytick.labelsize'] = 8  # Y tick labels
    plt.rcParams['legend.fontsize'] = 10  # Legend text

    # %%
    plt.rcParams.update({
            'pdf.fonttype': 42,  # Ensures text is saved as text, not curves
            'ps.fonttype': 42,   # Same for PostScript output
        })

    # %%
    name_dataset = "ACSI" if "adult" in dt_name.lower() else "Diamonds" if "diamond" in dt_name.lower() else "Dry Bean" if "bean" in dt_name.lower() else "ACSI" if "income" in dt_name.lower() else "Dutch" if "dutch" in dt_name.lower() else "dataset name"
    def plot_global_and_differencies_nb(exp,all_shap_values_dict,activations_of_models,proposals,indices,show=False,feat_names=None,dt_name="dt_name",selection = None,vertical_single_size=5,limit_features=10):
        '''this function is used to plot the global explanation of the model and the differences between the global explanations of the server and the aggregations'''
        # get the global explanation
        g_expl_s,g_expl_by_class_proposals,global_s,global_props = get_global_explanation(exp,all_shap_values_dict,indices,activations_of_models,proposals)
        # create the figure
        # plot the global explanation
        
        differencies = []
        kendalls = []
        frobs = []
        shap_CS = []
        spearmanss = []
        single_kendall = []
        single_frobs = []
        single_CS = []
        single_spearmans = []

        for i,p in enumerate(proposals):
            diff = g_expl_s-g_expl_by_class_proposals[p]
            differencies.append(diff)
            kendalls.append(compare_heatmaps_kendalltau(g_expl_s,g_expl_by_class_proposals[p]))
            frobs.append(compare_heatmaps_Frobenius(g_expl_s,g_expl_by_class_proposals[p]))
            shap_CS.append(compare_heatmaps_cosinesim(g_expl_s,g_expl_by_class_proposals[p]))
            spearmanss.append(compare_heatmaps_spearmanr(g_expl_s,g_expl_by_class_proposals[p]))

            single_kendall.append(single_kendalltau_f(global_s,global_props[p]))
            single_frobs.append(single_Frobs_f(global_s,global_props[p]))
            single_CS.append(single_cosinesim_f(global_s,global_props[p]))
            single_spearmans.append(single_spearmanr_f(global_s,global_props[p]))


        vmax,vmin = np.max(differencies),np.min(differencies)
        print("kendalltau's have",len(kendalls),"elements, and shape ",kendalls[0].shape)
        print("diff has ",len(frobs),"elements, and shape ",frobs[0].shape)
        print("shap_CS has ",len(shap_CS),"elements, and shape ",shap_CS[0].shape)
        print("spearmanss has ",len(spearmanss),"elements, and shape ",spearmanss[0].shape)

        """
        fig = plt.figure(figsize=(10,5))
        plt.imshow(g_expl_s.T,interpolation='none',cmap='coolwarm', rasterized=True,aspect='auto',vmin=-1,vmax=1)
        plt.yticks(np.arange(len(feat_names)),feat_names)
        plt.suptitle("Global explanation of the model\nby class (of the server)"+"\nfor setting "+" ".join([str(elemento) for elemento in exp]),fontsize=20)
        # put the values of the global explanation as string in the heatmap
        for i in range(g_expl_s.shape[0]):
            for j in range(g_expl_s.shape[1]):
                plt.text(i,j,str(np.round(g_expl_s[i,j],2)),ha='center', va='center', color='black')
        fig.subplots_adjust( left=None, bottom=None,  right=None, top=None, wspace=None, hspace=None)
        if show:
            plt.show()
            plt.close()
        else:
            if isinstance(exp,tuple) or isinstance(exp,list):
                expt = "_".join([str(elemento) for elemento in exp])
            else:
                expt = exp
            plt.savefig(dt_name+"_"+expt+"_global_explanation.pdf")
            plt.close()

        """
        # now we create a figure with the global feature importance for each class, of the server and the aggregations
        # create an image with as many rows as the proposals plus the server
        # and as many columns as the classes
        if selection is not None:
            new_proposals = selection
        else:
            new_proposals = proposals

        # get the max and min of the global explanation for each class, among the server and all the proposals
        fi_vmax = []
        fi_vmin = []
        for i in range(g_expl_s.shape[0]):
            fi_vmax.append(np.max([np.max(g_expl_s[i])]+[np.max(g_expl_by_class_proposals[p][i]) for p in new_proposals]))
            fi_vmin.append(np.min([np.min(g_expl_s[i])]+[np.min(g_expl_by_class_proposals[p][i]) for p in new_proposals]))
            print("the max and min of the global explanation are",fi_vmax[-1],fi_vmin[-1])
        #for i,p in enumerate(range(len(proposals))):
            #fi_vmax.append(np.max(differencies))
            #fi_vmin.append(np.min(differencies))
        
        assert len(fi_vmax) == len(fi_vmin), "the length of the max and min of the global explanation is different"        
        # in the other rows we plot the global explanation of the aggregations per class
        j=0
        server_color = "#005AB5"
        choosen_color = "#DC3220"
        def order_features_by_magnitude(expl, feat_names):
            """
            Orders features by absolute magnitude for a given class
            Returns ordered indices and ordered feature names
            """
            abs_magnitudes = np.abs(expl)
            ordered_indices = np.argsort(abs_magnitudes)[::-1]
            ordered_feat_names = [feat_names[i] for i in ordered_indices]
            return ordered_indices, ordered_feat_names

        # now we do a simpler plot; for each proposal in selection we plot the importance of the features and the one of the server
        j=0
        for jc,p in enumerate(proposals):
            if p not in new_proposals:
                continue
            if len(indices)>2: # this is for dry bean and diamonds
                fig = plt.figure(figsize=(4,3*(len(indices)//2)))
                gs = GridSpec(len(indices)//2+len(indices)%2,2,figure=fig)
            elif len(feat_names)<15: # this is for Dutch and Adult
                fig = plt.figure(figsize=(4,3))
                gs = GridSpec(1,2,figure=fig)
            else: # this is for ACSI
                fig = plt.figure(figsize=(4,3))
                gs = GridSpec(1,2,figure=fig)
            axes = []
            print(len(indices)//2+len(indices)%2,2)
            if len(indices)>2:
                for row in range(len(indices)//2):
                    for col in range(2):
                        ax = fig.add_subplot(gs[row,col])
                        axes.append(ax)
                if len(indices)%2 != 0:
                    ax = fig.add_subplot(gs[-1,0])
                    axes.append(ax)
            elif len(feat_names)<15:
                for col in range(2):
                    ax = fig.add_subplot(gs[0,col])
                    axes.append(ax)
            else:
                for row in range(2):
                    ax = fig.add_subplot(gs[0,row])
                    axes.append(ax)
            labl = res_d[p]
            axes = np.array(axes).flatten()
            print(len(axes))

            order_features = True
            for i,expl in enumerate(g_expl_by_class_proposals[p]):
                ax = axes[i]
                if order_features:
                    ordered_idx, ordered_feat = order_features_by_magnitude(expl, feat_names)
                else:
                    ordered_idx = np.arange(len(feat_names))
                    ordered_feat = feat_names
                if limit_features is not None:
                    ordered_idx = ordered_idx[:limit_features]
                    ordered_feat = ordered_feat[:limit_features]
                df = pd.DataFrame(data={"Feature": ordered_feat ,
                                        p: expl[ordered_idx],"Server":g_expl_s[i][ordered_idx]})
                x = np.arange(len(ordered_feat))
                width = 0.4
                ax.bar(x-width/2,df[p],width,label=res_d[p],color=choosen_color)
                ax.bar(x+width/2,df["Server"],width,label="Audit**",color=server_color)
                # the increment is 5% of the absolute max value
                increment = (np.max(np.abs([fi_vmin[i],fi_vmax[i]])))/100*5

                ax.set_ylim(fi_vmin[i]-increment,
                            fi_vmax[i]+increment)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                #if i==0:
                #    ax.legend()#fontsize=16)
                ax.set_title(   
                                f"Class {i}")
                #                f"{res_d[p]} - "
                #                f"$ShapGap_{{CosSim}}$ {shap_CS[jc][i]:.2f},\n"
                #                f"$ShapGap_{{L2}}$ {np.mean(frobs[jc][i]):.2f}")
                if limit_features>15:
                    ax.set_xticks(x, ordered_feat, rotation=90)#, fontsize=10)
                else:
                    ax.set_xticks(x, ordered_feat, rotation=90)#,fontsize=18)
            #plt.suptitle(name_dataset+" setting "+
            #             (" ".join([str(elemento) for elemento in (exp.replace("imb","$\\alpha=$").split("_") if isinstance(exp,str) else exp) ])
            #              if "%" not in exp else fancytitle(exp.split("%")))+" - "+labl)#,fontsize=18)

            handles, labels = ax.get_legend_handles_labels()
            # For the entire figure
            fig.legend(handles, labels,
                    bbox_to_anchor=(0.55, 0.0),  # Center, bottom of figure
                    loc='upper center',
                    ncol=2,  # Adjust based on number of items#
                    fontsize=plt.rcParams['legend.fontsize'])

            # Make room at the bottom
            plt.tight_layout()
            #plt.subplots_adjust(bottom=0.2)  

            if len(indices)%2 != 0:
                # now adjust the last plot if the # of classes is odd
                ax1 = axes[0]
                ax2 = axes[1]
                pos1 = ax1.get_position()
                pos2 = ax2.get_position()

                # Calculate the new position for the last subplot
                new_x0 = (pos1.x0 + pos2.x0) / 2  # Center horizontally
                new_x1 = (pos1.x1 + pos2.x1) / 2
                new_width = new_x1 - new_x0  # Keep the same width as others
                new_y0 = axes[-1].get_position().y0  # Retain the vertical position
                new_y1 = axes[-1].get_position().y1
                axes[-1].set_position([new_x0, new_y0, new_width, new_y1 - new_y0])

            if isinstance(exp,tuple) or isinstance(exp,list):
                expt = "_".join([str(elemento) for elemento in exp])
            else:
                expt = exp
            name_fig = dt_name+"_"+expt+"_global_explanation_by_class_"+p+".pdf"
            print("this would be "+name_fig)
            if show:
                plt.show()
                plt.close()
            else:
                name_fig = os.path.join("Experiments",name_fig)
                plt.savefig(name_fig,
                            bbox_inches='tight')
                print("saved the figure",name_fig)
                name_fig = name_fig[:-3]+"png"
                plt.savefig(name_fig)
                print("saved the figure",name_fig)
                plt.close()
            plt.clf()
            j+=1

        return kendalls,frobs,shap_CS,spearmanss,differencies,single_kendall,single_frobs,single_CS,single_spearmans

    # %%
    all_kendals_frobs = {}
    # test the function
    #focus = [#"012_iid",
    #         "012_imb_5.0"#,"012_imb_0.99"
    #         ]
    #focus = [f for f in focus if "150" in f]
    for exp in focus:
        print(exp)
        k,f,cs,sprmn,d,single_k,single_f,single_cs,single_sprmn = plot_global_and_differencies_nb(exp=exp,
                                                                    all_shap_values_dict = all_shap_values_dict_loaded,
                                                                    activations_of_models = activations_of_models,
                                                                    proposals=props,
                                                                    feat_names=feat_names,
                                                                    indices=indexes_per_calss[exp],
                                                                    dt_name = dt_name,
                                                                    show=True,#(True if exp==focus[0] else False),
                                                                    selection=["E_Q1","E_fait_weighted"],
                                                                    limit_features=10)
        all_kendals_frobs[exp] = (k,f,cs,sprmn,d,single_k,single_f,single_cs,single_sprmn)
        # _differences_global_explanation.pdf,
        # _global_explanation_by_class.pdf,
        # _global_explanation.pdf
        
        break

    # %%


    # %%
    all_metric_recomputed = {}
    for exp in focus:
        all_metric_recomputed[exp] = {}
        for metric_name,metric_fun in zip(
            ["Cosine",
            "Kendall",
            "L2",
            "Spearman"],
            [compare_heatmaps_cosinesim,
                compare_heatmaps_kendalltau,
                compare_heatmaps_Frobenius,
                compare_heatmaps_spearmanr]):
            all_metric_recomputed[exp][metric_name] = {}
            for p in props:
                all_metric_recomputed[exp][metric_name][p] = metric_fun(
                    all_shap_values_dict_loaded[exp]["Server"],
                    all_shap_values_dict_loaded[exp][p]
                )
            



    # %%
    print(name_dataset)

    if name_dataset not in ["ACSI" ,"Dutch"]:
        focus = [
        '012_imb_0.99',
        '012_imb_5.0',
        '012_iid',
        '016_imb_0.99',
        '016_imb_5.0',
        '016_iid',
        ]

    # %%


    # %%


    # %%
    prop2command = {
        'E_mixed':"\\emix",
    'E_avg_t_0':"\\ezeroavg",
    'E_f1_macro':"\\efmacro",
    'E_weighted':"\\ef",
    'E_hat':"\\eavg",
    'E_mix_f1':"\\emixf",
    "Server":"\\audit",
    "E_fait_weighted":"\\efaith",
    'E_Q1':"\\eqi",
        
    }

    # %%

    exp0 = list(all_metric_recomputed.keys())[0]
    prefix = "\\begin{table*}[h]\n\\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{|c|c|c|>{\columncolor{gray!20}}c|c|>{\columncolor{gray!20}}c|c|>{\columncolor{gray!20}}c|c|>{\columncolor{gray!20}}c|c|}\n"
    prefix += "\\hline\n"
    prefix += "Dataset & Clients & Distribution & "+"&".join([prop2command[p] for p in props])+" \\\\"
    prefix += "\\hline\n"
    suffix = " \n"
    suffix += "\\hline\n"
    suffix += "\t\\end{tabular}\n}\n\\end{table*}"

    string_to_write_to_file = ""
    for metric_name in all_metric_recomputed[exp0].keys():
        #print(metric_name)
        #print(prefix)
        string_to_write_to_file+="\n%"+(metric_name+" ")*5+"\n"
        string_to_write_to_file += prefix+"\n"
        for idx_exp,exp in enumerate(focus):
            print(exp)
            vals = np.array([all_metric_recomputed[exp][metric_name][p] for p in props])
            # find the best proposal
            if metric_name in ["L2"]:
                best_value = np.min(vals.mean(axis=1))
            else:
                best_value = np.max(vals.mean(axis=1))
            if dt_name == "adult":
                nc = str(51)
                distrs = "\\textit{natural}"
            else:
                if idx_exp%3==0:
                    nc = str(int(exp.replace("imb","$\\alpha=$").split("_")[0]))
                    nc = "\\multirow{3}{*}{"+nc+"}"
                else:
                    nc = ""
                distrs = "".join(exp.replace("imb","$\\alpha=$").split("_")[1:])
            

            row = " & ".join(["",nc,distrs]
                            +[(f"{v.mean():.2f}" if not v.mean().round(2) == best_value.round(2) else '\\textbf{'+f"{v.mean():.2f}"+'}')+f"$\\pm${v.std():.2f}" for j_max,v in enumerate(vals)])
            if idx_exp==0:
                # place the name of the dataset on a multirow
                row= "\\multirow{"+str(len(focus))+"}{*}{"+name_dataset+"}"+row
                
            #print(row+" \\\\")
            string_to_write_to_file+=row+" \\\\ \n"
            if idx_exp==2:
                #print("\cline{2-11}")
                string_to_write_to_file+="\cline{2-11}\n"
        
        #print(suffix)
        string_to_write_to_file+=suffix+"\n"
    filen = f_dt_name+"_table.tex"
    with open(filen,"w") as f:
        f.write(string_to_write_to_file)
        print("written the table to file",f_dt_name+"_table.tex")
        f.close()
        print("closed the file",filen)
        

    # %%

    # %%



