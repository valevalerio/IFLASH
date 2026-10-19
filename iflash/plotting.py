"""Figures used to analyse the experiments (explanations, Faithfulness, ShapGap).

Most functions take ``all_shap_values_dict[exp]``, a dictionary with the keys
``"Server"`` (Audit explanations, ``(n_samples, n_features)``), ``"Clients"``
(list of client explanations) and one key per aggregation strategy.
With ``show=False`` the figures are saved as PDF/PNG in the working directory.
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

from .aggregation import get_global_explanation, normalize_weightOLD, normalize_weights
from .metrics import (compare_heatmaps, compare_heatmaps_cosinesim, compare_heatmaps_Frobenius,
                      compare_heatmaps_kendalltau, compare_heatmaps_spearmanr, compute_diffs,
                      single_cosinesim_f, single_Frobs_f, single_kendalltau_f, single_spearmanr_f)
from .naming import colors, dataset_display_name, exp_filename, exp_label, fancytitle, res_d


# ------------------------------------------------------------- heatmaps
def plot_heatmap(hm, feat_names, title='', vmin=None, vmax=None):
    """Heatmap of an explanation matrix with the mean importance per feature on top.

    ``hm`` can be a client's SHAP matrix, the server's, or any aggregation.
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={'height_ratios': [1, 4]})
    title = title + "\n" + "Mean:" + str(np.round(np.mean(hm), 3)) + " Std:" + str(np.round(np.std(hm), 3))
    fig.suptitle(title, y=1.02)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(hm, xticklabels=feat_names, yticklabels=False, ax=ax[1], vmin=vmin, vmax=vmax, cmap=cmap)

    # shrink the upper bar plot to align it with the heatmap (which has a colorbar)
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0 - 0.05, box.width * 0.8, box.height * 1.2])
    sns.barplot(x=feat_names, y=np.mean(hm, axis=0), ax=ax[0], color=sns.color_palette()[0])
    ax[0].set_xticks([])
    ax[1].set_xticks(np.arange(len(feat_names)) + 0.5)
    ax[1].set_xticklabels(feat_names, rotation=45, ha='right')

    plt.show()
    plt.close()


def plot_heatmap_and_focus(hm, feat_names, shap_val_dict=None, title='', vmin=None, vmax=None,
                           all_means=[], idx=0, names_exp=[], fontsize=16):
    """Like :func:`plot_heatmap`, plus a horizontal bar plot comparing the mean
    absolute difference of every method (``all_means``), highlighting ``idx``."""
    fig, ax = plt.subplots(2, 2, figsize=(15, 5), gridspec_kw={'height_ratios': [1, 4]})
    title = title + "\n" + "Mean:" + str(np.round(np.mean(hm), 3)) + " Std:" + str(np.round(np.std(hm), 3))
    fig.suptitle(title, y=1.02, fontsize=fontsize)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    heat_map_ax, mean_bar_ax, means_ax, empty_ax = ax[1, 0], ax[0, 0], ax[0, 1], ax[1, 1]

    sns.heatmap(hm, xticklabels=feat_names, yticklabels=False, ax=heat_map_ax, vmin=vmin, vmax=vmax, cmap=cmap)
    box = mean_bar_ax.get_position()
    mean_bar_ax.set_position([box.x0, box.y0 - 0.05, box.width * 0.8, box.height * 1.2])
    heat_map_ax.set_ylabel("Samples", fontsize=fontsize)

    print(feat_names)
    sns.barplot(x=np.arange(hm.shape[1]), y=np.mean(np.abs(hm), axis=0), ax=mean_bar_ax, color=sns.color_palette()[0])
    mean_bar_ax.set_xticks([])
    mean_bar_ax.set_title("Mean of the features", fontsize=fontsize)
    heat_map_ax.set_xticks(np.arange(len(feat_names)) + 0.5)
    heat_map_ax.set_xticklabels(feat_names, rotation=45, ha='right', fontsize=fontsize)

    for i in range(len(all_means)):
        alpha = 0.2 if i != idx else 1.0
        means_ax.barh(y=names_exp[i], width=all_means[i], color=sns.color_palette()[0], alpha=alpha)
    means_ax.set_yticks(np.arange(len(names_exp)), names_exp)
    means_ax.set_ylabel("")
    means_ax.set_xlabel("Mean of \nabsolute $\\Delta$ features", fontsize=fontsize)
    means_ax.grid(axis='x', linestyle='--')

    if shap_val_dict is None:
        empty_ax.axis('off')
    plt.show()


def plot_differences(dictionary, proposals=["E_hat"], axes=None, colorbar=None, comparison=compare_heatmaps):
    """Symmetric matrix of the differences between every client, the server and
    the ``proposals``, drawn on ``axes``.

    Note: the client-vs-client block always uses :func:`compare_heatmaps`.
    """
    nc = len(dictionary["Clients"])
    server_idx = nc
    nproposals = len(proposals)
    differences_matrix = np.zeros((1 + nc + nproposals, 1 + nc + nproposals))
    differences_matrix[:nc, :nc] = compute_diffs(dictionary["Clients"])
    contribs = differences_matrix + differences_matrix.T
    # clients sorted by their average difference with the other clients
    row_mean = np.mean(contribs[:nc, :nc], axis=1)
    sorted_names = ["C" + str(i + 1) for i in np.argsort(row_mean)[::-1]]
    print(sorted_names)

    for i in range(nc):
        differences_matrix[i, server_idx] = comparison(dictionary["Clients"][i], dictionary["Server"])
    for i, prop in enumerate(proposals):
        differences_matrix[nc + i + 1, server_idx] = comparison(dictionary["Server"], dictionary[prop])
        for i_c in range(nc):
            differences_matrix[i_c, nc + 1 + i] = comparison(dictionary["Clients"][i_c], dictionary[prop])

    # for completeness, the comparisons between the proposals
    print([p for p in proposals])
    print([dictionary[p].shape for p in proposals])
    hms = [_ for _ in range(nc)] + [dictionary["Server"]] + [dictionary[p] for p in proposals]
    prop_idxs = [nc + i + 1 for i in range(nproposals)]
    for a, b in itertools.combinations(prop_idxs, 2):
        differences_matrix[a, b] = comparison(hms[a], hms[b])

    differences_matrix = differences_matrix + differences_matrix.T
    axes.set_xlabel("Clients ordered by difference (decr):\n" + ", ".join(sorted_names[:nc // 4 * 3]))
    im = axes.imshow(differences_matrix, interpolation='none', rasterized=True)
    axes.set_yticks(np.arange(nc + 1 + nproposals),
                    ["C" + str(i + 1) for i in range(nc)] + ["$E_S$"] + [res_d[p] for p in proposals], fontsize=13)
    axes.set_xticks([], [])
    if colorbar:
        axes.figure.colorbar(im, ax=axes)


# ------------------------------------------------------------ faithfulness
def _share_above(values, threshold):
    """Share of samples of each faithfulness array above ``threshold``."""
    return np.sum(np.array(values) > threshold) / len(values)


def plot_faithfulness_histograms_and_thresholds(results_dict, base_value, show=True, exp="Experiment_name"):
    """Left: Faithfulness distributions of the server and the clients.
    Right: share of explanations with Faithfulness above 0 and above 0.7.

    ``results_dict = {"Server": [array], "Clients": [array, ...]}``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    sns.histplot(results_dict["Server"][0], ax=ax, kde=True, stat="density", label="Server", linewidth=3)
    ax.set_xlim(-1, 1)
    for client in results_dict["Clients"]:
        sns.histplot(client, ax=ax, kde=True, stat="density")
    ax.set_title(exp + "\n" + "Faithf.  base value:" + str(base_value))
    ax.legend()

    ax = axes[1]
    n_clients = len(results_dict["Clients"])
    ax.set_xticks(np.arange(n_clients + 1))
    ax.set_xticklabels(["Server"] + ["C" + str(i + 1) for i in range(n_clients)])
    server = results_dict["Server"][0]
    ax.bar(np.arange(n_clients + 1),
           [_share_above(server, 0)] + [_share_above(c, 0) for c in results_dict["Clients"]],
           alpha=0.9)
    ax.bar(np.arange(n_clients + 1),
           [_share_above(server, 0.7)] + [_share_above(c, 0.7) for c in results_dict["Clients"]],
           alpha=0.7, label=">0.7")
    ax.legend()
    ax.grid(axis='y', linestyle='--')
    ax.set_title("Percentage of faithfulness values above 0 and 0.7")
    fig.suptitle(exp + "\n" + "Faithf.  base value:" + str(base_value), y=1.02, fontsize=18)
    if show:
        plt.show()


def _faithfulness_matrix(faits):
    """``(n_samples, 1 + C)`` matrix: server Faithfulness first, then the clients'."""
    ff = np.zeros((len(faits["Server"][0]), len(faits["Clients"]) + 1))
    ff[:, 0] = faits["Server"][0]
    for c in range(len(faits["Clients"])):
        ff[:, c + 1] = faits["Clients"][c]
    return ff


def plot_faith(exp, faits, all_shap_values_dict, te_set, activations=None, show_samples=10, chosen_idxs=None,
               show=True, vmax=None, dt_name="dt_name", feat_names=None):
    """Inspect the samples on which clients disagree the most.

    For each class (``chosen_idxs[i]`` = indices of class ``i``) the samples
    with the highest std of Faithfulness across clients are selected (true
    positives first). Produces: (1) std/CDF/boxplot of Faithfulness per class,
    (2) a strip figure with samples, activations, Faithfulness, Audit, closest
    and furthest client, E_avg, Delta and the difference matrix, (3) bar plots
    of single true/false positive explanations. Returns the plotted indexes.
    """
    fontsize = 20
    ff = _faithfulness_matrix(faits)
    print(ff.shape)
    plt.clf()

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    ax_density, ax_cdf, ax_boxplot = axes
    cmap = sns.color_palette("tab10", len(chosen_idxs))

    if chosen_idxs is not None:
        samples_per_class = show_samples // len(chosen_idxs)
        print(samples_per_class, "samples per class")
        random_idxs, all_tp, all_fp, yticks = [], [], [], []
        for i in range(len(chosen_idxs)):
            idxs = np.array(chosen_idxs[i])
            faiths = ff[idxs]

            # samples of this class sorted by the std of the Faithfulness across server and clients
            sorted_idxs = np.argsort(np.std(faiths, axis=1))[::-1]
            values = np.std(faiths[sorted_idxs], axis=1)
            values = values[~np.isnan(values)]
            ax_density.plot(values, label="std(faits)\n for class " + str(i))

            return_dict = ax_boxplot.boxplot(values, positions=[i], patch_artist=True, showfliers=True)
            patches = return_dict["boxes"]
            print(len(patches), "patches")
            if np.isnan(values).sum() > 0:
                print("class", i)
                print("Nans in the std of the faithfulness values")
            for patch in patches:
                patch.set_facecolor(cmap[i])

            # most discordant samples: true positives first, then false positives,
            # each sorted by the confidence of the prediction
            high_std_idxs = list(idxs[sorted_idxs[np.arange(samples_per_class)]])
            tp_first = [e for e in high_std_idxs if activations[e].argmax() == i]
            fp_last = [e for e in high_std_idxs if activations[e].argmax() != i]
            tp_first = sorted(tp_first, key=lambda x: activations[x].max(), reverse=True)
            fp_last = sorted(fp_last, key=lambda x: activations[x].max(), reverse=True)
            assert len(tp_first) + len(fp_last) == len(high_std_idxs)
            random_idxs = random_idxs + tp_first + fp_last
            all_tp += tp_first
            all_fp += fp_last
            yticks.append(len(random_idxs))

        if vmax is not None:
            ax_density.set_ylim(0, vmax)
            ax_boxplot.set_ylim(0, vmax)

        ax_boxplot.set_xticks(np.arange(len(chosen_idxs)), ["class " + str(i) for i in range(len(chosen_idxs))])
        for axx in [ax_density, ax_boxplot, ax_cdf]:
            axx.grid(True)
            axx.legend(fontsize=18)
            axx.tick_params(axis='both', which='major', labelsize=18)

        # cumulative distribution of the Faithfulness of server and clients, per class
        for i in range(len(chosen_idxs)):
            faiths = ff[np.array(chosen_idxs[i]), :]
            print(type(faiths), "type of faiths", faiths.shape, "shape of faiths")
            sns.kdeplot(faiths[:, 0], ax=ax_cdf, label="Server", cumulative=True, lw=3)
            for c in range(len(faits["Clients"])):
                label = "Clients, class " + str(i) if c == 0 else None
                sns.kdeplot(faiths[:, c + 1], ax=ax_cdf, label=label, cumulative=True, ls='--', color=cmap[i])
        ax_cdf.set_xlim(-1.5, 1.5)
        ax_cdf.set_ylim(-0.05, 1.05)
        ax_boxplot.set_title("std of the faithfulness values per class")
        ax_density.set_title("std of the faithfulness values")
        ax_cdf.set_title("Cumulative distribution function of the faithfulness values per class, ")
        ax_cdf.set_xlabel("faithfulness values")
        ax_cdf.set_ylabel("cumulative distribution function")
        ax_cdf.legend()
        if vmax is not None:
            for axe_ in [ax_density, ax_boxplot]:
                axe_.set_ylim(axe_.get_ylim()[0], vmax)

        plt.suptitle(exp + "\nstd of the client's faithfulness\n values of the explanations of the samples", fontsize=20)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(dt_name + "_" + exp + "_std_of_faithfulness_values.pdf")
        plt.close()

    # --- strip figure: samples, activations, faithfulness and explanations
    shap_exp = all_shap_values_dict[exp]
    fig, axes = plt.subplots(1, 9, figsize=(20, 10))
    (faithfulness_ax, samples_ax, activations_ax, server_ax, closest_C_ax,
     furthers_C_ax, e_hat_ax, delta_ax, recalculate_delta_ax) = axes

    samples_ax.set_xticks(np.arange(len(feat_names)), feat_names, rotation=90, ha='right')
    samples_ax.imshow(te_set.tensors[0][random_idxs, :], interpolation='none', rasterized=True)
    samples_ax.set_aspect('auto')
    samples_ax.set_title("$x$\nSamples", fontsize=fontsize)

    activations_ax.imshow(activations[random_idxs], interpolation='none', cmap='gray', rasterized=True)
    activations_ax.set_aspect('auto')
    activations_ax.set_title("$\\hat{f}(x)$\nActivations", fontsize=fontsize)
    activations_ax.set_xticks(np.arange(activations.shape[1]), ["class " + str(i) for i in range(activations.shape[1])],
                              rotation=90, ha='right')

    faithfulness_ax.imshow(ff[random_idxs, :], interpolation='none', vmin=-1, vmax=1, cmap='coolwarm', rasterized=True)
    print(ff[random_idxs[0], :], "worst")
    print(ff[random_idxs[-1], :])
    print()
    faithfulness_ax.set_yticks(np.arange(len(random_idxs)), random_idxs)
    x_ticks = ["Audit " + str(np.round(np.mean(ff[:, 0]), 2)) + "+-" + str(np.round(np.std(np.abs(ff[:, 0])), 2))] + \
              ["C" + str(i + 1) + " " + str(np.round(np.mean(ff[:, i + 1]), 2)) + "+-" + str(np.round(np.std(np.abs(ff[:, i + 1])), 2))
               for i in range(len(faits["Clients"]))]
    faithfulness_ax.set_xticks(np.arange(len(faits["Clients"]) + 1) + 0.5, x_ticks, rotation=90, ha='right')
    faithfulness_ax.set_aspect('auto')
    faithfulness_ax.set_title("faithfulness", fontsize=fontsize)

    server_ax.imshow(shap_exp["Server"][random_idxs, :], interpolation='none', rasterized=True)
    server_ax.set_aspect('auto')
    server_ax.set_title("$E_{S}$\naudit", fontsize=fontsize)

    e_hat_ax.imshow(shap_exp["E_hat"][random_idxs, :], interpolation='none', rasterized=True)
    e_hat_ax.set_aspect('auto')
    e_hat_ax.set_title("$E_{hat}$\n $\\frac{1}{" + str(len(faits["Clients"])) + "}(\\Sigma_{i} E_{Ci})$", fontsize=fontsize)

    delta_ax.imshow(shap_exp["Delta"][random_idxs, :], interpolation='none', cmap='coolwarm', rasterized=True)
    delta_ax.set_aspect('auto')
    delta_ax.set_title("$\\Delta$\ndifference", fontsize=fontsize)

    plot_differences(dictionary=shap_exp, proposals=["E_hat", "E_weighted"], axes=recalculate_delta_ax)
    recalculate_delta_ax.set_aspect('equal')

    # closest / furthest client w.r.t. the server (mean row-wise L2)
    diffs = [np.mean(compare_heatmaps_Frobenius(shap_exp["Server"], shap_exp["Clients"][i]))
             for i in range(len(faits["Clients"]))]
    closest, furthest = np.argmin(diffs), np.argmax(diffs)
    print("closest", closest, "furthest", furthest)
    closest_C_ax.imshow(shap_exp["Clients"][closest][random_idxs, :], interpolation='none', rasterized=True)
    closest_C_ax.set_aspect('auto')
    closest_C_ax.set_title("$E_{C" + str(closest + 1) + "}$\nclosest", fontsize=fontsize)
    closest_C_ax.set_xlabel("$avg(|\\delta_" + str(closest + 1) + "|):$\n" + str(np.round(diffs[closest], 2)), fontsize=fontsize)
    furthers_C_ax.imshow(shap_exp["Clients"][furthest][random_idxs, :], interpolation='none', rasterized=True)
    furthers_C_ax.set_aspect('auto')
    furthers_C_ax.set_title("$E_{C" + str(furthest + 1) + "}$\nfurthest", fontsize=fontsize)
    furthers_C_ax.set_xlabel("$avg(|\\delta_" + str(furthest + 1) + "|):$\n" + str(np.round(diffs[furthest], 2)), fontsize=fontsize)

    for ax in axes:
        if ax != faithfulness_ax:
            ax.set_yticks([])
        if ax not in [samples_ax, faithfulness_ax]:
            ax.set_xticks([])
    # horizontal lines separating the classes
    for ax in axes:
        if ax == recalculate_delta_ax:
            continue
        for y in yticks:
            ax.axhline(y - 0.5, color='black', lw=1.5)
    plt.suptitle(dt_name + " " + exp, y=1.02, fontsize=20)
    if show:
        plt.show()
    else:
        plt.savefig(dt_name + "_" + exp + "_faithfulness_strides.pdf")
    plt.close()

    # --- single-sample explanations of 3 true positives and 3 false positives
    all_tp, all_fp = np.array(all_tp), np.array(all_fp)
    np.random.shuffle(all_tp)
    np.random.shuffle(all_fp)
    single_examples = 3
    tp_idxs = np.random.choice(all_tp, single_examples)
    fp_idxs = np.random.choice(all_fp, single_examples)
    labs = np.array(["TP"] * single_examples + ["FP"] * single_examples)
    indexes = np.concatenate([tp_idxs, fp_idxs])
    print(len(tp_idxs), "true positives")
    print(len(fp_idxs), "false positives")
    print("indexes", indexes)
    print("labs", labs)
    assert len(indexes) == len(labs), "len indexes:" + str(len(indexes)) + " len labs:" + str(len(labs))
    for sample_idx, gold in zip(indexes, labs):
        print(gold, "sample", sample_idx)
        proposal = "E_hat"
        labels = ["Server", "Closest", "Furthest", proposal]
        all_explanation_df = pd.DataFrame()
        for i, single_explanation in enumerate([shap_exp["Server"], shap_exp["Clients"][closest],
                                                shap_exp["Clients"][furthest], shap_exp[proposal]]):
            labels[i] += ("\nClient" + str(i + 1) if i in [1, 2] else "")
            all_explanation_df[labels[i]] = single_explanation[sample_idx]
        plt.figure(figsize=(5, 7))
        single_sample_ax = all_explanation_df.plot(kind='barh', ax=plt.gca(),
                                                   color=[plt.cm.get_cmap('tab10')(i) for i in range(len(all_explanation_df.columns))])
        hatches = ''.join(h * len(all_explanation_df) for h in 'x/O.')
        for bar, hatch in zip(single_sample_ax.patches, hatches):
            bar.set_hatch(hatch)
        plt.legend()
        details = "\npredicted:" + str(activations[sample_idx].argmax()) + " true:" + str(te_set.tensors[1][sample_idx].numpy())
        plt.title(gold + "\nShap values of the sample " + str(sample_idx) + "\n" + exp + details)
        plt.yticks(np.arange(len(feat_names)), feat_names)
        plt.grid(True)
        if show:
            plt.show()
        else:
            plt.savefig(dt_name + "_" + exp + "_single_sample_explanations_" + gold + "_" + str(sample_idx).zfill(4) + ".pdf")
        plt.close()
    return indexes


def plot_table(faitfulness_dict_with_bv, show=True, class_idxs=None, threshold=0.7, exp=None):
    """Tables of the share of explanations with Faithfulness > 0 and > ``threshold``
    for each base value, and per class for the best base value.

    ``faitfulness_dict_with_bv = {base_value: {"Server": [array], "Clients": [...]}}``.
    Returns ``(base_values, shares)``.
    """
    dict_of_dicts = faitfulness_dict_with_bv
    num_clients = len(dict_of_dicts[list(dict_of_dicts.keys())[0]]["Clients"])
    values = sorted(list(dict_of_dicts.keys()))
    m_faits = -2000 * np.ones((len(values), 2))
    fait_by_class_of_values = {}

    for i, v in enumerate(values):
        fait_by_class = {}
        n_server = len(dict_of_dicts[v]["Server"][0])
        gt0 = (np.array(dict_of_dicts[v]["Server"]) > 0).sum() / n_server
        print("gt0", gt0)
        gt7 = (np.abs(np.array(dict_of_dicts[v]["Server"])) > threshold).sum() / n_server
        m_faits[i, 0] = gt0
        m_faits[i, 1] = gt7

        if class_idxs is not None:
            # number of explanations above threshold, per class
            fait_by_class["Server"] = np.zeros(len(class_idxs))
            fait_by_class["Clients"] = [np.zeros(len(class_idxs)) for _ in range(num_clients)]
            for class_id, idxs in enumerate(class_idxs):
                fait_by_class["Server"][class_id] = (dict_of_dicts[v]["Server"][0][idxs] > threshold).sum()
                for j in range(len(dict_of_dicts[v]["Clients"])):
                    fait_by_class["Clients"][j][class_id] = (dict_of_dicts[v]["Clients"][j][idxs] > threshold).sum()
        fait_by_class_of_values[v] = fait_by_class

    # per-class table for the base value with the highest share above threshold
    m_faits_by_class = np.zeros((1 + num_clients, len(class_idxs)))
    best_base_val = values[np.argsort(m_faits[:, 1])[::-1][0]]
    m_faits_by_class[0, :] = fait_by_class_of_values[best_base_val]["Server"][:]
    for j in range(num_clients):
        m_faits_by_class[j + 1, :] = fait_by_class_of_values[best_base_val]["Clients"][j][:]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    table = ax.table(cellText=np.round(m_faits * 100, 2), rowLabels=values,
                     colLabels=["gt 0 (%)", "gt " + str(threshold) + " (%)"], loc='center')
    table.set_fontsize(14)
    table.scale(1, 3)
    ax.set_title("Percentage of faithfulness values above 0 and threshold", y=1.2)
    ax.axis('off')

    ax = axes[1]
    print(["class " + str(i + 1) + "\nsupp:" + str(k) for i, k in enumerate(class_idxs)])
    table = ax.table(cellText=np.round(m_faits_by_class, 2).astype(int),
                     rowLabels=["Server"] + ["C" + str(i + 1) for i in range(num_clients)],
                     colLabels=["class " + str(i + 1) + "\nsupp:" + str(len(k)) for i, k in enumerate(class_idxs)],
                     loc='center')
    table.set_fontsize(18)
    table.scale(1, 3)
    ax.set_title("Number of samples with \nfaithfulness >" + str(threshold) + " by class\n using " + str(best_base_val) + " as base value", y=1.2)
    ax.axis('off')
    plt.tight_layout()
    plt.suptitle(exp if exp is not None else "Faithfulness", y=1.02, fontsize=18)
    if show:
        plt.show()
        plt.close()
    return values, m_faits


def plot_results(results, all_shap_values_dict, exp, el, faithfulnesses, f1_scores, most_excluded=None, scoress=None):
    """Debug view of the aggregation of a single sample ``el`` (used by
    :func:`iflash.aggregation.policy_selection` with ``show=True``)."""
    clients = all_shap_values_dict[exp]["Clients"]
    fix, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(len(clients)):
        axes[0].plot(clients[i][el], alpha=normalize_weightOLD(f1_scores[i]), color="grey")
    for k, l in results.items():
        axes[0].plot(l[-1], label=k, color=colors[k])
    axes[0].plot(all_shap_values_dict[exp]["Server"][el], label="Server", color="black", lw=2, ls='--')

    metr = ["cos", "sp", "kl"]
    axes[1].set_xticks(np.arange(len(metr)), metr)
    for j, k in enumerate(results.keys()):
        singl_scores = scoress[k]
        for i, m in enumerate(metr):
            axes[1].scatter(i + 0.01 * j, singl_scores[m], label=res_d[k] if i == 0 else None, color=colors[k])
        # the L2 has a different scale
        axes[2].scatter(0.01 * j, singl_scores["l2"], label=res_d[k], color=colors[k])
    axes[1].set_title("Scores")
    axes[2].set_title("L2 (SCORE)")
    plt.legend()
    plt.suptitle(f"Sample {el} - {exp}")
    plt.show()

    faith_el = faithfulnesses[el, :]
    q1, q2, q3 = np.percentile(faith_el, [25, 50, 75])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(np.arange(len(f1_scores)), sorted(faith_el))
    axes[1].scatter(np.arange(len(f1_scores)), sorted(normalize_weights(faith_el)))
    axes[2].scatter(np.arange(len(f1_scores)), sorted(f1_scores))
    for q in (q1, q2, q3):
        axes[0].axhline(q, color='red', lw=1.5, ls='--')
    mini, maxi = np.min(faith_el), np.max(faith_el)
    axes[0].set_yticks([mini, q1, q2, q3, maxi])
    axes[0].set_yticklabels([np.round(mini, 2)] + ["q" + str(i + 1) for i in range(3)] + [np.round(maxi, 2)])
    axes[0].set_title("Faithfulnesses")
    axes[1].set_title("Weights")
    axes[2].set_title("F1 Scores")
    for ax in axes:
        ax.grid(True)
    plt.suptitle(f"Sample {el} - {exp}")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(most_excluded[0])), most_excluded[0])
    plt.title("most excluded clients so far")
    plt.show()


def plot_faithfulness_vs_gap(faithfulnesses, class_idxs, exp, dt_name, show=False):
    """For each sample, the clients' Faithfulness against their gap from the most
    faithful client, one panel per class."""
    max_f = np.max(faithfulnesses, axis=1)
    faithfulnesses_delta = max_f[:, None] - faithfulnesses
    gap = np.argsort(faithfulnesses_delta, axis=1)
    max_gap = np.max(faithfulnesses_delta)

    faithfulnesses[np.isnan(faithfulnesses)] = 0.0
    print("nans are", np.isnan(faithfulnesses).sum())
    num_c = len(class_idxs)
    if exp == "004_iid" or np.isnan(max_gap) or np.isinf(max_gap):
        max_gap = 0.5

    cmap = plt.cm.get_cmap('tab10')
    fig, axes = plt.subplots(figsize=(5 * num_c, 10), nrows=1, ncols=num_c)
    if num_c == 1:
        axes = [axes]

    for class_id, idxs in enumerate(class_idxs):
        color_class = cmap(class_id)
        samples_below_06 = samples_below_0 = 0
        for el in idxs:
            xvalues = faithfulnesses_delta[el, gap[el, :]]
            yvalues = faithfulnesses[el, gap[el, :]]
            axes[class_id].plot(xvalues, yvalues, color=color_class, alpha=0.3)
            axes[class_id].scatter(xvalues, yvalues, color=color_class, alpha=0.2)
            samples_below_06 += int(np.any(yvalues < -0.6))
            samples_below_0 += int(np.any(yvalues < 0))

        ax = axes[class_id]
        ax.set_ylim(-1, 1)
        ax.set_xlim(-0.05, max_gap + (max_gap * 1 / 10))
        ax.set_title(f"Class {class_id}\nFaithfulness vs Gap\n"
                     f"Samples < -0.6: {samples_below_06}, < 0: {samples_below_0}", fontsize=20)
        ax.set_ylabel("Faithfulness")
        ax.set_xlabel("Gap")
        ax.grid(True)
        ax.axhline(-0.6, color='black', lw=1.5, ls='--')
        ax.axhline(0, color='black', lw=1.5, ls='--')

    label = exp_label(exp)
    plt.suptitle(label, fontsize=24)
    if show:
        print("Showing graphic")
        plt.show()
    else:
        plt.savefig(f"{dt_name}_{label}_faithfulness_vs_gap.pdf")
        plt.savefig(f"{dt_name}_{label}_faithfulness_vs_gap.png")
        print(f"Saved in {label}_faithfulness_vs_gap.pdf")
    plt.close()


def plot_recalulated_faiths(exp, faiths, base_value=0.2, proposals=["E_hat"], dt_name="dt_name", show=False):
    """Faithfulness of the aggregated explanations compared with the Audit:
    cumulative distributions, share of samples above -0.5/0/0.5, and sorted values."""
    b_v = base_value
    props = proposals
    faits = faiths[exp][b_v]
    print(len(faits.keys()), "recalculated faiths (2 are for the Server, and the Clients)")
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    shape_of_ = (faits["Server"].T.shape[0], len(props) + 1)
    print(shape_of_)
    image = np.zeros(shape=shape_of_)
    for j, p in enumerate(["Server"] + props):
        print(p, end='\t')
        if p == "Clients":
            print("not plotted")
            continue
        if p == 'Server':
            ls, lw, col = '-', 6, "black"
        else:
            ls, lw, col = "--", 4, colors[p]
        faits[p] = np.array(faits[p])
        print("plotted;", faits[p].shape, end="\t\t")
        print(col)
        lines_ax = sns.kdeplot(faits[p].T, linestyle=ls, linewidth=lw, ax=ax[0], color=col, cumulative=True, label=res_d[p])
        lines_ax.lines[-1].set_color(col)
        image[:, j] = faits[p].T.reshape(-1)
    image = np.array([np.sort(np.round(fa, 1)).T for fa in image[:, :].T])
    ax[2].imshow(image, aspect='auto', cmap='coolwarm', interpolation='none', rasterized=True)

    threshes = sorted([-.5, 0, .5])
    hatchings = ['//', '+', 'x']
    xti = []
    for good_thresh, htc in zip(threshes, hatchings):
        gt0_server = np.sum(faits["Server"] > good_thresh) / faits["Server"].shape[1]
        ax[1].barh(len(props) + 1, gt0_server, label="faith > " + str(np.round(good_thresh, 2)), color='white', hatch=htc)
        tixks = ["Server"]
        maxes = []
        for jj, p in enumerate(props):
            gt0 = np.sum(faits[p] > good_thresh) / faits[p].shape[1] - gt0_server
            ax[1].barh([len(props) - jj], gt0, color=colors[p], left=gt0_server, hatch=htc)
            tixks += [res_d[p]]
            maxes += [gt0]
        ax[1].vlines(gt0_server, -1, len(props) + 1, lw=3, ls='--')
        xti.append(gt0_server)

    ax[1].set_xticks(xti, [str(int(x_ti * 100)) + "%" for x_ti in xti], fontsize=14, rotation=90)
    ax[1].set_yticks(1 + np.arange(len(props) + 1)[::-1], tixks, rotation=0, fontsize=20)
    ax[1].grid(axis='y')
    ax[1].set_ylim(0.5, len(props) + 1.5)
    ax[1].set_xlim(0, 1.05)
    ax[2].set_yticks(np.arange(len(props) + 1), ["Server"] + [res_d[ppr] for ppr in props], rotation=0, fontsize=20)
    ax[2].set_ylabel("Methods")
    ax[2].set_xlabel("Samples")
    ax[2].set_title("sorted Faithfulness of each method\n" + dt_name + " " + exp_label(exp), fontsize=20)
    ax[0].set_title("Cumulative distribution of the faithfulness\n" + dt_name + " " + exp_label(exp), fontsize=20)
    ax[1].set_title("comparative percentage faitfulness", fontsize=20)
    ax[1].legend(fontsize=18)
    ax[1].set_xlabel("Percentage of samples", fontsize=18)
    for good_thresh in threshes:
        ax[0].axvline(good_thresh, color='k', linestyle='--')
    ax[0].grid(True)
    ax[0].legend(loc='upper left', fontsize=14)
    ax[0].set_xlim(-1.25, 1.25)

    plt.tight_layout()
    if isinstance(exp, (list, tuple)):
        exp = exp_label(exp)
    if isinstance(exp, str):
        exp = exp.split("_")
    plt.suptitle(dt_name + " " + exp_label(exp) + " faithfulness analysis", fontsize=20)
    file_n = dt_name + "_" + exp_label(exp, "_") + "_faithfulness_analysis.pdf"
    if not show:
        plt.savefig(file_n)
        print("created files", file_n)
        plt.close()
    else:
        plt.show()


# ------------------------------------------------------------- ShapGap
def plot_density_comparisons(exp, all_shap_values_dict, proposals, difference_function=compare_heatmaps_Frobenius,
                             name_comparison="Frobenius", ax=None, show=False, xlabel="differences",
                             dt_name="dt_name", ax_cumulative=None, cumulative=False, vmin=0, vmax=0.4,
                             new_names=res_d):
    """PDF (and optionally CDF) of the row-wise difference between the Audit and
    each proposal. Returns ``{proposal: differences}``."""
    shap_values = all_shap_values_dict[exp]
    diffs = {}
    for p in proposals:
        color = colors[p]
        diff = difference_function(shap_values["Server"], shap_values[p])
        labl = p if p not in new_names else new_names[p]
        sns.kdeplot(diff, ax=ax, label=labl, lw=3, color=color, cumulative=False)
        if cumulative:
            sns.kdeplot(diff, ax=ax_cumulative, label=labl, lw=3, color=color, cumulative=True)
        diffs[p] = diff

    ax.set_title("pdf\n" + name_comparison, fontsize=20)
    ax.legend(fontsize=13)
    ax.set_xlabel(xlabel)
    if cumulative:
        ax_cumulative.legend(fontsize=13)
        ax_cumulative.set_ylabel("cumulative density")
        ax_cumulative.set_title("cdf\n" + name_comparison, fontsize=20)
        ax_cumulative.set_xlabel(xlabel)
        ax_cumulative.grid(True)
        ax_cumulative.set_aspect('auto')
        ax_cumulative.set_ylim(-0.05, 1.05)

    if name_comparison == "Kendalltau":
        ax.set_xlim(-1, 1.25)
        ax_cumulative.set_xlim(-1, 1.25)
    elif name_comparison == "CosSim":
        ax.set_xlim(-1.05, 1.05)
        ax_cumulative.set_xlim(-1.05, 1.05)
    if name_comparison == "Frobenius":
        ax.set_xlim(vmin, vmax)
        ax_cumulative.set_xlim(vmin, vmax)
    ax.grid(True)
    ax.set_aspect('auto')
    plt.suptitle(exp_label(exp) + " " + dt_name, fontsize=20)
    return diffs


def plot_desity_comparisons_functions(exp, all_shap_values_dict, activations_of_models, colors, show=True,
                                      vmax=1.25, dt_name='dt name'):
    """Grid of :func:`plot_density_comparisons` for cosine similarity, Spearman
    and L2, for every method in ``colors``."""
    comparison_measures = [("Frobenius", "difference"),
                           ("Spearman", "correlation"),
                           ("CosSim", "similarity")][::-1]
    comparing_functions = [compare_heatmaps_Frobenius,
                           compare_heatmaps_spearmanr,
                           compare_heatmaps_cosinesim][::-1]
    fig, axes = plt.subplots(2, len(comparison_measures), figsize=(len(comparison_measures) * 5, 10))
    axes = axes.flatten()
    props = list(colors.keys())
    for i, ((comparison_measure, metrics_unit), comp_f) in enumerate(zip(comparison_measures, comparing_functions)):
        print(comparison_measure, comp_f)
        plot_density_comparisons(exp, all_shap_values_dict, props, difference_function=comp_f,
                                 name_comparison=comparison_measure, ax=axes[i], show=False,
                                 ax_cumulative=axes[i + len(comparison_measures)], cumulative=True,
                                 xlabel=metrics_unit, dt_name=dt_name, vmax=vmax)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(dt_name + "_" + exp_label(exp, "_") + "_density_comparisons.pdf")
    plt.close()
    return fig


# --------------------------------------------------- global explanations
def _order_features_by_magnitude(expl, feat_names):
    """Feature indices and names sorted by decreasing absolute importance."""
    ordered_indices = np.argsort(np.abs(expl))[::-1]
    return ordered_indices, [feat_names[i] for i in ordered_indices]


def plot_global_and_differencies(exp, all_shap_values_dict, activations_of_models, proposals, indices, show=False,
                                 feat_names=None, dt_name="dt_name", selection=None, vertical_single_size=5,
                                 limit_features=10):
    """Global explanation by class of each selected proposal next to the Audit's
    (the paper's "global explanation by class" figures), with the ShapGap per class.

    Returns the per-class and global ShapGap values of every proposal:
    ``(kendalls, frobs, shap_CS, spearmans, differencies,
    single_kendall, single_frobs, single_CS, single_spearmans)``.
    """
    g_expl_s, g_expl_by_class_proposals, global_s, global_props = get_global_explanation(
        exp, all_shap_values_dict, indices, activations_of_models, proposals)
    name_dataset = dataset_display_name(dt_name)

    differencies, kendalls, frobs, shap_CS, spearmanss = [], [], [], [], []
    single_kendall, single_frobs, single_CS, single_spearmans = [], [], [], []
    for p in proposals:
        by_class = g_expl_by_class_proposals[p]
        differencies.append(g_expl_s - by_class)
        kendalls.append(compare_heatmaps_kendalltau(g_expl_s, by_class))
        frobs.append(compare_heatmaps_Frobenius(g_expl_s, by_class))
        shap_CS.append(compare_heatmaps_cosinesim(g_expl_s, by_class))
        spearmanss.append(compare_heatmaps_spearmanr(g_expl_s, by_class))
        single_kendall.append(single_kendalltau_f(global_s, global_props[p]))
        single_frobs.append(single_Frobs_f(global_s, global_props[p]))
        single_CS.append(single_cosinesim_f(global_s, global_props[p]))
        single_spearmans.append(single_spearmanr_f(global_s, global_props[p]))

    print("kendalltau's have", len(kendalls), "elements, and shape ", kendalls[0].shape)
    print("diff has ", len(frobs), "elements, and shape ", frobs[0].shape)
    print("shap_CS has ", len(shap_CS), "elements, and shape ", shap_CS[0].shape)
    print("spearmanss has ", len(spearmanss), "elements, and shape ", spearmanss[0].shape)

    new_proposals = selection if selection is not None else proposals

    # common y range per class, across the server and the selected proposals
    fi_vmax, fi_vmin = [], []
    for i in range(g_expl_s.shape[0]):
        fi_vmax.append(np.max([np.max(g_expl_s[i])] + [np.max(g_expl_by_class_proposals[p][i]) for p in new_proposals]))
        fi_vmin.append(np.min([np.min(g_expl_s[i])] + [np.min(g_expl_by_class_proposals[p][i]) for p in new_proposals]))
        print("the max and min of the global explanation are", fi_vmax[-1], fi_vmin[-1])

    server_color = "#005AB5"
    choosen_color = "#DC3220"
    for jc, p in enumerate(proposals):
        if p not in new_proposals:
            continue
        # layout: two columns of classes; one row if binary
        if len(indices) > 2:
            fig = plt.figure(figsize=(10, 5 * (len(indices) // 2 + 1)))
            gs = GridSpec(len(indices) // 2 + len(indices) % 2, 2, figure=fig)
        elif len(feat_names) < 15:
            fig = plt.figure(figsize=(10, 6))
            gs = GridSpec(1, 2, figure=fig)
        else:
            fig = plt.figure(figsize=(10, 10))
            gs = GridSpec(2, 1, figure=fig)
        print(len(indices) // 2 + len(indices) % 2, 2)
        axes = []
        if len(indices) > 2:
            for row in range(len(indices) // 2):
                for col in range(2):
                    axes.append(fig.add_subplot(gs[row, col]))
            if len(indices) % 2 != 0:
                axes.append(fig.add_subplot(gs[-1, 0]))
        elif len(feat_names) < 15:
            axes = [fig.add_subplot(gs[0, col]) for col in range(2)]
        else:
            axes = [fig.add_subplot(gs[row, 0]) for row in range(2)]
        labl = res_d[p]
        axes = np.array(axes).flatten()
        print(len(axes))

        for i, expl in enumerate(g_expl_by_class_proposals[p]):
            ax = axes[i]
            ordered_idx, ordered_feat = _order_features_by_magnitude(expl, feat_names)
            if limit_features is not None:
                ordered_idx = ordered_idx[:limit_features]
                ordered_feat = ordered_feat[:limit_features]
            x = np.arange(len(ordered_feat))
            width = 0.4
            ax.bar(x - width / 2, expl[ordered_idx], width, label=res_d[p], color=choosen_color)
            ax.bar(x + width / 2, g_expl_s[i][ordered_idx], width, label="Audit**", color=server_color)
            increment = np.max(np.abs([fi_vmin[i], fi_vmax[i]])) / 100 * 5  # 5% margin
            ax.set_ylim(fi_vmin[i] - increment, fi_vmax[i] + increment)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            if i == 0:
                ax.legend(fontsize=16)
            ax.set_title("Class " + str(i) + "\n$ShapGap_{CosSim}$ " + str(np.round(shap_CS[jc][i], 3)) +
                         ",\n$ShapGap_{L2}$ " + str(np.round(np.mean(frobs[jc][i]), 3)), fontsize=13)
            ax.set_xticks(x, ordered_feat, rotation=90, fontsize=10 if limit_features > 15 else 18)

        if "%" not in exp:
            setting = exp_label(exp.replace("imb", "$\\alpha=$").split("_") if isinstance(exp, str) else exp)
        else:
            setting = fancytitle(exp.split("%"))
        plt.suptitle(name_dataset + " setting " + setting + " - " + labl, fontsize=18)
        plt.tight_layout()
        if len(indices) % 2 != 0:
            # centre the last subplot when the number of classes is odd
            pos1, pos2 = axes[0].get_position(), axes[1].get_position()
            new_x0 = (pos1.x0 + pos2.x0) / 2
            new_x1 = (pos1.x1 + pos2.x1) / 2
            last = axes[-1].get_position()
            axes[-1].set_position([new_x0, last.y0, new_x1 - new_x0, last.y1 - last.y0])

        if show:
            plt.show()
            plt.close()
        else:
            name_fig = dt_name + "_" + exp_filename(exp) + "_global_explanation_by_class_" + p + ".pdf"
            plt.savefig(name_fig, bbox_inches='tight')
            print("saved the figure", name_fig)
            name_fig = name_fig[:-3] + "png"
            plt.savefig(name_fig)
            print("saved the figure", name_fig)
            plt.close()
        plt.clf()

    return kendalls, frobs, shap_CS, spearmanss, differencies, single_kendall, single_frobs, single_CS, single_spearmans
