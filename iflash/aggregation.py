"""Aggregation strategies of the clients' explanations (Section "Aggregation" of the paper).

Every strategy works sample by sample: given the SHAP explanations of one test
sample produced by the ``C`` clients (``client_explanations``, shape
``(C, n_features)``) and their Faithfulness (``faithfulnesses``, shape
``(C,)``), it returns one aggregated explanation of shape ``(n_features,)``.
:func:`policy_selection` applies all of them to a whole experiment.
"""
import warnings

import numpy as np


# ------------------------------------------------------------------ helpers
def weighted_average(data, weights):
    """Weighted average of ``data`` along the clients axis."""
    return np.average(data, axis=0, weights=weights)


def normalize_weights(weights):
    """Map Faithfulness from [-1, 1] to [0, 1]: ``(w + 1) / 2``."""
    weights = weights + 1
    return weights / 2


def normalize_weightOLD(weights):
    """Min-max style rescaling to (0, 1], only used for plot transparency."""
    weights = weights - np.min(weights)
    weights += 0.1
    return weights / np.max(weights)


# --------------------------------------------------------------- strategies
def fait_top_quartile(client_explanations, faithfulnesses, most_exluded, quartile="top"):
    """E_Q1: average the explanations of the most faithful clients.

    ``quartile="top"`` keeps clients above the 75th percentile of Faithfulness
    (the 25% most faithful), ``"mid-top"`` those above the 25th percentile, and
    ``"center"`` those between the 25th and 75th percentile.
    ``most_exluded[0]`` accumulates, per client, how often it was discarded.
    """
    q1, q3 = np.percentile(faithfulnesses, [25, 75])
    if quartile == "top":
        selected_idx = np.where(faithfulnesses > q3)[0]
    elif quartile == "mid-top":
        selected_idx = np.where(faithfulnesses > q1)[0]
    elif quartile == "center":
        selected_idx = np.where((faithfulnesses > q1) & (faithfulnesses < q3))[0]
    else:
        raise ValueError("Quartile must be top, mid-top or center")
    not_selected = ~np.isin(np.arange(len(faithfulnesses)), selected_idx)
    most_exluded[0] = most_exluded[0] + not_selected.astype(int)
    return np.mean(client_explanations[selected_idx], axis=0)


def faith_selection(client_explanations, faithfulnesses, threshold=0.0, num_filtered=None, most_excluded=None):
    """Faithfulness-weighted average of the explanations with Faithfulness > threshold
    (plain average if none passes)."""
    selected_idx = np.where(faithfulnesses > threshold)[0]
    most_excluded[0] = most_excluded[0] + (faithfulnesses <= threshold)
    if len(selected_idx) == 0:
        return np.mean(client_explanations, axis=0)
    num_filtered[0] += 1
    w = normalize_weights(faithfulnesses)
    return weighted_average(client_explanations[selected_idx], w[selected_idx])


def E_avg_t_0(client_explanations, faithfulnesses, threshold=0.0, num_filtered=None, most_excluded=None):
    """E^0_Avg: plain average of the explanations with Faithfulness > threshold
    (plain average of all of them if none passes)."""
    selected_idx = np.where(faithfulnesses > threshold)[0]
    if len(selected_idx) == 0:
        return np.mean(client_explanations, axis=0)
    num_filtered[0] += 1
    most_excluded[0] = most_excluded[0] + (faithfulnesses <= threshold)
    return np.mean(client_explanations[selected_idx], axis=0)


def mixed_selection(client_explanations, faithfulnesses, f1_scores, threshold=0.0):
    """E_Mix: Faithfulness-weighted average of the explanations with Faithfulness >
    threshold; if none passes, average of all explanations weighted by F1."""
    selected_idx = np.where(faithfulnesses > threshold)[0]
    if len(selected_idx) == 0:
        return weighted_average(client_explanations, f1_scores)
    w = normalize_weights(faithfulnesses)
    return weighted_average(client_explanations[selected_idx], w[selected_idx])


def f1_faith_selection(client_explanations, faithfulnesses, f1_scores, threshold=0.0):
    """E_MixF1: F1-weighted average of the explanations with Faithfulness >
    threshold; if none passes, F1-weighted average of all explanations."""
    selected_idx = np.where(faithfulnesses > threshold)[0]
    if len(selected_idx) == 0:
        return weighted_average(client_explanations, f1_scores)
    return weighted_average(client_explanations[selected_idx], f1_scores[selected_idx])


# ------------------------------------------------------------- experiment
def print_differences(counting_differences, total):
    """Report how often each strategy coincides with the plain average."""
    for label, count in counting_differences.items():
        percentage = np.round(count / total * 100, 2)
        print(f"The explanations {label} equal to the average are {count} over {total} ({percentage}%)")


def policy_selection(exp, faithfulnesses, all_shap_values_dict, f1_scores, class_idxs=None, dt_name='dt_name', show=False):
    """Apply every Faithfulness-based strategy to all the samples of experiment ``exp``.

    ``faithfulnesses["Clients"]`` has shape ``(C, n_samples)``. The aggregated
    explanations are stored in ``all_shap_values_dict[exp]`` under the keys
    ``E_fait_weighted``, ``E_avg_t_0``, ``E_mixed``, ``E_mix_f1`` and ``E_Q1``
    (see :mod:`iflash.naming` for the paper names), and the updated dictionary
    is returned. With ``show=True`` a random 0.1% of the samples is plotted.
    """
    faithfulnesses = faithfulnesses["Clients"].T
    results = {
        "E_fait_weighted": [],  # average weighted by faithfulness
        "E_avg_t_0": [],        # average of the clients with faithfulness > 0
        "E_mixed": [],          # faithfulness > 0 weighted by faithfulness, else by f1
        "E_mix_f1": [],         # faithfulness > 0 weighted by f1
        "E_Q1": [],             # average of the top-quartile most faithful clients
    }
    counting_differences = {k: 0 for k in results}
    # how many times each client is filtered out, and how many times a filter applies
    most_excluded = [np.zeros(faithfulnesses.shape[1])]
    num_filtered = [0]
    clients = all_shap_values_dict[exp]["Clients"]

    for el in range(faithfulnesses.shape[0]):
        client_explanations = np.array([clients[i][el] for i in range(len(clients))])
        faith_el = faithfulnesses[el, :]
        if np.isnan(faith_el).sum() > 0:
            warnings.warn("Nans in the faithfulness values" + "\nclients: " + str(np.arange(len(faith_el))[np.isnan(faith_el)]))
            faith_el[np.isnan(faith_el)] = -1.0

        w = normalize_weights(faith_el)
        results["E_fait_weighted"].append(weighted_average(client_explanations, w))
        results["E_avg_t_0"].append(E_avg_t_0(client_explanations, faith_el, threshold=0.0, num_filtered=num_filtered, most_excluded=most_excluded))
        results["E_mixed"].append(mixed_selection(client_explanations, faith_el, f1_scores, threshold=0.0))
        results["E_mix_f1"].append(f1_faith_selection(client_explanations, faith_el, f1_scores, threshold=0.0))
        results["E_Q1"].append(fait_top_quartile(client_explanations, faith_el, most_excluded, quartile="top"))

        avg_expl = np.average(client_explanations, axis=0)
        for k, e in results.items():
            assert len(e[-1]) == len(avg_expl)
            if np.allclose(e[-1], avg_expl):
                counting_differences[k] += 1

        if show and np.random.rand() < 0.001:
            from .metrics import single_cosinesim_f, single_Frobs_f, single_kendalltau_f, single_spearmanr_f
            from .plotting import plot_results
            server_el = all_shap_values_dict[exp]["Server"][el]
            all_scores = {k: {"cos": single_cosinesim_f(server_el, results[k][-1]),
                              "l2": single_Frobs_f(server_el, results[k][-1]),
                              "sp": single_spearmanr_f(server_el, results[k][-1]),
                              "kl": single_kendalltau_f(server_el, results[k][-1])}
                          for k in results}
            plot_results(results, all_shap_values_dict, exp, el, faithfulnesses, f1_scores, most_excluded=most_excluded, scoress=all_scores)

    print_differences(counting_differences, len(results["E_fait_weighted"]))
    print("filter applied", num_filtered, "times")
    most_excluded_idx = np.argsort(most_excluded)[::-1]
    print("most excluded clients indexes:", most_excluded_idx)
    print(most_excluded)

    for key, value in results.items():
        all_shap_values_dict[exp][key] = np.array(value)
    return all_shap_values_dict


# ------------------------------------------------------- global explanations
def get_global_explanation(exp, all_shap_values_dict, original_indices, activations_of_models, proposals):
    """Global explanation by class: the mean SHAP values of the samples predicted
    as each class, for the Server (Audit) and for every proposal.

    Returns ``(server_by_class, proposals_by_class, server_global, proposals_global)``.
    """
    print(all_shap_values_dict[exp].keys())

    def get_g_expl(key, selected_idx):
        return np.mean([all_shap_values_dict[exp][key][i] for i in selected_idx], axis=0)

    random_k = list(activations_of_models[exp].keys())[0]
    activas = activations_of_models[exp][random_k]
    predicted_class = activas.argmax(axis=1)

    g_expl_by_class = []
    g_expl_by_class_proposals = {p: [] for p in proposals}
    for cl_id in range(len(original_indices)):
        class_idxs = np.where(predicted_class == cl_id)[0]
        g_expl_by_class.append(get_g_expl("Server", class_idxs))
        for p in proposals:
            g_expl_by_class_proposals[p].append(get_g_expl(p, class_idxs))

    all_idxs = np.arange(activas.shape[0])
    global_S = get_g_expl("Server", all_idxs)
    prop_global_s = {p: get_g_expl(p, all_idxs) for p in proposals}
    return (np.array(g_expl_by_class),
            {k: np.array(v) for k, v in g_expl_by_class_proposals.items()},
            global_S, prop_global_s)
