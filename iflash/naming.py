"""Names, labels and colors shared by every plot and table.

The keys used throughout the code (e.g. ``"E_Q1"``) are the keys under which
each aggregated explanation is stored in the experiment dictionaries. The
mapping to the symbols used in the paper is:

==================  ===============  ==============================================
code key            paper symbol     description
==================  ===============  ==============================================
``E_hat``           E_Avg            plain average of the clients' SHAP values
``E_weighted``      E_F1             average weighted by the clients' F1 score
``E_f1_macro``      E_F1macro        average weighted by the clients' macro F1
``E_fait_weighted`` E_faith          average weighted by (normalised) Faithfulness
``E_avg_t_0``       E^0_Avg          average of explanations with Faithfulness > 0
``E_mixed``         E_Mix            E_faith on Faithfulness > 0, else E_F1macro
``E_mix_f1``        E_MixF1          macro-F1 weighted average on Faithfulness > 0
``E_Q1``            E_Q1             average of the top-quartile most faithful
``COMP``            FedFCM           competitor (clustering based)
``SHAP_FL``         SHAP-FL          competitor (federated background sampling)
``Server``          Audit            centralised explainer (upper reference)
==================  ===============  ==============================================
"""

# Colorblind-friendly palette (Paul Tol's "vibrant"/"muted" schemes).
colors = {
    "E_hat": "#0077BB",            # canonical mean                      Strong Blue
    "E_weighted": "#EE7733",       # f1 (f1_weighted)                    Orange
    "E_fait_weighted": "#009988",  # faithfulness                        Teal
    "E_avg_t_0": "#CC3311",        # filtering on 0 faithfulness, avg    Red
    "E_mixed": "#33BBEE",          # f1 + faith avg + filtering on 0     Sky Blue
    "E_mix_f1": "#885500",         # filtering + average on f1           Brown
    "E_avg_t_06": "#DDAA33",       # average + filtering on -0.6 faith   Yellow
    "E_f1_macro": "#444444",       # f1 macro                            Dark Gray
    "E_france_W": "#004488",       # faithfulness weighted, t=-0.6       Navy Blue
    "E_Q1": "#BB5566",             # faithfulness top quartile           Red-Purple
    "Server": "#000000",           # Audit                               Black
    "COMP": "#FF5733",             # competitor: FedFCM (clustering)
    "SHAP_FL": "#197D2CFF",        # competitor: SHAP-FL                 Green
}

# LaTeX labels of each aggregation method ("d" stands for definitions).
res_d = {
    "E_hat": "$E_{avg}$",
    "E_weighted": "$E_{f1_w}$",
    "E_fait_weighted": "$E_{faith}$",
    "E_avg_t_06": "$E_{avg}^{-0.6}$",
    "E_avg_t_0": "$E_{avg}^{0}$",
    "E_france_W": "$E_{faith}^{-0.6}$",
    "E_mixed": "$E_{mix}^{0}$",
    "E_f1_macro": "$E_{f1_{M}}$",
    "E_mix_f1": "$E_{mix_{f1}}^0$",
    "E_Q1": "$E_{Q1}$",
    "E_harm": "$E_{harm}$",
    "COMP": "$FedFCM$",
    "SHAP_FL": "$SHAP-FL$",
    "Server": "Audit**",
}


def dataset_display_name(dt_name):
    """Human readable dataset name used in figure titles."""
    name = dt_name.lower()
    if "adult" in name:
        return "Adult"
    if "diamond" in name:
        return "Diamonds"
    if "bean" in name:
        return "Dry Bean"
    if "income" in name:
        return "ACSI"
    if "dutch" in name:
        return "Dutch"
    return "dataset name"


def exp_label(exp, sep=" "):
    """Join an experiment identifier (string, tuple or list) into a label."""
    return sep.join(str(e) for e in exp)


def exp_filename(exp):
    """Experiment identifier as a file-name fragment (tuples joined by ``_``)."""
    if isinstance(exp, (tuple, list)):
        return "_".join(str(e) for e in exp)
    return exp


def fancytitle(exp):
    """Title for cross-device experiments, ``exp`` = (distribution, "<N>_nodi")."""
    nodes = exp[1].replace("_", " ").replace("nodi", "nodes")
    if exp[0] == "099":
        return nodes + " $\\alpha$=0.99"
    if exp[0] == "5":
        return nodes + " $\\alpha$=5.0"
    if exp[0] == "iid":
        return nodes + " iid"
