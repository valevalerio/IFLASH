"""iFLASH: interpreting Federated Learning by Aggregating SHAP explanations.

Server-side analysis library: Faithfulness of the clients' explanations,
aggregation strategies, ShapGap measures against the centralised Audit,
plots and statistical tests.

    import iflash as flx
    all_shap_values_dict = flx.policy_selection(exp, faiths, all_shap_values_dict, f1_scores)

Modules
-------
naming        method keys, paper labels (``res_d``) and colors (``colors``)
metrics       ShapGap measures (L2, cosine, Spearman, Kendall)
faithfulness  Faithfulness of local explanations
aggregation   aggregation strategies (E_avg, E_faith, E^0_avg, E_mix, E_mixF1, E_Q1)
plotting      figures
statistics    Friedman / Nemenyi tests and critical-difference diagrams

All public names are re-exported here, so ``from iflash import *`` gives the
same namespace as the original single-file ``flexai_lib``.
"""
from .naming import colors, res_d, dataset_display_name, exp_label, exp_filename, fancytitle
from .metrics import (single_kendalltau_f, single_spearmanr_f, single_mean_abs_f, single_cosinesim_f,
                      single_Frobs_f, compare_heatmaps, compare_heatmaps_kendalltau, compare_heatmaps_spearmanr,
                      compare_heatmaps_cosinesim, compare_heatmaps_Frobenius, compute_diffs)
from .faithfulness import aix_model, compute_faithfulness, recalculate_faiths, get_exps_faiths
from .aggregation import (weighted_average, normalize_weights, normalize_weightOLD, fait_top_quartile,
                          faith_selection, E_avg_t_0, mixed_selection, f1_faith_selection, policy_selection,
                          print_differences, get_global_explanation)
from .plotting import (plot_heatmap, plot_heatmap_and_focus, plot_differences,
                       plot_faithfulness_histograms_and_thresholds, plot_faith, plot_table, plot_results,
                       plot_faithfulness_vs_gap, plot_recalulated_faiths, plot_density_comparisons,
                       plot_desity_comparisons_functions, plot_global_and_differencies)
from .statistics import remap_names, create_df_and_check

# Names that the original ``from flexai_lib import *`` also brought into the
# notebooks' namespace.
import itertools, warnings
from decimal import Decimal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import torch
from scipy.stats import kendalltau, spearmanr
from matplotlib.gridspec import GridSpec
try:
    import scikit_posthocs as sp
    from aix360.metrics.local_metrics import faithfulness_metric
except ImportError:  # optional: only needed for CD diagrams / Faithfulness
    pass
