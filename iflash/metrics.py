"""Comparison measures between feature-importance (SHAP) explanations.

``single_*`` functions compare two explanations of the same sample
(1-D arrays of shape ``(n_features,)``); ``compare_heatmaps_*`` functions
compare two explanation matrices of shape ``(n_samples, n_features)`` row by
row and return one value per sample. These are the ShapGap measures of the
paper (L2, cosine similarity, Spearman and Kendall correlation).
"""
import itertools

import numpy as np
from scipy.stats import kendalltau, spearmanr


# ---------------------------------------------------------------- per sample
def single_kendalltau_f(fi1, fi2):
    """Kendall tau correlation between two feature importances."""
    return kendalltau(fi1, fi2).correlation


def single_spearmanr_f(fi1, fi2):
    """Spearman correlation between two feature importances."""
    return spearmanr(fi1, fi2).statistic


def single_mean_abs_f(fi1, fi2):
    """Mean absolute difference between two feature importances."""
    return np.mean(np.abs(fi1 - fi2))


def single_cosinesim_f(fi1, fi2):
    """Cosine similarity between two feature importances (ShapGap_Cos)."""
    return np.sum(fi1 * fi2) / (np.sqrt(np.sum(fi1 ** 2)) * np.sqrt(np.sum(fi2 ** 2)))


def single_Frobs_f(fi1, fi2):
    """L2 norm of the difference between two feature importances (ShapGap_L2)."""
    return np.sqrt(np.sum((fi1 - fi2) ** 2))


# --------------------------------------------------------------- per matrix
def compare_heatmaps(h1, h2):
    """Mean absolute difference between two explanation matrices (a scalar)."""
    diff = h1 - h2
    return np.mean(np.mean(np.abs(diff), axis=1))


def compare_heatmaps_kendalltau(h1, h2):
    """Row-wise Kendall tau between two explanation matrices."""
    return np.array([single_kendalltau_f(h1[i], h2[i]) for i in range(len(h1))])


def compare_heatmaps_spearmanr(h1, h2):
    """Row-wise Spearman correlation between two explanation matrices."""
    return np.array([single_spearmanr_f(h1[i], h2[i]) for i in range(h1.shape[0])])


def compare_heatmaps_cosinesim(matrix1, matrix2):
    """Row-wise cosine similarity between two explanation matrices."""
    return np.array([single_cosinesim_f(matrix1[i], matrix2[i]) for i in range(matrix1.shape[0])])


def compare_heatmaps_Frobenius(h1, h2):
    """Row-wise L2 distance between two explanation matrices."""
    return np.array([single_Frobs_f(h1[i], h2[i]) for i in range(h1.shape[0])])


def compute_diffs(heatmaps_list, comparison=compare_heatmaps):
    """Upper-triangular matrix of pairwise ``comparison`` between explanations."""
    nc = len(heatmaps_list)
    differences_matrix = np.zeros((nc, nc))
    for a, b in itertools.combinations(range(nc), 2):
        differences_matrix[a, b] = comparison(heatmaps_list[a], heatmaps_list[b])
    return differences_matrix
