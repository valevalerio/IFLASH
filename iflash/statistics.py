"""Statistical comparison of the aggregation strategies.

Friedman test followed by the Nemenyi post-hoc test, drawn as a
critical-difference (CD) diagram together with a boxen plot of the values.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

from .naming import colors, exp_filename, res_d


def remap_names(ranking_df, colors_dict):
    """Rename the method columns of ``ranking_df`` with their paper labels and
    return the matching ``{label: color}`` dictionary."""
    res_colors = {res_d[k]: v for k, v in colors_dict.items()}
    ranking_df.columns = [res_d[col] for col in ranking_df.columns]
    print(res_colors, "res_colors")
    print(ranking_df.columns, "columns of the ranking_df")
    return ranking_df, res_colors


def create_df_and_check(methods_names, values, title="", title_size=18, show=True, exp="", metrica='',
                        dt_name='dt_name', vmin=0, vmax=0.4, hmargin=0.1, vertica_image_size=5, fname=None):
    """Friedman + Nemenyi test over the methods and, if significant, a figure with
    a boxen plot of the values (top) and the CD diagram (bottom).

    ``values`` has one row per sample (block) and one column per method in
    ``methods_names`` (code keys, see :mod:`iflash.naming`). Values are ranked
    row-wise, so lower is better; similarity measures (everything except
    ``metrica == "$ShapGap_{L2}$"``) are expected to be passed negated and are
    flipped back for the boxen plot. With ``show=False`` the figure is saved in
    ``cddiagrams/``.
    """
    import scikit_posthocs as sp

    df = pd.DataFrame(values, columns=methods_names)
    ranks = df.rank(axis=1)
    assert ranks.values.shape[1] == len(methods_names), "the ranks have a different number of columns than the methods names"

    if len(values) >= 3:
        _, p = ss.friedmanchisquare(*values)
        cd = sp.posthoc_nemenyi_friedman(ranks)
    else:
        raise ValueError("not enough values to perform the friedman test")

    if p < 0.05:
        if len(values) >= 2:
            fig, ax = plt.subplots(2, 1, figsize=(10, vertica_image_size))
            axes = ax.flatten()
            ranks_res, colors_res = remap_names(ranking_df=ranks, colors_dict=colors)
            if (np.sum(cd < 0.05) > 0).any():
                # the adjacency matrix must have the same labels on rows and columns
                cd.columns = [res_d[col] for col in cd.columns]
                cd.index = cd.columns
                print("plotting CD diagram")
                axes[1].set_xlim(1, len(methods_names))
                sp.critical_difference_diagram(ranks_res.T.mean(axis=1), cd, label_props={"fontsize": title_size - 1},
                                               color_palette=colors_res, ax=axes[1], text_h_margin=hmargin)
            else:
                axes[1].text(0.5, 0.5, "no significant difference", fontsize=20)

            if metrica != "$ShapGap_{L2}$":
                values = -values
                df = -df
                ylab = "Faithfulness" if metrica == "Faithfulness" else "Similarity"
            else:
                ylab = "Distance"

            axes[0].grid(axis='y')
            sns.boxenplot(data=df, ax=axes[0], palette=[colors[k] for k in df.columns], fill=False)
            axes[0].set_xticks(np.arange(len(methods_names)), ranks_res.columns, rotation=30, fontsize=15)
            axes[0].set_ylabel(ylab, fontsize=title_size)
            axes[0].set_ylim(vmin, vmax)

        print(cd.round(2))
        # methods that are never significantly different from the others
        cd1 = cd.copy()
        np.fill_diagonal(cd1.values, 0)
        s = np.sum(cd1.values, axis=1)
        print(s)
        idx = np.where(s < 0.05 * len(methods_names))[0]
        print(s[idx])

        plt.suptitle(title, fontsize=title_size)
        plt.tight_layout()
        if fname is None:
            fname = dt_name + "_" + exp_filename(exp) + "_CD_" + metrica + ".pdf"
        if show:
            plt.show()
        else:
            plt.savefig("cddiagrams/" + fname)
            plt.savefig("cddiagrams/" + fname.replace(".pdf", ".png"))
            plt.close()
    else:
        print("the differences are not significant\n" + title)
    print()
