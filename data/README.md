# Datasets

No data is shipped with this repository. All datasets are public:

| Dataset | Setting | Records | Task | Source |
|---|---|---|---|---|
| Adult | cross-silo (12, 16 clients) | 48,842 | income > 50K (binary) | [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) |
| Diamond | cross-silo (12, 16 clients) | 53,134 | cut quality (5 classes) | [Diamonds (ggplot2 / Kaggle)](https://www.kaggle.com/datasets/shivam2503/diamonds) |
| Dry Bean | cross-silo (12, 16 clients) | 13,611 | bean type (7 classes) | [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |
| Dutch | cross-device (50, 70, 150 clients) | 60,420 | income > 50K (binary) | Dutch census, as distributed in [Quy et al., *A survey on datasets for fairness-aware ML*](https://github.com/tailequy/fairness_dataset) |
| ACSI (ACS Income) | cross-device (51 clients, natural split) | 1,664,500 | income (binary) | [folktables](https://github.com/socialfoundations/folktables) |

## Expected layout (cross-silo)

The cross-silo scripts expect each dataset in `Experiments/<dataset_name>/`
(e.g. `Experiments/adult/adult.csv`, `Experiments/dry_bean/Dry_Bean_Dataset.xlsx`).
The loaders and preprocessing (target encoding of categorical features,
one-hot variants such as `diamond_1_hot`, `adult_TargEncoded`) are in
[`src/dataset_utils.py`](../src/dataset_utils.py) and `src/clean_*.py`.

Partitions, models, SHAP values and histories are then written by the
pipeline under `Experiments/<dataset_name>/<experiment>/`, where
`<experiment>` is e.g. `012_iid`, `016_imb_5.0`, `012_imb_0.99`
(number of clients, distribution, Dirichlet α).
