"""Faithfulness of local explanations (Alvarez-Melis & Jaakkola, 2018).

For every test sample, each client (and the server/Audit) produces a SHAP
explanation; Faithfulness measures how well the importance assigned to each
feature correlates with the drop in the predicted probability when that
feature is replaced by a base value. It is the quality score used by the
Faithfulness-based aggregation strategies in :mod:`iflash.aggregation`.
"""
import numpy as np
import torch


class aix_model():
    """Wrap a torch classifier with the ``predict``/``predict_proba`` API
    expected by ``aix360.metrics.faithfulness_metric``."""

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        x = torch.Tensor(x)
        return self.model(x).argmax(dim=1)

    def predict_proba(self, x):
        x = torch.Tensor(x)
        return torch.softmax(self.model(x), dim=1).detach().numpy()


def _faithfulness_of(model, x, coefs, base_value, num_samples):
    """Faithfulness of each explanation in ``coefs`` for the first ``num_samples`` rows."""
    from aix360.metrics.local_metrics import faithfulness_metric

    faits = []
    for sample, fi in zip(x[:num_samples], coefs[:num_samples]):
        faits.append(faithfulness_metric(model=model,
                                         x=sample,
                                         coefs=fi,
                                         base=base_value * np.ones(shape=fi.shape[0])))
    return np.array(faits)


def compute_faithfulness(neural_model, exp, all_shap_values_dict, base_value, te_set, num_samples=2000):
    """Faithfulness of the Server and of every client explanation of experiment ``exp``.

    ``te_set`` is a ``TensorDataset`` with one-hot targets.

    Returns ``(class_idxs, faith_dict, activations)`` where ``class_idxs[i]``
    are the test indices of class ``i``, ``faith_dict = {"Server": [array],
    "Clients": [array_c1, array_c2, ...]}`` and ``activations`` are the
    predicted probabilities of the model on the test set.
    """
    shap_values_dict = all_shap_values_dict[exp]
    model = aix_model(neural_model)
    num_clients = len(shap_values_dict["Clients"])

    num_classes = te_set.tensors[1].shape[1]
    ordinal_targets = te_set.tensors[1].argmax(axis=1)
    class_idxs = [np.where(ordinal_targets == i)[0] for i in range(num_classes)]
    x = te_set.tensors[0].numpy()

    activations = model.predict_proba(x)
    print(activations.shape)

    faith_dict = {"Server": [], "Clients": []}
    faith_dict["Server"].append(_faithfulness_of(model, x, shap_values_dict["Server"], base_value, num_samples))
    for client_coefs in shap_values_dict["Clients"][:num_clients]:
        faith_dict["Clients"].append(_faithfulness_of(model, x, client_coefs, base_value, num_samples))

    return class_idxs, faith_dict, activations


def recalculate_faiths(neural_model, exp, all_shap_values_dict, base_value, te_set, proposals, num_samples=2000):
    """Faithfulness of the aggregated explanations (``proposals``) of experiment ``exp``.

    Returns ``(class_idxs, faith_dict, activations)`` with
    ``faith_dict = {proposal: [array]}``.
    """
    shap_values_dict = all_shap_values_dict[exp]
    model = aix_model(neural_model)

    # NOTE: unlike compute_faithfulness, targets are compared without argmax
    # here; class_idxs is kept as in the original analysis.
    num_classes = te_set.tensors[1].shape[1]
    class_idxs = [np.where(te_set.tensors[1] == i)[0] for i in range(num_classes)]
    x = te_set.tensors[0].numpy()

    activations = model.predict_proba(x)
    print(activations.shape)

    faith_dict = {p: [_faithfulness_of(model, x, shap_values_dict[p], base_value, num_samples)]
                  for p in proposals}
    return class_idxs, faith_dict, activations


def get_exps_faiths(neural_model, exp, base_values, all_shap_values_dict, te_set):
    """Run :func:`compute_faithfulness` for several base values and print a summary
    of the share of explanations with Faithfulness > 0 and > 0.7."""
    results_dict = {}
    activas = {}
    for b_v in base_values:
        print("Base value:", b_v)
        indexes, results_dict[b_v], activas[b_v] = compute_faithfulness(
            neural_model, exp=exp, all_shap_values_dict=all_shap_values_dict,
            base_value=b_v, te_set=te_set)

        server = np.array(results_dict[b_v]["Server"])
        n_server = len(results_dict[b_v]["Server"][0])
        print("Server:")
        print("\t>0:", (server > 0).sum() / n_server, " \t\t>0.7:", (server > 0.7).sum() / n_server)
        print("Clients:")
        for client in results_dict[b_v]["Clients"]:
            client = np.array(client)
            print("\t>0:", (client > 0).sum() / len(client), " \t\t>0.7:", (client > 0.7).sum() / len(client))
    return indexes, results_dict, activas
