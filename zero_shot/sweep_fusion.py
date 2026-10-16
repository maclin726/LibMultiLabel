"""
Sweep the rank-fusion parameters (alpha, beta, k) with the zero proxy.

Base scores and ranks are computed once per test batch and shared by every
configuration, so the sweep costs roughly one regular evaluation.

Special configurations:
    alpha=1, beta=1  -> tree only (unseen labels get score 0)
    alpha=0, beta=0  -> TF-IDF only (all labels ranked by document similarity)

Usage:
    sweep_fusion.py --model_path=<model_path> \
    --train_instance_data_path=<train_data> \
    --test_instance_data_path=<test_data> \
    --label_feature_path=<label_data> \
    --output_path=<output_json> \
    [--propensity_a=<a>] \
    [--propensity_b=<b>] \
    [--batch_size=<n>]

Options:
    --model_path=<model_path>                Path to the model file (string).
    --train_instance_data_path=<train_data>  Path to the training instance data file (string).
    --test_instance_data_path=<test_data>    Path to the testing instance data file (string).
    --label_feature_path=<label_data>        Path to the label data file (string).
    --output_path=<output_json>              Where to write the sweep results (string).
    --propensity_a=<a>                       PSP propensity parameter A [default: 0.55].
    --propensity_b=<b>                       PSP propensity parameter B [default: 1.5].
    --batch_size=<n>                         Test batch size [default: 64].
"""

import json
import math
import os
import pickle
import time

import numpy as np
from docopt import docopt

import libmultilabel.linear as linear
from model_predict import (
    METRIC_LIST,
    MixedPredictor,
    binarize_labels,
    count_label_positives,
    load_svm_data,
)

DEFAULT_K = 60
DEFAULT_BETA = 0.01
BETAS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
KS = (1, 5, 10, 20, 30, 60, 100, 200, 500, 1000)


class NumpyCompatUnpickler(pickle.Unpickler):
    """Load models pickled under numpy 2.x (numpy._core) with numpy 1.x (numpy.core)."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if not module.startswith("numpy._core"):
                raise
            return super().find_class("numpy.core" + module[len("numpy._core") :], name)


def load_model(model_path):
    with open(model_path, "rb") as model_file:
        return NumpyCompatUnpickler(model_file).load()["model"]


class WidthSafePredictor(MixedPredictor):
    """MixedPredictor that feeds the tree only the feature columns it was trained on.

    Test/label files may contain feature indices beyond the training vocabulary;
    those columns carry no tree weights but still count for TF-IDF similarity.
    """

    def predict_values_on_seen_label(self, x):
        num_model_features = self.supervised_model.flat_model.weights.shape[0]
        return self.supervised_model.predict_values(x[:, :num_model_features])


def load_data_any_width(model, train_path, test_path, label_feature_path):
    """Like model_predict.load_experiment_data, but pads all files to a common width instead of failing."""
    num_model_features = model.flat_model.weights.shape[0]
    (X_train, y_train), (X_test, y_test), (X_label, _) = (
        load_svm_data(path) for path in (train_path, test_path, label_feature_path)
    )
    width = max(num_model_features, X_train.shape[1], X_test.shape[1], X_label.shape[1])
    X_train, X_test, X_label = (X.tocsr().copy() for X in (X_train, X_test, X_label))
    for X in (X_train, X_test, X_label):
        X.resize((X.shape[0], width))
    return X_train, y_train, X_test, y_test, X_label


def build_configs():
    """Return the (name, alpha, beta, k) configurations to evaluate."""
    configs = [("beta", 1.0, beta, DEFAULT_K) for beta in BETAS]
    configs += [("k", 1.0, DEFAULT_BETA, k) for k in KS if k != DEFAULT_K]
    configs.append(("tfidf_only", 0.0, 0.0, DEFAULT_K))
    return configs


def fuse(s_hat_ranks, doc_ranks, seen_mask, alpha, beta, k):
    """Zero-proxy reciprocal rank fusion, identical to MixedPredictor with strategy=rank_rrf."""
    scores = np.empty(s_hat_ranks.shape)
    scores[:, seen_mask] = alpha / (s_hat_ranks[:, seen_mask] + k) + (1 - alpha) / (doc_ranks[:, seen_mask] + k)
    scores[:, ~seen_mask] = (1 - beta) / (doc_ranks[:, ~seen_mask] + k)
    return scores


def main():
    args = docopt(__doc__)
    propensity_a = float(args["--propensity_a"])
    propensity_b = float(args["--propensity_b"])
    batch_size = int(args["--batch_size"])

    model = load_model(args["--model_path"])
    X_train, y_train, X_test, y_test, X_label = load_data_any_width(
        model,
        args["--train_instance_data_path"],
        args["--test_instance_data_path"],
        args["--label_feature_path"],
    )

    num_labels = X_label.shape[0]
    label_pos_count = count_label_positives(y_train, num_labels)
    y_train, y_test = binarize_labels(y_train, y_test, num_labels)
    seen_labels = np.nonzero(np.sum(y_train, axis=0)[0])[1]
    all_labels = np.arange(num_labels, dtype="int")
    unseen_labels = np.setdiff1d(all_labels, seen_labels)
    print(f"labels: {num_labels}, seen: {seen_labels.shape[0]}, unseen: {unseen_labels.shape[0]}")
    print(f"test instances: {X_test.shape[0]}")

    predictor = WidthSafePredictor(all_labels, seen_labels, model, X_label, "rank_rrf")

    configs = build_configs()
    metrics = [
        linear.get_metrics(
            METRIC_LIST,
            num_classes=num_labels,
            unseen_labels=unseen_labels,
            label_pos_counts=label_pos_count,
            num_instances=X_train.shape[0],
            propensity_a=propensity_a,
            propensity_b=propensity_b,
        )
        for _ in configs
    ]

    start_time = time.perf_counter()
    num_batches = math.ceil(X_test.shape[0] / batch_size)
    for batch_num in range(num_batches):
        batch = slice(batch_num * batch_size, (batch_num + 1) * batch_size)
        x = X_test[batch]
        placeholder = np.zeros((x.shape[0], num_labels))
        s_hat_seen, seen_doc_sim, unseen_doc_sim = predictor._get_base_scores(x, None, batch_num)
        s_hat_full, doc_sim_full = predictor._expand_base_scores(placeholder, s_hat_seen, seen_doc_sim, unseen_doc_sim)
        s_hat_ranks = predictor.ranks_of(s_hat_full)
        doc_ranks = predictor.ranks_of(doc_sim_full)
        targets = y_test[batch].toarray()

        for (_, alpha, beta, k), metric in zip(configs, metrics):
            metric.update(fuse(s_hat_ranks, doc_ranks, predictor.seen_mask, alpha, beta, k), targets)

        if batch_num % 100 == 0:
            print(f"batch {batch_num + 1}/{num_batches}, {time.perf_counter() - start_time:.0f}s", flush=True)

    results = []
    for (name, alpha, beta, k), metric in zip(configs, metrics):
        values = metric.compute()
        results.append({"sweep": name, "alpha": alpha, "beta": beta, "k": k, **{m: values[m] for m in METRIC_LIST}})
        print(linear.tabulate_metrics(values, f"{name}: alpha={alpha} beta={beta} k={k}"), flush=True)

    os.makedirs(os.path.dirname(args["--output_path"]), exist_ok=True)
    with open(args["--output_path"], "w") as output_file:
        json.dump(
            {
                "model_path": args["--model_path"],
                "test_instance_data_path": args["--test_instance_data_path"],
                "num_labels": int(num_labels),
                "num_seen": int(seen_labels.shape[0]),
                "num_unseen": int(unseen_labels.shape[0]),
                "propensity_a": propensity_a,
                "propensity_b": propensity_b,
                "seconds": time.perf_counter() - start_time,
                "results": results,
            },
            output_file,
            indent=1,
        )
    print(f"wrote to {args['--output_path']}")


if __name__ == "__main__":
    main()
