"""
Sweep the rank-fusion parameters (alpha, beta, k) with the zero proxy.

Base scores and ranks are computed once per test batch and shared by every
configuration, so the sweep costs roughly one regular evaluation.

Special configurations:
    alpha=1, beta=1  -> tree only (unseen labels get score 0)
    alpha=0, beta=0  -> TF-IDF only (all labels ranked by document similarity)

The fusion rules to sweep are chosen with the strategies option (comma-separated):
    rank_rrf     beta / k sweep, tree-only and TF-IDF-only rows (default)
    raw          raw-score fusion: beta sweep at alpha=1, plus alpha sweep at beta=0.01
    rank_normal  linear-rank fusion (1 - rank/L): beta sweep at alpha=1

Usage:
    sweep_fusion.py --model_path=<model_path> \
    --train_instance_data_path=<train_data> \
    --test_instance_data_path=<test_data> \
    --label_feature_path=<label_data> \
    --output_path=<output_json> \
    [--propensity_a=<a>] \
    [--propensity_b=<b>] \
    [--batch_size=<n>] \
    [--strategies=<list>]

Options:
    --model_path=<model_path>                Path to the model file (string).
    --train_instance_data_path=<train_data>  Path to the training instance data file (string).
    --test_instance_data_path=<test_data>    Path to the testing instance data file (string).
    --label_feature_path=<label_data>        Path to the label data file (string).
    --output_path=<output_json>              Where to write the sweep results (string).
    --propensity_a=<a>                       PSP propensity parameter A [default: 0.55].
    --propensity_b=<b>                       PSP propensity parameter B [default: 1.5].
    --batch_size=<n>                         Test batch size [default: 64].
    --strategies=<list>                      Comma-separated fusion strategies [default: rank_rrf].
"""

import json
import math
import os
import time

import numpy as np
from docopt import docopt

import libmultilabel.linear as linear
from model_predict import (
    METRIC_LIST,
    MixedPredictor,
    binarize_labels,
    count_label_positives,
    load_experiment_data,
    load_model,
)

DEFAULT_K = 60
DEFAULT_BETA = 0.01
BETAS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
KS = (1, 5, 10, 20, 30, 60, 100, 200, 500, 1000)
ABLATION_BETAS = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1)
RAW_ALPHAS = (0.5, 0.9)


# model_predict now handles numpy-2 pickles and mismatched feature widths itself.
WidthSafePredictor = MixedPredictor
load_data_any_width = load_experiment_data


def build_configs(strategies):
    """Return the (strategy, name, alpha, beta, k) configurations to evaluate."""
    configs = []
    for strategy in strategies:
        if strategy == "rank_rrf":
            configs += [(strategy, "beta", 1.0, beta, DEFAULT_K) for beta in BETAS]
            configs += [(strategy, "k", 1.0, DEFAULT_BETA, k) for k in KS if k != DEFAULT_K]
            configs.append((strategy, "tfidf_only", 0.0, 0.0, DEFAULT_K))
        elif strategy in ("raw", "rank_normal"):
            configs += [(strategy, "beta", 1.0, beta, DEFAULT_K) for beta in ABLATION_BETAS]
            if strategy == "raw":
                configs += [(strategy, "alpha", alpha, DEFAULT_BETA, DEFAULT_K) for alpha in RAW_ALPHAS]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    return configs


def fuse(strategy, s_hat_full, doc_sim_full, s_hat_ranks, doc_ranks, seen_mask, alpha, beta, k):
    """Zero-proxy fusion, identical to MixedPredictor._combine_scores for the same strategy."""
    scores = np.empty(s_hat_ranks.shape)
    if strategy == "rank_rrf":
        seen_part = alpha / (s_hat_ranks + k) + (1 - alpha) / (doc_ranks + k)
        unseen_part = (1 - beta) / (doc_ranks + k)
    elif strategy == "rank_normal":
        num_labels = s_hat_ranks.shape[1]
        seen_part = alpha * (1.0 - s_hat_ranks / num_labels) + (1 - alpha) * (1.0 - doc_ranks / num_labels)
        unseen_part = (1 - beta) * (1.0 - doc_ranks / num_labels)
    else:  # raw
        seen_part = alpha * s_hat_full + (1 - alpha) * doc_sim_full
        unseen_part = (1 - beta) * doc_sim_full
    scores[:, seen_mask] = seen_part[:, seen_mask]
    scores[:, ~seen_mask] = unseen_part[:, ~seen_mask]
    return scores


def main():
    args = docopt(__doc__)
    propensity_a = float(args["--propensity_a"])
    propensity_b = float(args["--propensity_b"])
    batch_size = int(args["--batch_size"])
    strategies = args["--strategies"].split(",")

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

    configs = build_configs(strategies)
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

        for (strategy, _, alpha, beta, k), metric in zip(configs, metrics):
            scores = fuse(strategy, s_hat_full, doc_sim_full, s_hat_ranks, doc_ranks, predictor.seen_mask, alpha, beta, k)
            metric.update(scores, targets)

        if batch_num % 100 == 0:
            print(f"batch {batch_num + 1}/{num_batches}, {time.perf_counter() - start_time:.0f}s", flush=True)

    results = []
    for (strategy, name, alpha, beta, k), metric in zip(configs, metrics):
        values = metric.compute()
        results.append(
            {"strategy": strategy, "sweep": name, "alpha": alpha, "beta": beta, "k": k, **{m: values[m] for m in METRIC_LIST}}
        )
        print(linear.tabulate_metrics(values, f"{strategy} {name}: alpha={alpha} beta={beta} k={k}"), flush=True)

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
