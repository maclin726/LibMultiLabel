"""
Usage:
    model_predict.py --model_path=<model_path> \
    --train_instance_data_path=<train_data> \
    --test_instance_data_path=<test_data> \
    --label_feature_path=<label_data> \
    --run_name=<run_name_str> \
    --strategy=<strategy_str> \
    [--propensity_a=<a>] \
    [--propensity_b=<b>] \
    [--save_prediction_cache]

Options:
    --model_path=<model_path>                Path to the model file (string).
    --train_instance_data_path=<train_data>  Path to the training instance data file (string).
    --test_instance_data_path=<test_data>    Path to the testing instance data file (string).
    --label_feature_path=<label_features>    Path to the label data file (string).
    --run_name=<run_name_str>                Name of the run (string).
    --strategy=<strategy_str>                Strategy for prediction (string).
    --propensity_a=<a>                       PSP propensity parameter A [default: 0.55].
    --propensity_b=<b>                       PSP propensity parameter B [default: 1.5].
    --save_prediction_cache                  Save and reuse intermediate batch predictions.
"""

import json
import math
import os
import pickle
import sys
import time

# Allow this script to import LibMultiLabel when run from zero_shot/.
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from docopt import docopt
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

import libmultilabel.linear as linear


BATCH_SIZE = 16
LABEL_NEIGHBOR_COUNT = 5
PROXY_NEIGHBOR_COUNT = 3
RRF_K = 60
ATTENTION_PROJECTION_DIM = 1024
LOG_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "logs")

# Experiment grid. Change these values to run a different search.
PROXY_TYPES = ("zero",)
ALPHAS = (1,)
BETAS = (0.01,)
METRIC_LIST = (
    "P@1",
    "P@3",
    "P@5",
    "R@1",
    "R@3",
    "R@5",
    "NDCG@1",
    "NDCG@3",
    "NDCG@5",
    "ZSR@10",
    "ZSR@50",
    "ZSR@100",
    "PSP@1",
    "PSP@3",
    "PSP@5",
)


def load_svm_data(file_path, /, *args, **kwargs):
    """Load one-based, multilabel data in SVMlight format."""
    kwargs.setdefault("multilabel", True)
    kwargs.setdefault("zero_based", False)
    return load_svmlight_file(file_path, *args, **kwargs)


def predict_values_by_tfidf(instance_tfidf, label_tfidf):
    """Calculate instance-to-label TF-IDF similarity scores."""
    return (instance_tfidf @ label_tfidf.T).toarray()


def metrics_in_batches(
    X,
    y,
    predictor,
    unseen_labels,
    metric_list,
    cache_path,
    label_pos_count,
    num_instances_train,
    *,
    propensity_a=0.55,
    propensity_b=1.5,
    **predictor_kwargs,
):
    """Predict and update metrics without materializing all scores at once."""
    num_batches = math.ceil(X.shape[0] / BATCH_SIZE)
    metrics = linear.get_metrics(
        metric_list,
        num_classes=y.shape[1],
        unseen_labels=unseen_labels,
        label_pos_counts=label_pos_count,
        num_instances=num_instances_train,
        propensity_a=propensity_a,
        propensity_b=propensity_b,
    )

    for batch_num in range(num_batches):
        batch_start = batch_num * BATCH_SIZE
        batch_end = (batch_num + 1) * BATCH_SIZE
        predictions = predictor.predict_on_all_label(
            X[batch_start:batch_end],
            save_path=cache_path,
            batch_num=batch_num,
            **predictor_kwargs,
        )
        targets = y[batch_start:batch_end].toarray()
        metrics.update(predictions, targets)

    return metrics.compute()


class MixedPredictor:
    """Combine supervised predictions for seen labels with zero-shot scores."""

    def __init__(
        self,
        all_label_map,
        seen_labels,
        supervised_model,
        label_feature,
        strategy="raw",
    ):
        self.all_label_map = all_label_map
        self.seen_labels = seen_labels
        self.supervised_model = supervised_model
        self.label_feature = label_feature
        self.strategy = strategy

        self.unseen_labels = np.setdiff1d(all_label_map, seen_labels)
        assert (
            self.unseen_labels.shape[0] + self.seen_labels.shape[0] == self.all_label_map.shape[0]
        ), "The set of seen labels should be a subset of all labels."

        self.seen_label_feature = label_feature[self.seen_labels]
        self.unseen_label_feature = label_feature[self.unseen_labels]
        self.label_neighbors = self.get_kneighbors()

        self.seen_mask = np.zeros(self.all_label_map.shape[0], dtype=bool)
        self.seen_mask[self.seen_labels] = True
        self.unseen_mask = ~self.seen_mask

    def get_kneighbors(self, n_neighbors=LABEL_NEIGHBOR_COUNT):
        """Return seen-label neighbor indices local to ``self.seen_labels``."""
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors.fit(self.seen_label_feature)
        return neighbors.kneighbors(self.label_feature, return_distance=False)

    def predict_values_on_seen_label(self, x):
        return self.supervised_model.predict_values(x)

    def predict_values_on_unseen_label(self, x):
        return predict_values_by_tfidf(x, self.unseen_label_feature)

    @staticmethod
    def ranks_of(scores):
        """Convert scores to zero-based ranks, with the largest score ranked first."""
        return np.argsort(np.argsort(-scores, axis=1), axis=1)

    @staticmethod
    def _cache_paths(save_path, batch_num):
        cache_directory = os.path.dirname(save_path)
        return {
            "s_hat_seen": f"{cache_directory}/s_hat_seen_{batch_num}.npy",
            "seen_label_doc_sim": f"{cache_directory}/seen_label_doc_sim_{batch_num}.npy",
            "unseen_label_doc_sim": f"{cache_directory}/unseen_label_doc_sim_{batch_num}.npy",
        }

    def _get_base_scores(self, x, save_path, batch_num):
        """Load cached scores when enabled, otherwise compute them for one batch."""
        cache_paths = self._cache_paths(save_path, batch_num) if save_path else None
        if cache_paths and all(os.path.exists(path) for path in cache_paths.values()):
            return (
                np.load(cache_paths["s_hat_seen"]),
                np.load(cache_paths["seen_label_doc_sim"]),
                np.load(cache_paths["unseen_label_doc_sim"]),
            )

        start_time = time.time()
        s_hat_seen = self.predict_values_on_seen_label(x)
        s_hat_seen = 1 / (1 + np.exp(-s_hat_seen))
        seen_label_doc_sim = predict_values_by_tfidf(x, self.seen_label_feature)
        unseen_label_doc_sim = predict_values_by_tfidf(x, self.unseen_label_feature)
        elapsed = time.time() - start_time
        print(f"predicting on batch {batch_num} took {elapsed:.2f} seconds")

        if cache_paths:
            cache_directory = os.path.dirname(save_path)
            os.makedirs(cache_directory, exist_ok=True)
            np.save(cache_paths["s_hat_seen"], s_hat_seen)
            np.save(cache_paths["seen_label_doc_sim"], seen_label_doc_sim)
            np.save(cache_paths["unseen_label_doc_sim"], unseen_label_doc_sim)

        return s_hat_seen, seen_label_doc_sim, unseen_label_doc_sim

    def _expand_base_scores(self, predictions, s_hat_seen, seen_label_doc_sim, unseen_label_doc_sim):
        """Place seen and unseen scores in arrays spanning the global label space."""
        s_hat_full = np.full_like(predictions, -np.inf)
        s_hat_full[:, self.seen_mask] = s_hat_seen

        doc_sim_full = np.full_like(predictions, -np.inf)
        doc_sim_full[:, self.seen_mask] = seen_label_doc_sim
        doc_sim_full[:, self.unseen_mask] = unseen_label_doc_sim
        return s_hat_full, doc_sim_full

    def _proxy_scores(self, x, proxy_type, predictions, s_hat_full):
        """Build supervised proxy scores for unseen labels."""
        proxy = np.zeros((x.shape[0], self.unseen_labels.shape[0]))

        if proxy_type == "zero":
            return proxy

        if proxy_type == "insert_closest":
            nearest_seen_local = self.label_neighbors[self.unseen_labels, 0]
            nearest_seen_global = self.seen_labels[nearest_seen_local]
            delta_feature = self.unseen_label_feature - self.seen_label_feature[nearest_seen_local]
            if hasattr(x, "dot"):
                score_difference = x.dot(delta_feature.T)
            else:
                score_difference = x @ delta_feature.T
            sign = np.sign(score_difference.toarray())
            proxy = s_hat_full[:, nearest_seen_global] + sign * 1e-8
            s_hat_full[:, self.unseen_labels] = proxy
            return proxy

        if proxy_type == "avg":
            nearest_seen_local = self.label_neighbors[self.unseen_labels, :PROXY_NEIGHBOR_COUNT]
            nearest_seen_global = self.seen_labels[nearest_seen_local]
            proxy = np.average(s_hat_full[:, nearest_seen_global], axis=2)
            s_hat_full[:, self.unseen_labels] = proxy
            return proxy

        if proxy_type == "weighted_avg":
            nearest_seen_labels = self.label_neighbors[self.unseen_labels, :PROXY_NEIGHBOR_COUNT]
            distances, _ = (
                NearestNeighbors(n_neighbors=PROXY_NEIGHBOR_COUNT)
                .fit(self.seen_label_feature)
                .kneighbors(self.unseen_label_feature)
            )
            similarities = 1 / (distances + 1e-10)
            weights = similarities / np.sum(similarities, axis=1, keepdims=True)
            return np.sum(predictions[:, nearest_seen_labels] * weights[None, :, :], axis=2)

        if proxy_type == "attention":
            key = nn.Linear(self.seen_label_feature.shape[1], ATTENTION_PROJECTION_DIM, bias=False)
            query = nn.Linear(self.unseen_label_feature.shape[1], ATTENTION_PROJECTION_DIM, bias=False)
            seen_label_tensor = torch.tensor(self.seen_label_feature.toarray(), dtype=torch.float32)
            unseen_label_tensor = torch.tensor(self.unseen_label_feature.toarray(), dtype=torch.float32)
            weights = torch.softmax(query(unseen_label_tensor) @ key(seen_label_tensor).T, dim=-1)
            return predictions[:, self.seen_labels] @ weights.detach().numpy().T

        if proxy_type == "min":
            nearest_seen_labels = self.label_neighbors[self.unseen_labels, :PROXY_NEIGHBOR_COUNT]
            return np.min(predictions[:, nearest_seen_labels], axis=2)

        raise ValueError("Unknown proxy type for unseen labels")

    def _combine_scores(
        self,
        predictions,
        proxy_type,
        alpha,
        beta,
        s_hat_seen,
        seen_label_doc_sim,
        unseen_label_doc_sim,
        proxy,
        s_hat_ranks,
        doc_ranks,
    ):
        """Fuse supervised, proxy, and document-similarity scores."""
        if self.strategy == "raw":
            predictions[:, self.seen_labels] = alpha * s_hat_seen + (1 - alpha) * seen_label_doc_sim
            if proxy_type == "zero":
                predictions[:, self.unseen_labels] = (1 - beta) * unseen_label_doc_sim
            else:
                predictions[:, self.unseen_labels] = beta * proxy + (1 - beta) * unseen_label_doc_sim

        elif self.strategy == "rank_rrf":
            predictions[:, self.seen_mask] = alpha * (1.0 / (s_hat_ranks[:, self.seen_mask] + RRF_K)) + (1 - alpha) * (
                1.0 / (doc_ranks[:, self.seen_mask] + RRF_K)
            )
            if proxy_type == "zero":
                predictions[:, self.unseen_mask] = (1 - beta) * (1.0 / (doc_ranks[:, self.unseen_mask] + RRF_K))
            else:
                predictions[:, self.unseen_mask] = beta * (1.0 / (s_hat_ranks[:, self.unseen_mask] + RRF_K)) + (
                    1 - beta
                ) * (1.0 / (doc_ranks[:, self.unseen_mask] + RRF_K))

        elif self.strategy == "rank_normal":
            num_labels = predictions.shape[1]
            predictions[:, self.seen_mask] = alpha * (1.0 - s_hat_ranks[:, self.seen_mask] / num_labels) + (
                1 - alpha
            ) * (1.0 - doc_ranks[:, self.seen_mask] / num_labels)
            if proxy_type == "zero":
                predictions[:, self.unseen_mask] = (1 - beta) * (1.0 - doc_ranks[:, self.unseen_mask] / num_labels)
            else:
                predictions[:, self.unseen_mask] = beta * (1.0 - s_hat_ranks[:, self.unseen_mask] / num_labels) + (
                    1 - beta
                ) * (1.0 - doc_ranks[:, self.unseen_mask] / num_labels)

        return predictions

    def predict_on_all_label(self, x, alpha, beta, proxy_type, save_path=None, batch_num=None):
        """Predict all seen and unseen labels for a batch of instances."""
        predictions = np.zeros((x.shape[0], self.all_label_map.shape[0]))
        s_hat_seen, seen_label_doc_sim, unseen_label_doc_sim = self._get_base_scores(x, save_path, batch_num)
        s_hat_full, doc_sim_full = self._expand_base_scores(
            predictions,
            s_hat_seen,
            seen_label_doc_sim,
            unseen_label_doc_sim,
        )
        proxy = self._proxy_scores(x, proxy_type, predictions, s_hat_full)
        s_hat_ranks = self.ranks_of(s_hat_full)
        doc_ranks = self.ranks_of(doc_sim_full)
        return self._combine_scores(
            predictions,
            proxy_type,
            alpha,
            beta,
            s_hat_seen,
            seen_label_doc_sim,
            unseen_label_doc_sim,
            proxy,
            s_hat_ranks,
            doc_ranks,
        )


def parse_arguments():
    """Parse command-line arguments using the module docstring."""
    return docopt(__doc__)


def print_arguments(args):
    print(f"Model Path: {args['--model_path']}")
    print(f"Train Instance Data Path: {args['--train_instance_data_path']}")
    print(f"Test Instance Data Path: {args['--test_instance_data_path']}")
    print(f"Label Feature Path: {args['--label_feature_path']}")
    print(f"PSP propensity A: {args['--propensity_a']}")
    print(f"PSP propensity B: {args['--propensity_b']}")
    print(f"Prediction cache: {'enabled' if args['--save_prediction_cache'] else 'disabled'}")


def load_model(model_path):
    """Load the trained linear model and report its load time."""
    start_time = time.time()
    with open(model_path, "rb") as model_file:
        print(model_path)
        model = pickle.load(model_file)["model"]
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    return model


def load_experiment_data(model, train_data_path, test_data_path, label_feature_path):
    """Load train/test instances and label features."""
    if model.name == "1vsrest":
        num_model_features = model.weights.shape[0]
    else:
        print("Tree model")
        num_model_features = model.flat_model.weights.shape[0]

        X_train, y_train = load_svm_data(
            train_data_path, n_features=num_model_features
        )
        X_test, y_test = load_svm_data(
            test_data_path, n_features=num_model_features
        )
        X_label, _ = load_svm_data(
            label_feature_path, n_features=num_model_features
        )
    return X_train, y_train, X_test, y_test, X_label


def count_label_positives(y_train, num_labels):
    """Count positive training instances for each label."""
    label_pos_count = np.zeros(num_labels, dtype=int)
    for labels in y_train:
        for label in labels:
            label_pos_count[int(label)] += 1
    return label_pos_count


def binarize_labels(y_train, y_test, num_labels):
    """Convert iterable label IDs to sparse indicator matrices."""
    binarizer = MultiLabelBinarizer(
        classes=np.arange(num_labels, dtype="float"),
        sparse_output=True,
    )
    binarizer.fit(y_train + y_test)
    return binarizer.transform(y_train), binarizer.transform(y_test)


def run_experiments(
    X_test,
    y_test,
    predictor,
    unseen_labels,
    prediction_cache_path,
    run_name,
    label_pos_count,
    num_instances_train,
    shared_timing_seconds,
    *,
    propensity_a=0.55,
    propensity_b=1.5,
):
    """Evaluate every configured proxy/alpha/beta combination and write logs."""
    os.makedirs(LOG_DIRECTORY, exist_ok=True)

    for proxy_type in PROXY_TYPES:
        evaluation_start_time = time.perf_counter()
        logs = {
            "alpha": [],
            "beta": [],
            "propensity_a": propensity_a,
            "propensity_b": propensity_b,
            "prediction_cache_enabled": prediction_cache_path is not None,
            "full_metrics": [],
        }

        for alpha in ALPHAS:
            for beta in BETAS:
                if alpha < beta:
                    continue

                metric_dict = metrics_in_batches(
                    X_test,
                    y_test,
                    predictor,
                    unseen_labels,
                    METRIC_LIST,
                    prediction_cache_path,
                    label_pos_count=label_pos_count,
                    num_instances_train=num_instances_train,
                    propensity_a=propensity_a,
                    propensity_b=propensity_b,
                    alpha=alpha,
                    beta=beta,
                    proxy_type=proxy_type,
                )
                logs["alpha"].append(alpha)
                logs["beta"].append(beta)
                logs["full_metrics"].append({metric: metric_dict[metric] for metric in METRIC_LIST})

                title = f"a={alpha} b={beta}, proxy={proxy_type} Test"
                print(linear.tabulate_metrics(metric_dict, title), flush=True)

        timing_seconds = dict(shared_timing_seconds)
        timing_seconds["evaluation"] = time.perf_counter() - evaluation_start_time
        timing_seconds["total"] = sum(timing_seconds.values())
        logs["timing_seconds"] = timing_seconds

        log_filename = f"logs_{proxy_type}_{run_name}_updatedPSP.json"
        log_path = os.path.join(LOG_DIRECTORY, log_filename)
        with open(log_path, "w") as log_file:
            json.dump(logs, log_file)
        print(f"wrote to {log_path}")


def main():
    args = parse_arguments()
    print_arguments(args)
    print("Start processing...")

    propensity_a = float(args["--propensity_a"])
    propensity_b = float(args["--propensity_b"])
    prediction_cache_path = args["--model_path"] if args["--save_prediction_cache"] else None

    model_loading_start_time = time.perf_counter()
    model = load_model(args["--model_path"])
    model_loading_time = time.perf_counter() - model_loading_start_time

    dataset_loading_start_time = time.perf_counter()
    X_train, y_train, X_test, y_test, X_label = load_experiment_data(
        model,
        args["--train_instance_data_path"],
        args["--test_instance_data_path"],
        args["--label_feature_path"],
    )
    dataset_loading_time = time.perf_counter() - dataset_loading_start_time

    setup_start_time = time.perf_counter()
    num_labels = X_label.shape[0]
    label_pos_count = count_label_positives(y_train, num_labels)
    y_train, y_test = binarize_labels(y_train, y_test, num_labels)
    seen_labels = np.nonzero(np.sum(y_train, axis=0)[0])[1]
    all_labels = np.arange(num_labels, dtype="int")
    unseen_labels = np.setdiff1d(all_labels, seen_labels)

    predictor = MixedPredictor(
        all_labels,
        seen_labels,
        model,
        X_label,
        args["--strategy"],
    )
    setup_time = time.perf_counter() - setup_start_time

    run_experiments(
        X_test,
        y_test,
        predictor,
        unseen_labels,
        prediction_cache_path,
        args["--run_name"],
        label_pos_count,
        X_train.shape[0],
        shared_timing_seconds={
            "model_loading": model_loading_time,
            "dataset_loading": dataset_loading_time,
            "setup": setup_time,
        },
        propensity_a=propensity_a,
        propensity_b=propensity_b,
    )


if __name__ == "__main__":
    main()
