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
from docopt import docopt
from sklearn.datasets import load_svmlight_file
import scipy.sparse as sparse
from sklearn.preprocessing import MultiLabelBinarizer

import libmultilabel.linear as linear


BATCH_SIZE = 16
RRF_K = 60
LOG_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "logs")

# Experiment grid. Change these values to run a different search.
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

def load_svm_features_keep_empty(file_path):
    """Load SVMlight rows while preserving blank zero-feature rows."""
    with open(file_path, "r") as data_file:
        empty_rows = [not line.strip() for line in data_file]

    nonempty_features, _ = load_svm_data(file_path)
    num_features = nonempty_features.shape[1]
    rows = []
    nonempty_index = 0
    for is_empty in empty_rows:
        if is_empty:
            rows.append(sparse.csr_matrix((1, num_features), dtype=nonempty_features.dtype))
        else:
            rows.append(nonempty_features[nonempty_index])
            nonempty_index += 1

    return sparse.vstack(rows, format="csr")


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

        self.seen_mask = np.zeros(self.all_label_map.shape[0], dtype=bool)
        self.seen_mask[self.seen_labels] = True
        self.unseen_mask = ~self.seen_mask

    def predict_values_on_seen_label(self, x):
        # Columns beyond the training vocabulary have no tree weights; they only count for TF-IDF.
        return self.supervised_model.predict_values(x[:, : num_model_features(self.supervised_model)])

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

    def _combine_scores(
        self,
        predictions,
        alpha,
        beta,
        s_hat_seen,
        seen_label_doc_sim,
        unseen_label_doc_sim,
        s_hat_ranks,
        doc_ranks,
    ):
        """Fuse supervised and document-similarity scores; unseen labels use TF-IDF only."""
        if self.strategy == "raw":
            predictions[:, self.seen_labels] = alpha * s_hat_seen + (1 - alpha) * seen_label_doc_sim
            predictions[:, self.unseen_labels] = (1 - beta) * unseen_label_doc_sim

        elif self.strategy == "rank_rrf":
            predictions[:, self.seen_mask] = alpha * (1.0 / (s_hat_ranks[:, self.seen_mask] + RRF_K)) + (1 - alpha) * (
                1.0 / (doc_ranks[:, self.seen_mask] + RRF_K)
            )
            predictions[:, self.unseen_mask] = (1 - beta) * (1.0 / (doc_ranks[:, self.unseen_mask] + RRF_K))

        elif self.strategy == "rank_normal":
            num_labels = predictions.shape[1]
            predictions[:, self.seen_mask] = alpha * (1.0 - s_hat_ranks[:, self.seen_mask] / num_labels) + (
                1 - alpha
            ) * (1.0 - doc_ranks[:, self.seen_mask] / num_labels)
            predictions[:, self.unseen_mask] = (1 - beta) * (1.0 - doc_ranks[:, self.unseen_mask] / num_labels)

        return predictions

    def predict_on_all_label(self, x, alpha, beta, save_path=None, batch_num=None):
        """Predict all seen and unseen labels for a batch of instances."""
        predictions = np.zeros((x.shape[0], self.all_label_map.shape[0]))
        s_hat_seen, seen_label_doc_sim, unseen_label_doc_sim = self._get_base_scores(x, save_path, batch_num)
        s_hat_full, doc_sim_full = self._expand_base_scores(
            predictions,
            s_hat_seen,
            seen_label_doc_sim,
            unseen_label_doc_sim,
        )
        s_hat_ranks = self.ranks_of(s_hat_full)
        doc_ranks = self.ranks_of(doc_sim_full)
        return self._combine_scores(
            predictions,
            alpha,
            beta,
            s_hat_seen,
            seen_label_doc_sim,
            unseen_label_doc_sim,
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


class NumpyCompatUnpickler(pickle.Unpickler):
    """Load models pickled under numpy 2.x (numpy._core) with numpy 1.x (numpy.core)."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if not module.startswith("numpy._core"):
                raise
            return super().find_class("numpy.core" + module[len("numpy._core") :], name)


def num_model_features(model):
    return model.weights.shape[0] if model.name == "1vsrest" else model.flat_model.weights.shape[0]


def load_model(model_path):
    """Load the trained linear model and report its load time."""
    start_time = time.time()
    with open(model_path, "rb") as model_file:
        print(model_path)
        model = NumpyCompatUnpickler(model_file).load()["model"]
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    return model


def load_experiment_data(model, train_data_path, test_data_path, label_feature_path):
    """Load train/test instances and label features, padded to a common feature width."""
    (X_train, y_train), (X_test, y_test), (X_label, _) = (
        load_svm_data(path) for path in (train_data_path, test_data_path, label_feature_path)
    )
    width = max(num_model_features(model), X_train.shape[1], X_test.shape[1], X_label.shape[1])
    X_train, X_test, X_label = (X.tocsr().copy() for X in (X_train, X_test, X_label))
    for X in (X_train, X_test, X_label):
        X.resize((X.shape[0], width))
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
    """Evaluate every configured alpha/beta combination and write logs."""
    os.makedirs(LOG_DIRECTORY, exist_ok=True)

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
            )
            logs["alpha"].append(alpha)
            logs["beta"].append(beta)
            logs["full_metrics"].append({metric: metric_dict[metric] for metric in METRIC_LIST})

            title = f"a={alpha} b={beta} Test"
            print(linear.tabulate_metrics(metric_dict, title), flush=True)

    timing_seconds = dict(shared_timing_seconds)
    timing_seconds["evaluation"] = time.perf_counter() - evaluation_start_time
    timing_seconds["total"] = sum(timing_seconds.values())
    logs["timing_seconds"] = timing_seconds

    # The "zero" prefix (zero proxy) is kept so existing log names stay valid.
    log_filename = f"logs_zero_{run_name}_updatedPSP.json"
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
