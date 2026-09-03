"""
Usage:
    model_tfidf_predict.py --model_path=<model_path> \
    --train_instance_data_path=<train_data> \
    --test_instance_data_path=<test_data> \
    --label_feature_path=<label_data> \
    --run_name=<run_name_str> \
    [--propensity_a=<a>] \
    [--propensity_b=<b>]

Options:
    --model_path=<model_path>                Unused; retained for launcher compatibility.
    --train_instance_data_path=<train_data>  Path to the training instance data file.
    --test_instance_data_path=<test_data>    Path to the testing instance data file.
    --label_feature_path=<label_features>    Path to the label-feature data file.
    --run_name=<run_name_str>                Name shown in the evaluation output.
    --propensity_a=<a>                       PSP propensity parameter A [default: 0.55].
    --propensity_b=<b>                       PSP propensity parameter B [default: 1.5].
"""

import math
import os
import sys

# Allow this script to import LibMultiLabel when run from zero_shot/.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
import scipy.sparse as sparse
from docopt import docopt
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer

import libmultilabel.linear as linear


BATCH_SIZE = 256
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
    label_pos_count,
    num_instances_train,
    *,
    propensity_a=0.55,
    propensity_b=1.5,
    **predictor_kwargs,
):
    """Evaluate TF-IDF predictions without materializing all scores at once."""
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
        predictions = predictor.predict_values_on_all_label(
            X[batch_start:batch_end],
            **predictor_kwargs,
        )
        targets = y[batch_start:batch_end].toarray()
        metrics.update(predictions, targets)

    return metrics.compute()


class TfidfPredictor:
    """Score every label using instance-to-label TF-IDF similarity."""

    def __init__(self, label_feature):
        self.label_feature = label_feature

    def predict_values_on_all_label(self, x):
        return predict_values_by_tfidf(x, self.label_feature)


def parse_arguments():
    """Parse command-line arguments using the module docstring."""
    return docopt(__doc__)


def print_arguments(args):
    print(f"Model Path (unused): {args['--model_path']}")
    print(f"Train Instance Data Path: {args['--train_instance_data_path']}")
    print(f"Test Instance Data Path: {args['--test_instance_data_path']}")
    print(f"Label Feature Path: {args['--label_feature_path']}")
    print(f"PSP propensity A: {args['--propensity_a']}")
    print(f"PSP propensity B: {args['--propensity_b']}")


def pad_feature_matrix(matrix, num_features):
    """Right-pad a sparse matrix so all TF-IDF matrices share one width."""
    num_missing_features = num_features - matrix.shape[1]
    if num_missing_features == 0:
        return matrix

    padding = sparse.csr_matrix(
        (matrix.shape[0], num_missing_features),
        dtype=matrix.dtype,
    )
    return sparse.hstack((matrix, padding), format="csr")


def load_experiment_data(train_data_path, test_data_path, label_feature_path):
    """Load and align train, test, and label-feature matrices."""
    X_train, y_train = load_svm_data(train_data_path)
    X_test, y_test = load_svm_data(test_data_path)
    X_label, _ = load_svm_data(label_feature_path)

    num_features = max(X_train.shape[1], X_test.shape[1], X_label.shape[1])
    X_train = pad_feature_matrix(X_train, num_features)
    X_test = pad_feature_matrix(X_test, num_features)
    X_label = pad_feature_matrix(X_label, num_features)
    return X_train, y_train, X_test, y_test, X_label


def binarize_labels(y_train, y_test, num_labels):
    """Convert iterable label IDs to sparse indicator matrices."""
    binarizer = MultiLabelBinarizer(
        classes=np.arange(num_labels, dtype="float"),
        sparse_output=True,
    )
    binarizer.fit(y_train + y_test)
    return binarizer.transform(y_train), binarizer.transform(y_test)


def get_label_statistics(y_train):
    """Return seen-label indices and positive training counts per label."""
    label_pos_count = np.asarray(y_train.sum(axis=0)).ravel().astype(int)
    seen_labels = np.flatnonzero(label_pos_count)
    return seen_labels, label_pos_count


def main():
    args = parse_arguments()
    print_arguments(args)
    print("Start processing...")

    propensity_a = float(args["--propensity_a"])
    propensity_b = float(args["--propensity_b"])

    X_train, y_train, X_test, y_test, X_label = load_experiment_data(
        args["--train_instance_data_path"],
        args["--test_instance_data_path"],
        args["--label_feature_path"],
    )

    num_labels = X_label.shape[0]
    y_train, y_test = binarize_labels(y_train, y_test, num_labels)
    seen_labels, label_pos_count = get_label_statistics(y_train)
    all_labels = np.arange(num_labels, dtype="int")
    unseen_labels = np.setdiff1d(all_labels, seen_labels)
    print(f"Seen labels: {seen_labels.shape[0]}")
    print(f"Unseen labels: {unseen_labels.shape[0]}")

    predictor = TfidfPredictor(X_label)
    metric_dict = metrics_in_batches(
        X_test,
        y_test,
        predictor,
        unseen_labels,
        METRIC_LIST,
        label_pos_count=label_pos_count,
        num_instances_train=X_train.shape[0],
        propensity_a=propensity_a,
        propensity_b=propensity_b,
    )

    title = f"{args['--run_name']} TF-IDF"
    print(linear.tabulate_metrics(metric_dict, title), flush=True)


if __name__ == "__main__":
    main()
