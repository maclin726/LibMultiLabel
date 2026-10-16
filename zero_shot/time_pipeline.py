"""
Measure wall-clock time and peak memory of the full CPU pipeline:
tree training, then zero-shot inference over the full label space.

Inference timing excludes metric computation and the label nearest-neighbor
search, which the zero-proxy configuration does not use.

Usage:
    time_pipeline.py --training_file=<train_data> \
    --test_file=<test_data> \
    --label_feature_path=<label_data> \
    --output_dir=<output_dir> \
    [--batch_size=<n>] \
    [--beta=<beta>]

Options:
    --training_file=<train_data>         Training data in SVM format (string).
    --test_file=<test_data>              Test data in SVM format (string).
    --label_feature_path=<label_data>    Label TF-IDF features in SVM format (string).
    --output_dir=<output_dir>            Where to save the model and timing.json (string).
    --batch_size=<n>                     Test batch size for inference [default: 64].
    --beta=<beta>                        Unseen-label fusion weight [default: 0.01].
"""

import json
import math
import os
import platform
import resource
import time

import numpy as np
from docopt import docopt

import libmultilabel.linear as linear
from model_predict import RRF_K, binarize_labels
from sweep_fusion import WidthSafePredictor, load_data_any_width, load_model

TOP_K = 100


class ZeroProxyPredictor(WidthSafePredictor):
    """MixedPredictor without the label kNN search that only non-zero proxies need."""

    def get_kneighbors(self, n_neighbors=None):
        return None


def peak_rss_gib():
    """Peak resident set size of this process so far (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def main():
    args = docopt(__doc__)
    output_dir = args["--output_dir"]
    batch_size = int(args["--batch_size"])
    beta = float(args["--beta"])
    os.makedirs(output_dir, exist_ok=True)
    timing = {
        "hostname": platform.node(),
        "cpus_allocated": len(os.sched_getaffinity(0)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }

    def log(key, value):
        timing[key] = value
        print(f"{key}: {value}", flush=True)
        with open(os.path.join(output_dir, "timing.json"), "w") as timing_file:
            json.dump(timing, timing_file, indent=1)

    timing_path = os.path.join(output_dir, "timing.json")
    model_path = os.path.join(output_dir, "linear_pipeline.pickle")
    previous = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
    if "model_size_gib" in previous and os.path.exists(model_path):
        # Training already finished in an earlier job: keep its numbers and only redo inference.
        print("Reusing trained model and training timings from", output_dir, flush=True)
        training_keys = ("train_data_loading_seconds", "num_train_instances", "num_seen_labels",
                         "training_seconds", "peak_rss_after_training_gib", "model_size_gib")
        timing.update({key: previous[key] for key in training_keys})
        timing["training_job_id"] = previous.get("slurm_job_id")
        model = load_model(model_path)
    else:
        # Training, following main.py / linear_trainer.py.
        start = time.perf_counter()
        preprocessor = linear.Preprocessor(include_test_labels=False, remove_no_label_data=False)
        datasets = linear.load_dataset("svm", args["--training_file"], args["--test_file"])
        datasets = preprocessor.fit_transform(datasets)
        log("train_data_loading_seconds", time.perf_counter() - start)
        log("num_train_instances", int(datasets["train"]["x"].shape[0]))
        log("num_seen_labels", int(datasets["train"]["y"].shape[1]))

        start = time.perf_counter()
        model = linear.train_tree(datasets["train"]["y"], datasets["train"]["x"], options=None, K=100, dmax=10)
        log("training_seconds", time.perf_counter() - start)
        log("peak_rss_after_training_gib", peak_rss_gib())

        linear.save_pipeline(output_dir, preprocessor, model)
        log("model_size_gib", os.path.getsize(model_path) / 2**30)
        del datasets

    # Inference, following model_predict.py with alpha=1 and the zero proxy.
    start = time.perf_counter()
    X_train, y_train, X_test, y_test, X_label = load_data_any_width(
        model, args["--training_file"], args["--test_file"], args["--label_feature_path"]
    )
    num_labels = X_label.shape[0]
    y_train, _ = binarize_labels(y_train, y_test, num_labels)
    seen_labels = np.nonzero(np.sum(y_train, axis=0)[0])[1]
    predictor = ZeroProxyPredictor(np.arange(num_labels, dtype="int"), seen_labels, model, X_label, "rank_rrf")
    log("inference_setup_seconds", time.perf_counter() - start)
    log("num_test_instances", int(X_test.shape[0]))
    log("num_labels", int(num_labels))

    tree_seconds = tfidf_seconds = fusion_seconds = 0.0
    num_batches = math.ceil(X_test.shape[0] / batch_size)
    for batch_num in range(num_batches):
        x = X_test[batch_num * batch_size : (batch_num + 1) * batch_size]

        start = time.perf_counter()
        s_hat_seen = predictor.predict_values_on_seen_label(x)
        tree_seconds += time.perf_counter() - start

        start = time.perf_counter()
        seen_doc_sim = (x @ predictor.seen_label_feature.T).toarray()
        unseen_doc_sim = (x @ predictor.unseen_label_feature.T).toarray()
        tfidf_seconds += time.perf_counter() - start

        start = time.perf_counter()
        scores = np.zeros((x.shape[0], num_labels))
        s_hat_full, doc_sim_full = predictor._expand_base_scores(scores, s_hat_seen, seen_doc_sim, unseen_doc_sim)
        s_hat_ranks = predictor.ranks_of(s_hat_full)
        doc_ranks = predictor.ranks_of(doc_sim_full)
        seen = predictor.seen_mask
        scores[:, seen] = 1.0 / (s_hat_ranks[:, seen] + RRF_K)
        scores[:, ~seen] = (1 - beta) / (doc_ranks[:, ~seen] + RRF_K)
        np.argpartition(-scores, TOP_K, axis=1)[:, :TOP_K]
        fusion_seconds += time.perf_counter() - start

        if batch_num % 200 == 0:
            print(f"batch {batch_num + 1}/{num_batches}", flush=True)

    log("inference_tree_seconds", tree_seconds)
    log("inference_tfidf_seconds", tfidf_seconds)
    log("inference_fusion_seconds", fusion_seconds)
    inference_seconds = tree_seconds + tfidf_seconds + fusion_seconds
    log("inference_seconds", inference_seconds)
    log("inference_ms_per_doc", 1000 * inference_seconds / X_test.shape[0])
    log("peak_rss_inference_job_gib", peak_rss_gib())
    log("peak_rss_gib", max(peak_rss_gib(), timing["peak_rss_after_training_gib"]))


if __name__ == "__main__":
    main()
