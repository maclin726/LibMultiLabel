"""
Replace TF-IDF with BM25 as the lexical scorer and evaluate the same rank fusion.

Each document is the query and each label description is a BM25 "document",
so document frequencies and lengths come from the label descriptions. Tree
scores are unchanged. For reference, the TF-IDF scorer is evaluated in the
same run and should reproduce the numbers of sweep_fusion.py.

Usage:
    bm25_fusion.py --model_path=<model_path> \
    --train_instance_data_path=<train_data> \
    --test_instance_data_path=<test_data> \
    --label_feature_path=<label_data> \
    --test_text_path=<test_text> \
    --label_description_path=<label_desc> \
    --label_mapping_path=<label_mapping> \
    --output_path=<output_json> \
    [--ordered_descriptions] \
    [--propensity_a=<a>] \
    [--propensity_b=<b>] \
    [--batch_size=<n>] \
    [--k1=<k1>] \
    [--b=<b>]

Options:
    --model_path=<model_path>                Path to the model file (string).
    --train_instance_data_path=<train_data>  Path to the training instance data file (string).
    --test_instance_data_path=<test_data>    Path to the testing instance data file (string).
    --label_feature_path=<label_data>        Path to the label data file (string).
    --test_text_path=<test_text>             Raw test file: id<TAB>labels<TAB>text (string).
    --label_description_path=<label_desc>    Label descriptions, one per line (string).
    --label_mapping_path=<label_mapping>     Original label id for each label index (string).
    --output_path=<output_json>              Where to write the results (string).
    --ordered_descriptions                   Descriptions are already in label-index order, not
                                             indexed by the original label id.
    --propensity_a=<a>                       PSP propensity parameter A [default: 0.55].
    --propensity_b=<b>                       PSP propensity parameter B [default: 1.5].
    --batch_size=<n>                         Test batch size [default: 64].
    --k1=<k1>                                BM25 term-frequency saturation [default: 1.2].
    --b=<b>                                  BM25 length normalization [default: 0.75].
"""

import json
import math
import os
import re
import time

import numpy as np
import scipy.sparse as sparse
from docopt import docopt
from sklearn.feature_extraction.text import CountVectorizer

import libmultilabel.linear as linear
from model_predict import (
    METRIC_LIST,
    MixedPredictor,
    binarize_labels,
    count_label_positives,
    load_experiment_data,
    load_model,
)
from sweep_fusion import DEFAULT_K, fuse

# (name, alpha, beta) for every scorer; alpha=0, beta=0 ranks all labels by the scorer alone.
CONFIGS = (("fused", 1.0, 0.001), ("fused", 1.0, 0.01), ("scorer_only", 0.0, 0.0))
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9&#!?+.-]+")


def tokenize(text):
    """Tokenizer used to build the TF-IDF features."""
    return TOKEN_PATTERN.findall(text)


def load_label_descriptions(description_path, mapping_path, ordered=False):
    """Return label descriptions ordered by label index, and the index of each original label id."""
    with open(description_path, encoding="latin-1") as description_file:
        descriptions = [line.rstrip("\r\n") for line in description_file]
    with open(mapping_path) as mapping_file:
        mapping = [line.strip() for line in mapping_file if line.strip()]
    if not ordered:
        descriptions = [descriptions[int(label)] for label in mapping]
    if len(descriptions) != len(mapping):
        raise ValueError(f"{len(descriptions)} label descriptions, but {len(mapping)} labels in {mapping_path}")
    return descriptions, {label: i for i, label in enumerate(mapping)}


def check_description_alignment(label_features, descriptions, min_correlation=0.9):
    """Check that description i belongs to label-feature row i.

    The number of distinct terms in a description should track the number of
    nonzero features in its row; a wrong label order destroys this relation.
    """
    if len(descriptions) != label_features.shape[0]:
        raise ValueError(f"{len(descriptions)} label descriptions, but {label_features.shape[0]} label feature rows")
    term_counts = np.array([len(set(tokenize(text.lower()))) for text in descriptions])
    feature_counts = np.diff(label_features.tocsr().indptr)
    correlation = np.corrcoef(term_counts, feature_counts)[0, 1]
    print(f"description length vs label features: correlation {correlation:.3f}")
    if not correlation >= min_correlation:
        raise ValueError("label descriptions do not line up with the label feature rows")


def load_test_texts(test_text_path, label_to_index, y_test):
    """Read raw test texts and check that rows and labels match the SVM test file."""
    texts, mismatches = [], 0
    with open(test_text_path, encoding="utf-8") as test_file:
        for line in test_file:
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t", 2)
            texts.append(parts[2] if len(parts) == 3 else "")
            row = len(texts) - 1
            if row < len(y_test):
                raw_labels = {label_to_index[label] for label in parts[1].split()}
                mismatches += raw_labels != {int(label) for label in y_test[row]}
    if len(texts) != len(y_test):
        raise ValueError(f"{test_text_path} has {len(texts)} rows, the SVM test file has {len(y_test)}")
    if mismatches:
        raise ValueError(f"{mismatches} test rows have labels that differ from the SVM test file")
    print(f"raw test texts match the SVM test file: {len(texts)} rows")
    return texts


def bm25_label_weights(descriptions, k1, b):
    """Return the fitted vectorizer and the BM25 weight of every term in every label description."""
    vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=True, token_pattern=None, dtype=np.float32)
    term_counts = vectorizer.fit_transform(descriptions).tocoo()
    num_labels = term_counts.shape[0]
    document_frequency = np.bincount(term_counts.col, minlength=term_counts.shape[1])
    idf = np.log(1 + (num_labels - document_frequency + 0.5) / (document_frequency + 0.5))
    lengths = np.asarray(term_counts.sum(axis=1)).ravel()
    length_ratio = lengths[term_counts.row] / max(lengths.mean(), 1e-12)
    tf = term_counts.data
    weights = idf[term_counts.col] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length_ratio))
    return vectorizer, sparse.csr_matrix((weights, (term_counts.row, term_counts.col)), shape=term_counts.shape)


def main():
    args = docopt(__doc__)
    propensity_a = float(args["--propensity_a"])
    propensity_b = float(args["--propensity_b"])
    batch_size = int(args["--batch_size"])

    model = load_model(args["--model_path"])
    X_train, y_train, X_test, y_test, X_label = load_experiment_data(
        model,
        args["--train_instance_data_path"],
        args["--test_instance_data_path"],
        args["--label_feature_path"],
    )
    num_labels = X_label.shape[0]
    descriptions, label_to_index = load_label_descriptions(
        args["--label_description_path"], args["--label_mapping_path"], args["--ordered_descriptions"]
    )
    check_description_alignment(X_label, descriptions)
    test_texts = load_test_texts(args["--test_text_path"], label_to_index, y_test)

    vectorizer, label_weights = bm25_label_weights(descriptions, float(args["--k1"]), float(args["--b"]))
    queries = (vectorizer.transform(test_texts) > 0).astype(np.float32).tocsr()
    print(f"BM25 vocabulary: {label_weights.shape[1]} terms")

    label_pos_count = count_label_positives(y_train, num_labels)
    y_train, y_test = binarize_labels(y_train, y_test, num_labels)
    seen_labels = np.nonzero(np.sum(y_train, axis=0)[0])[1]
    all_labels = np.arange(num_labels, dtype="int")
    unseen_labels = np.setdiff1d(all_labels, seen_labels)
    print(f"labels: {num_labels}, seen: {seen_labels.shape[0]}, unseen: {unseen_labels.shape[0]}")

    predictor = MixedPredictor(all_labels, seen_labels, model, X_label, "rank_rrf")
    configs = [(scorer, *config) for scorer in ("tfidf", "bm25") for config in CONFIGS]
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
        s_hat_full, tfidf_full = predictor._expand_base_scores(placeholder, s_hat_seen, seen_doc_sim, unseen_doc_sim)
        s_hat_ranks = predictor.ranks_of(s_hat_full)
        lexical = {"tfidf": tfidf_full, "bm25": (queries[batch] @ label_weights.T).toarray()}
        lexical_ranks = {scorer: predictor.ranks_of(scores) for scorer, scores in lexical.items()}
        targets = y_test[batch].toarray()

        for (scorer, _, alpha, beta), metric in zip(configs, metrics):
            scores = fuse(
                "rank_rrf", s_hat_full, lexical[scorer], s_hat_ranks, lexical_ranks[scorer],
                predictor.seen_mask, alpha, beta, DEFAULT_K,
            )
            metric.update(scores, targets)

        if batch_num % 100 == 0:
            print(f"batch {batch_num + 1}/{num_batches}, {time.perf_counter() - start_time:.0f}s", flush=True)

    results = []
    for (scorer, name, alpha, beta), metric in zip(configs, metrics):
        values = metric.compute()
        results.append({"scorer": scorer, "sweep": name, "alpha": alpha, "beta": beta, "k": DEFAULT_K,
                        **{m: values[m] for m in METRIC_LIST}})
        print(linear.tabulate_metrics(values, f"{scorer} {name}: alpha={alpha} beta={beta}"), flush=True)

    os.makedirs(os.path.dirname(args["--output_path"]), exist_ok=True)
    with open(args["--output_path"], "w") as output_file:
        json.dump(
            {
                "model_path": args["--model_path"],
                "test_instance_data_path": args["--test_instance_data_path"],
                "bm25_k1": float(args["--k1"]),
                "bm25_b": float(args["--b"]),
                "num_labels": int(num_labels),
                "num_seen": int(seen_labels.shape[0]),
                "num_unseen": int(unseen_labels.shape[0]),
                "seconds": time.perf_counter() - start_time,
                "results": results,
            },
            output_file,
            indent=1,
        )
    print(f"wrote to {args['--output_path']}")


if __name__ == "__main__":
    main()
