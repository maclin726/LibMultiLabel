"""
Error analysis of unseen-label retrieval with the final fusion setting.

For every test document and each of its relevant unseen labels, record the
label's rank in the fused ranking, its TF-IDF similarity, how many features
the document and the label description share, and the shared terms that
contribute most to the similarity. Then summarize recall of these pairs by
the amount of lexical overlap.

Shared terms are named with a TF-IDF vectorizer refitted on the raw training
documents and label descriptions, so their weights approximate, but need not
equal, those of the precomputed features used for ranking.

Usage:
    error_analysis.py --model_path=<model_path> \
    --train_instance_data_path=<train_data> \
    --test_instance_data_path=<test_data> \
    --label_feature_path=<label_data> \
    --train_text_path=<train_text> \
    --test_text_path=<test_text> \
    --label_description_path=<label_desc> \
    --label_mapping_path=<label_mapping> \
    --output_dir=<output_dir> \
    [--ordered_descriptions] \
    [--batch_size=<n>] \
    [--beta=<beta>]

Options:
    --model_path=<model_path>                Path to the model file (string).
    --train_instance_data_path=<train_data>  Path to the training instance data file (string).
    --test_instance_data_path=<test_data>    Path to the testing instance data file (string).
    --label_feature_path=<label_data>        Path to the label data file (string).
    --train_text_path=<train_text>           Raw training file: id<TAB>labels<TAB>text (string).
    --test_text_path=<test_text>             Raw test file: id<TAB>labels<TAB>text (string).
    --label_description_path=<label_desc>    Label descriptions, one per line (string).
    --label_mapping_path=<label_mapping>     Original label id for each label index (string).
    --output_dir=<output_dir>                Directory for pairs.csv and summary.json (string).
    --ordered_descriptions                   Descriptions are already in label-index order, not
                                             indexed by the original label id.
    --batch_size=<n>                         Test batch size [default: 64].
    --beta=<beta>                            Unseen-label weight [default: 0.001].
"""

import csv
import json
import math
import os

import numpy as np
from docopt import docopt
from sklearn.feature_extraction.text import TfidfVectorizer

from bm25_fusion import check_description_alignment, load_label_descriptions, load_test_texts, tokenize
from model_predict import MixedPredictor, binarize_labels, load_experiment_data, load_model
from sweep_fusion import DEFAULT_K, fuse

TOP_TERMS = 8
EXCERPT_CHARS = 400
OVERLAP_BINS = ((0, 0), (1, 2), (3, 5), (6, 10), (11, None))
CUTOFFS = (10, 100)


def load_train_texts(train_text_path):
    with open(train_text_path, encoding="utf-8") as train_file:
        return [
            parts[2] if len(parts) == 3 else ""
            for parts in (line.rstrip("\r\n").split("\t", 2) for line in train_file if line.strip())
        ]


def top_shared_terms(doc_vector, label_vector, terms):
    """Shared terms ordered by their contribution to the cosine similarity."""
    shared, doc_positions, label_positions = np.intersect1d(
        doc_vector.indices, label_vector.indices, return_indices=True
    )
    contributions = doc_vector.data[doc_positions] * label_vector.data[label_positions]
    order = np.argsort(-contributions)[:TOP_TERMS]
    return "; ".join(f"{terms[shared[i]]} ({contributions[i]:.3f})" for i in order)


def recall_summary(pairs, key):
    return {
        "pairs": len(pairs),
        **{f"recall@{cutoff}": (sum(p["fused_rank"] < cutoff for p in pairs) / len(pairs) if pairs else None)
           for cutoff in CUTOFFS},
        "key": key,
    }


def main():
    args = docopt(__doc__)
    batch_size = int(args["--batch_size"])
    beta = float(args["--beta"])

    model = load_model(args["--model_path"])
    X_train, y_train, X_test, y_test_raw, X_label = load_experiment_data(
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
    index_to_label = {i: label for label, i in label_to_index.items()}
    test_texts = load_test_texts(args["--test_text_path"], label_to_index, y_test_raw)

    vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=True, token_pattern=None, dtype=np.float32)
    vectorizer.fit(load_train_texts(args["--train_text_path"]) + descriptions)
    terms = vectorizer.get_feature_names_out()
    refit_docs = vectorizer.transform(test_texts).tocsr()
    refit_labels = vectorizer.transform(descriptions).tocsr()

    y_train, y_test = binarize_labels(y_train, y_test_raw, num_labels)
    seen_labels = np.nonzero(np.sum(y_train, axis=0)[0])[1]
    all_labels = np.arange(num_labels, dtype="int")
    predictor = MixedPredictor(all_labels, seen_labels, model, X_label, "rank_rrf")
    unseen_mask = predictor.unseen_mask
    label_nonzeros = [set(X_label[j].indices) for j in range(num_labels)]

    pairs, instance_recall = [], []
    num_batches = math.ceil(X_test.shape[0] / batch_size)
    for batch_num in range(num_batches):
        start = batch_num * batch_size
        x = X_test[start : start + batch_size]
        placeholder = np.zeros((x.shape[0], num_labels))
        s_hat_seen, seen_doc_sim, unseen_doc_sim = predictor._get_base_scores(x, None, batch_num)
        s_hat_full, doc_sim_full = predictor._expand_base_scores(placeholder, s_hat_seen, seen_doc_sim, unseen_doc_sim)
        s_hat_ranks = predictor.ranks_of(s_hat_full)
        doc_ranks = predictor.ranks_of(doc_sim_full)
        scores = fuse("rank_rrf", s_hat_full, doc_sim_full, s_hat_ranks, doc_ranks, predictor.seen_mask, 1.0, beta, DEFAULT_K)
        fused_ranks = predictor.ranks_of(scores)

        for offset in range(x.shape[0]):
            row = start + offset
            relevant_unseen = [j for j in y_test[row].indices if unseen_mask[j]]
            if not relevant_unseen:
                continue
            doc_nonzeros = set(x[offset].indices)
            instance_recall.append(np.mean([fused_ranks[offset, j] < 100 for j in relevant_unseen]))
            for j in relevant_unseen:
                pairs.append({
                    "test_row": row,
                    "label_index": int(j),
                    "original_label": index_to_label[int(j)],
                    "description": descriptions[j],
                    "fused_rank": int(fused_ranks[offset, j]),
                    "tfidf_rank": int(doc_ranks[offset, j]),
                    "tfidf_score": float(doc_sim_full[offset, j]),
                    "shared_features": len(doc_nonzeros & label_nonzeros[j]),
                    "top_shared_terms": top_shared_terms(refit_docs[row], refit_labels[j], terms),
                    "doc_excerpt": test_texts[row][:EXCERPT_CHARS],
                })

    os.makedirs(args["--output_dir"], exist_ok=True)
    with open(os.path.join(args["--output_dir"], "pairs.csv"), "w", newline="", encoding="utf-8") as pairs_file:
        writer = csv.DictWriter(pairs_file, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)

    by_overlap = []
    for low, high in OVERLAP_BINS:
        bucket = [p for p in pairs if p["shared_features"] >= low and (high is None or p["shared_features"] <= high)]
        by_overlap.append(recall_summary(bucket, f"{low}+" if high is None else f"{low}-{high}"))
    quartiles = np.quantile([p["tfidf_score"] for p in pairs], [0.25, 0.5, 0.75])
    edges = [-np.inf, *quartiles, np.inf]
    by_similarity = [
        recall_summary([p for p in pairs if edges[q] < p["tfidf_score"] <= edges[q + 1]], f"Q{q + 1}")
        for q in range(4)
    ]
    summary = {
        "beta": beta,
        "k": DEFAULT_K,
        "instances_with_unseen": len(instance_recall),
        "ZSR@100": float(np.mean(instance_recall)),
        "similarity_quartile_edges": [float(q) for q in quartiles],
        "by_shared_features": by_overlap,
        "by_similarity_quartile": by_similarity,
    }
    with open(os.path.join(args["--output_dir"], "summary.json"), "w") as summary_file:
        json.dump(summary, summary_file, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
