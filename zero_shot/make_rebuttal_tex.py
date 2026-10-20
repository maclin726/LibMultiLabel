"""Turn sweep/timing JSON files under /nfs-stor/linh.vu/runs/rebuttal into LaTeX snippets.

Writes into the paper directory:
    gen-beta-sweep.tex    beta-sweep table
    gen-beta-figure.tex   pgfplots figure: ZSR@10 / ZSR@100 vs beta
    gen-k-sweep.tex       k-sweep table
    gen-tree-only.tex     tree-only / TF-IDF-only / fused rows for each dataset
    gen-runtime.tex       CPU training/inference time and peak memory
    gen-strategies.tex    raw-score / linear-rank fusion rows for each dataset (from model_predict.py logs)
    gen-strategy-sweep.tex  beta (and raw alpha) sweep for RRF, raw-score and linear-rank fusion
"""

import json
import os
import sys

ROOT = "/nfs-stor/linh.vu/runs/rebuttal"
PREDICT_LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
PAPER = sys.argv[1] if len(sys.argv) > 1 else "/l/users/linh.vu/LibMultiLabel/zeroshot_paper"
DATASETS = [
    ("eurlex4k", "EURLEX4K"),
    ("eurlex57k", "EURLEX57K"),
    ("amazoncat13k", "AmazonCat-13K"),
    ("amazontitles131k", "AmazonTitles-131K"),
    ("amazon131k", "Amazon-131K"),
]
COLUMNS = ["P@1", "P@3", "P@5", "ZSR@10", "ZSR@50", "ZSR@100"]
COLORS = ["blue", "red", "teal", "orange", "violet"]
MARKS = ["*", "square*", "triangle*", "diamond*", "pentagon*"]


def load(kind, name):
    path = os.path.join(ROOT, kind, f"{name}.json" if kind == "sweep" else f"{name}/timing.json")
    return json.load(open(path)) if os.path.exists(path) else None


def pct(value):
    return f"{100 * value:.2f}"


def find(results, sweep, **params):
    for row in results:
        if row["sweep"] == sweep and all(row[key] == value for key, value in params.items()):
            return row
    return None


def sweep_table(sweep, param, caption, label):
    sweeps = {name: load("sweep", name) for name, _ in DATASETS}
    present = [(name, title) for name, title in DATASETS if sweeps[name]]
    metrics = ["P@1", "P@5", "ZSR@100"]
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{c|" + "|".join(["ccc"] * len(present)) + "}",
        r"\toprule",
        " & ".join([""] + [rf"\multicolumn{{3}}{{c{'|' if i < len(present) - 1 else ''}}}{{{title}}}" for i, (_, title) in enumerate(present)]) + r" \\",
        " & ".join([r"$\beta$" if param == "beta" else "$k$"] + ["P@1", "P@5", "Z@100"] * len(present)) + r" \\",
        r"\midrule",
    ]
    values = sorted({row[param] for name, _ in present for row in sweeps[name]["results"] if row["sweep"] in (sweep, "beta")})
    for value in values:
        cells = []
        for name, _ in present:
            row = find(sweeps[name]["results"], "beta", alpha=1.0, beta=value, k=60) if param == "beta" else None
            if param == "k":
                row = find(sweeps[name]["results"], "k", k=value) or (find(sweeps[name]["results"], "beta", beta=0.01, k=60) if value == 60 else None)
            cells += [pct(row[m]) if row else "--" for m in metrics]
        if all(cell == "--" for cell in cells):
            continue
        shown = "1 (tree only)" if param == "beta" and value == 1.0 else f"{value:g}"
        lines.append(" & ".join([shown] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
    return "\n".join(lines)


def beta_figure():
    """ZSR@10 and ZSR@100 against beta (log scale), one curve per dataset."""
    panels = []
    for metric in ("ZSR@10", "ZSR@100"):
        plots = []
        for i, (name, title) in enumerate(DATASETS):
            sweep = load("sweep", name)
            if not sweep:
                continue
            rows = sorted((r for r in sweep["results"] if r["sweep"] == "beta" and 0 < r["beta"] < 1), key=lambda r: r["beta"])
            coords = " ".join(f"({r['beta']:g},{100 * r[metric]:.2f})" for r in rows)
            legend = rf" \addlegendentry{{{title}}}" if metric == "ZSR@100" else ""
            plots.append(rf"\addplot[color={COLORS[i]},mark={MARKS[i]},mark size=1.2pt] coordinates {{{coords}}};{legend}")
        legend_style = r", legend style={font=\tiny,at={(0.03,0.03)},anchor=south west}" if metric == "ZSR@100" else ""
        panels += [
            r"\begin{tikzpicture}",
            rf"\begin{{axis}}[width=0.5\textwidth,height=0.32\textwidth,xmode=log,xlabel={{$\beta$}},ylabel={{{metric} (\%)}},"
            rf"tick label style={{font=\scriptsize}},label style={{font=\small}},ymin=0{legend_style}]",
            *plots,
            r"\end{axis}",
            r"\end{tikzpicture}",
        ]
    return "\n".join([
        r"\begin{figure*}[t]",
        r"\centering",
        *panels,
        r"\caption{Zero-shot recall as a function of the unseen-label weight $\beta$ ($\alpha=1$, $k=60$). "
        r"P@1 is identical to the tree-only model for every $\beta>0$, and P@3/P@5 change by less than one point "
        r"(Table~\ref{tab:beta_sweep}); increasing $\beta$ mainly pushes unseen labels down the ranking, "
        r"first out of the top 10 and then out of the top 100.}",
        r"\label{fig:beta_sweep}",
        r"\end{figure*}",
        "",
    ])


def tree_only_rows():
    lines = ["% Rows to paste into each dataset's results table (same column order: PSP@1-5, P@1-5, N@1-5, Z@10-100)."]
    order = ["PSP@1", "PSP@3", "PSP@5", "P@1", "P@3", "P@5", "NDCG@1", "NDCG@3", "NDCG@5", "ZSR@10", "ZSR@50", "ZSR@100"]
    for name, title in DATASETS:
        sweep = load("sweep", name)
        if not sweep:
            continue
        lines.append(f"% {title}")
        for label, row in [
            ("TF-IDF only", find(sweep["results"], "tfidf_only")),
            ("Tree only", find(sweep["results"], "beta", beta=1.0)),
            ("OURS", find(sweep["results"], "beta", beta=0.01, k=60)),
        ]:
            lines.append(" & ".join([label] + [pct(row[m]) for m in order]) + r" \\")
    return "\n".join(lines) + "\n"


def strategy_rows():
    """Rows for the fusion-strategy ablation: same alpha=1, beta=0.01, zero proxy as OURS."""
    lines = ["% Fusion-strategy rows to paste under OURS (same column order: PSP@1-5, P@1-5, N@1-5, Z@10-100)."]
    order = ["PSP@1", "PSP@3", "PSP@5", "P@1", "P@3", "P@5", "NDCG@1", "NDCG@3", "NDCG@5", "ZSR@10", "ZSR@50", "ZSR@100"]
    for name, title in DATASETS:
        lines.append(f"% {title}")
        for label, strategy in [("Raw-score fusion", "raw"), ("Linear-rank fusion", "rank_normal")]:
            path = os.path.join(PREDICT_LOGS, f"logs_zero_{name}_zs_{strategy}_updatedPSP.json")
            if not os.path.exists(path):
                lines.append(f"% {label}: missing {os.path.basename(path)}")
                continue
            row = json.load(open(path))["full_metrics"][0]
            lines.append(" & ".join([label] + [pct(row[m]) for m in order]) + r" \\")
    return "\n".join(lines) + "\n"


def strategy_sweep_table():
    """P@1 / P@5 / ZSR@100 against beta for each fusion strategy; RRF rows come from the main sweep."""
    rrf = {name: load("sweep", name) for name, _ in DATASETS}
    ablation = {name: load("sweep", f"{name}_raw_rank_normal") for name, _ in DATASETS}
    present = [(name, title) for name, title in DATASETS if rrf[name] or ablation[name]]
    metrics = ["P@1", "P@5", "ZSR@100"]
    blocks = [
        ("Reciprocal-rank (OURS)", "rank_rrf", "beta", [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]),
        ("Raw-score", "raw", "beta", [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]),
        ("Raw-score", "raw", "alpha", [0.5, 0.9]),
        ("Linear-rank", "rank_normal", "beta", [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]),
    ]
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{Fusion strategies under the same sweep (zero proxy, $k=60$ for reciprocal rank). "
        r"Raw-score fusion adds the sigmoid tree score and the TF-IDF similarity directly; linear-rank fusion "
        r"replaces each score by $1-r/L$. Rows vary $\beta$ at $\alpha=1$, except the raw-score $\alpha$ rows "
        r"($\beta=0.01$), which also mix TF-IDF into seen-label scores.}",
        r"\label{tab:strategy_sweep}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll|" + "|".join(["ccc"] * len(present)) + "}",
        r"\toprule",
        " & ".join(["", ""] + [rf"\multicolumn{{3}}{{c{'|' if i < len(present) - 1 else ''}}}{{{title}}}" for i, (_, title) in enumerate(present)]) + r" \\",
        " & ".join(["Fusion", "Setting"] + ["P@1", "P@5", "Z@100"] * len(present)) + r" \\",
    ]
    previous = None
    for label, strategy, param, values in blocks:
        for value in values:
            cells = []
            for name, _ in present:
                if strategy == "rank_rrf":
                    row = find(rrf[name]["results"], "beta", alpha=1.0, beta=value, k=60) if rrf[name] else None
                else:
                    row = find(ablation[name]["results"], param, strategy=strategy, **{param: value}) if ablation[name] else None
                cells += [pct(row[m]) if row else "--" for m in metrics]
            if all(cell == "--" for cell in cells):
                continue
            if label != previous:
                lines.append(r"\midrule")
            setting = rf"$\{param}={value:g}$"
            lines.append(" & ".join([label if label != previous else "", setting] + cells) + r" \\")
            previous = label
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
    return "\n".join(lines)


def runtime_table():
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{CPU cost of our method. Training is a single LibMultiLabel tree; inference covers tree scoring, "
        r"TF-IDF similarity to all labels, and rank fusion over the full label space (metric computation excluded). "
        r"All runs use 32 cores of a node with two AMD EPYC 7742 processors.}",
        r"\label{tab:runtime}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & Train (min) & Infer. (ms/doc) & Peak RAM (GiB) & Model (GiB) \\",
        r"\midrule",
    ]
    for name, title in DATASETS:
        timing = load("timing", name)
        if not timing or "peak_rss_gib" not in timing:
            lines.append(f"{title} & -- & -- & -- & -- \\\\")
            continue
        lines.append(
            f"{title} & {timing['training_seconds'] / 60:.1f} & {timing['inference_ms_per_doc']:.1f} & "
            f"{timing['peak_rss_gib']:.1f} & {timing['model_size_gib']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    outputs = {
        "gen-beta-sweep.tex": sweep_table(
            "beta", "beta",
            r"Effect of the unseen-label weight $\beta$ ($\alpha=1$, $k=60$). $\beta=1$ assigns every unseen label a score of 0, which recovers the tree-only ranking.",
            "tab:beta_sweep",
        ),
        "gen-beta-figure.tex": beta_figure(),
        "gen-k-sweep.tex": sweep_table(
            "k", "k", r"Effect of the RRF constant $k$ ($\alpha=1$, $\beta=0.01$).", "tab:k_sweep"
        ),
        "gen-tree-only.tex": tree_only_rows(),
        "gen-runtime.tex": runtime_table(),
        "gen-strategies.tex": strategy_rows(),
        "gen-strategy-sweep.tex": strategy_sweep_table(),
    }
    for filename, text in outputs.items():
        with open(os.path.join(PAPER, filename), "w") as output_file:
            output_file.write(text)
        print(f"wrote {filename}")


if __name__ == "__main__":
    main()
