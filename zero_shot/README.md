# Zero-shot experiments

All experiments go through one slurm launcher, `run_zs`, which holds the data,
model and PSP settings for each dataset:

```
sbatch -J <name> run_zs <mode> <dataset> [strategy]
```

| mode      | what it runs                                   | output                                          |
|-----------|------------------------------------------------|-------------------------------------------------|
| `train`   | `main.py` tree training                        | `/nfs-stor/linh.vu/runs/<dataset>_zs_tree_*`    |
| `eval`    | `main.py --eval` on seen labels                | slurm log                                       |
| `predict` | `model_predict.py` (tree + TF-IDF fusion)      | `logs/logs_{proxy}_{run_name}_updatedPSP.json`  |
| `tfidf`   | `model_tfidf_predict.py` (TF-IDF only)         | slurm log                                       |
| `sweep`   | `sweep_fusion.py` (beta / k sweep)             | `/nfs-stor/linh.vu/runs/rebuttal/sweep/`        |
| `timing`  | `time_pipeline.py` (runtime, peak memory)      | `/nfs-stor/linh.vu/runs/rebuttal/timing/`       |

Datasets: `eurlex4k`, `eurlex57k`, `amazoncat13k`, `amazontitles131k`, `amazon131k`.

Slurm logs go to `/nfs-stor/linh.vu/runs/slurm_logs/<job-name>.<job-id>.out`.
Defaults are 128G / 32 CPUs / 48h; override on the command line when needed,
e.g. `sbatch --mem=200G -J amz131k_train run_zs train amazon131k`.

## Strategies (`predict` mode)

- `rank_rrf` (default): Reciprocal Rank Fusion.
- `raw`: raw scores only.
- `rank_normal`: ranking only.

## Paper tables

`python make_rebuttal_tex.py [paper_dir]` turns the sweep/timing JSON into the
`gen-*.tex` files in `../zeroshot_paper`.
`baseline_epoch_times.py <log files>` extracts per-epoch times from baseline logs.

Old launchers and logs are archived in `/nfs-stor/linh.vu/runs/old_logs/`.
