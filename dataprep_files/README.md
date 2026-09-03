# Data Preparation Utilities

These utilities prepare extreme multi-label text datasets for zero-shot
experiments, SVM-based models, PECOS, and DEXML.

## Input format

Training and test files must use one sample per line:

```text
document_id<TAB>label_1 label_2 ...<TAB>document text
```

Example:

```text
0	49 551 666	text of the first document
1	145 299	text of the second document
```

The label-description file, normally named `Y.txt`, must contain one label
description per line:

```text
description for label 0
description for label 1
description for label 2
```

Important assumptions:

- Labels are zero-based integer IDs.
- Label `i` corresponds to line `i` of `Y.txt`.
- Document text must not contain embedded tabs or newlines.
- `n_labels` must equal the number of lines in `Y.txt` and must be greater
  than the largest label ID.

## Dataset configuration

Each main utility accepts a dataset JSON through `-c` or `--config`. Start by
copying one of the provided `config_*.json` files and updating its paths:

```json
{
  "dataset_path": "/path/to/DATASET",
  "train_file": "/path/to/DATASET/original/trn.txt",
  "test_file": "/path/to/DATASET/original/tst.txt",
  "label_description_file": "/path/to/DATASET/original/Y.txt",
  "MAX_FEATURES": -1,
  "n_labels": 131073,
  "percent_zeroshot": 0.01
}
```

Configuration fields:

| Field | Meaning |
| --- | --- |
| `dataset_path` | Root directory in which generated folders are created. |
| `train_file` | Original SVM-text training file. |
| `test_file` | Original SVM-text test file. |
| `label_description_file` | Original label-description file. |
| `MAX_FEATURES` | Feature limit used by the scikit-learn TF-IDF utility. A non-positive value means no explicit limit. |
| `n_labels` | Total label count used as the DEXML label-matrix width. |
| `percent_zeroshot` | Target fraction of test labels to make unseen during training, such as `0.01` for 1%. |
| `validation_file` | Optional validation file used by the DEXML converter. |

The checked-in configs contain machine-specific absolute paths. Change them
before running the utilities on another machine or dataset.

## Recommended directory layout

```text
DATASET/
├── original/
│   ├── trn.txt
│   ├── tst.txt
│   └── Y.txt
├── zeroshot/       # created by 1.create_zeroshot.py
└── dexml/          # created by 2.create_svm_text_to_dexml.py
```

EURLEX-4K currently stores its source files directly under the dataset root
instead of under `original/`. That layout works for zero-shot generation, but
see the DEXML path warning below.

## Workflow overview

Choose the branch needed by the downstream model:

```text
original data
    |
    v
1.create_zeroshot.py
    |
    +--> 3_1.text_to_svm.py --------> scikit-learn TF-IDF SVM files
    |
    +--> 3_3.text_to_svm_pecos.py --> PECOS TF-IDF SVM files
    |
    +--> 2.create_svm_text_to_dexml.py
              |
              v
        3_2.text_to_dexml.py -------> DEXML tokens and label embeddings
```

`4.get_data_stats.py` can be run after generating a zero-shot dataset and its
SVM files.

## 1. Create a zero-shot split

```bash
python /nfs-stor/linh.vu/dataprep_files/1.create_zeroshot.py \
  --config /nfs-stor/linh.vu/dataprep_files/config_amazon131k.json
```

The script uses a fixed random seed and writes:

```text
DATASET/zeroshot/trn.txt
DATASET/zeroshot/tst.txt
DATASET/zeroshot/Y.txt
DATASET/zeroshot/stats.txt
```

`percent_zeroshot` is a target percentage of unique labels appearing in the
test set, not a percentage of training samples. Training samples containing
selected labels are removed when doing so does not remove required labels from
the final task. Check `stats.txt` for the achieved zero-shot percentage.

Setting `percent_zeroshot` to `0.0` keeps the training set unchanged while
still creating the normalized `zeroshot/` directory.

## 2A. Generate scikit-learn TF-IDF SVM files

Run this after creating `DATASET/zeroshot/`:

```bash
python /nfs-stor/linh.vu/dataprep_files/3_1.text_to_svm.py \
  --config /nfs-stor/linh.vu/dataprep_files/config_amazon131k.json
```

The script always reads `trn.txt`, `tst.txt`, and `Y.txt` from
`DATASET/zeroshot/`. It fits TF-IDF on the training documents plus label
descriptions, transforms the test documents, and writes:

```text
DATASET/zeroshot/label_mapping.txt
DATASET/zeroshot/<DATASET_NAME>_tfidf_train.svm
DATASET/zeroshot/<DATASET_NAME>_tfidf_test.svm
DATASET/zeroshot/<DATASET_NAME>_tfidf_lf.svm
```

This utility depends on NumPy, SciPy, scikit-learn, Joblib, and
LibMultiLabel. The LibMultiLabel path is currently hard-coded as
`/home/linh.vu/LibMultiLabel`; update the script or environment if the library
is installed elsewhere.

## 2B. Generate PECOS TF-IDF SVM files

`3_3.text_to_svm_pecos.py` reads the files specified directly in its dataset
config. To process the zero-shot split, create a config whose input paths point
to `zeroshot/`:

```json
{
  "dataset_path": "/path/to/DATASET",
  "train_file": "/path/to/DATASET/zeroshot/trn.txt",
  "test_file": "/path/to/DATASET/zeroshot/tst.txt",
  "label_description_file": "/path/to/DATASET/zeroshot/Y.txt",
  "MAX_FEATURES": -1,
  "n_labels": 131073,
  "percent_zeroshot": 0.01
}
```

PECOS also requires a vectorizer config. By default, the script reads
`DATASET/config.json`. A minimal example is:

```json
{
  "type": "tfidf",
  "kwargs": {}
}
```

Run the complete PECOS pipeline with:

```bash
conda activate zeroshot

python /nfs-stor/linh.vu/dataprep_files/3_3.text_to_svm_pecos.py \
  --config /path/to/config_dataset_zeroshot.json \
  --output-dir /path/to/DATASET/zeroshot
```

Use a different PECOS vectorizer config when needed:

```bash
python /nfs-stor/linh.vu/dataprep_files/3_3.text_to_svm_pecos.py \
  --config /path/to/config_dataset_zeroshot.json \
  --vectorizer-config /path/to/pecos_tfidf_config.json \
  --output-dir /path/to/DATASET/zeroshot
```

Main generated files are:

```text
<prefix>_tfidf_train_ext_pecos.svm
<prefix>_tfidf_test_ext_pecos.svm
<prefix>_tfidf_lf_pecos.svm
label_mapping.txt
tfidf-model/
*.tfidf.npz
```

The default prefix is the lower-case dataset directory name with punctuation
removed. For example, `EURLEX-4K` becomes `eurlex4k`.

Useful reuse options:

```bash
# Reuse an existing PECOS model, but regenerate all three matrices.
python /nfs-stor/linh.vu/dataprep_files/3_3.text_to_svm_pecos.py \
  --config /path/to/config.json --reuse-model

# Reuse existing *.tfidf.npz matrices and only regenerate SVM files.
python /nfs-stor/linh.vu/dataprep_files/3_3.text_to_svm_pecos.py \
  --config /path/to/config.json --reuse-tfidf
```

The reuse commands expect the model or matrices to have the same paths,
prefix, label order, and vectorizer configuration as the current run.

## 2C. Generate DEXML data

DEXML preparation has two stages.

### Convert text and labels

```bash
python /nfs-stor/linh.vu/dataprep_files/2.create_svm_text_to_dexml.py \
  --config /nfs-stor/linh.vu/dataprep_files/config_amazon131k.json
```

This creates:

```text
DATASET/dexml/
├── Y.trn.npz
├── Y.tst.npz
└── raw/
    ├── trn_X.txt
    ├── tst_X.txt
    └── Y.txt
```

If `validation_file` is configured, the script also creates `raw/val_X.txt`
and prepares its labels in memory. The current implementation does not save a
`Y.val.npz` matrix.

Path warning: this utility obtains the zero-shot input paths by replacing the
word `original` with `zeroshot` in the configured paths. Therefore, the
recommended config paths must contain an `original/` component. If they do
not—as in the current EURLEX-4K config—the converter reads the configured
files directly rather than `DATASET/zeroshot/`.

### Generate label embeddings and tokenized files

After the first DEXML stage, run:

```bash
python /nfs-stor/linh.vu/dataprep_files/3_2.text_to_dexml.py \
  --config /nfs-stor/linh.vu/dataprep_files/config_amazon131k.json
```

This stage:

- encodes label descriptions with `Salesforce/SFR-Embedding-2_R`;
- saves `DATASET/dexml/lbl_embs_sfr.npy`;
- tokenizes `trn_X.txt`, `tst_X.txt`, and `Y.txt` with
  `bert-base-uncased`;
- uses a maximum length of 32 for dataset names containing `Title`, otherwise
  128;
- creates `.dat` and `.dat.meta` files under `DATASET/dexml/raw/`.

A CUDA GPU is recommended for label embedding. This stage requires PyTorch,
Transformers, Sentence Transformers, NumPy, access to the model files, and the
external tokenizer script currently hard-coded at
`/home/linh.vu/DEXML/utils/tokenization_utils.py`.

The provided `run` file is a Slurm example for this stage. Its cluster
resources, dataset config, and paths are hard-coded and should be edited before
submission:

```bash
sbatch /nfs-stor/linh.vu/dataprep_files/run
```

## 3. Compute statistics and verify outputs

```bash
python /nfs-stor/linh.vu/dataprep_files/4.get_data_stats.py \
  --dataset_path /path/to/DATASET/zeroshot \
  --original_train_file /path/to/DATASET/original/trn.txt
```

The command reports:

- train and test sizes;
- average labels per sample;
- labels present in test but absent from training;
- achieved zero-shot percentage;
- missing labels relative to the original train/test universe;
- approximate shapes of all `.svm` files in the target directory.

Results are printed and saved to `DATASET/zeroshot/stats.txt`.

## Utility status

| File | Purpose | Recommended usage |
| --- | --- | --- |
| `1.create_zeroshot.py` | Create the filtered zero-shot split. | Main pipeline utility. |
| `2.create_svm_text_to_dexml.py` | Convert text and labels to the DEXML directory layout. | Main DEXML utility. |
| `3_1.text_to_svm.py` | Generate scikit-learn TF-IDF SVM files. | Main SVM utility. |
| `3_2.text_to_dexml.py` | Generate DEXML label embeddings and tokenized files. | Main DEXML utility; environment-specific. |
| `3_3.text_to_svm_pecos.py` | Generate TF-IDF matrices and SVM files with PECOS. | Main PECOS utility. |
| `4.get_data_stats.py` | Report zero-shot and SVM statistics. | Main validation utility. |
| `run` | Slurm submission example for DEXML generation. | Edit before use. |
| `dataproc.py` | Earlier hard-coded TF-IDF experiment. | Reference only. |
| `index_checking.py` | Checks sequential indices for one hard-coded dataset. | Update its path before use. |
| `total_label_check.py` | Checks label counts for one hard-coded dataset. | Update its paths before use. |
| `gen_sfr.ipynb`, `test.ipynb` | Exploratory notebooks. | Reference only. |

## Common checks

Before starting a large run, verify:

1. All paths in the selected dataset config exist.
2. The label IDs are integers in the range `0` to `n_labels - 1`.
3. `Y.txt` has exactly `n_labels` lines.
4. The generated `zeroshot/stats.txt` reports zero missing labels.
5. The train, test, and label-feature SVM files use the same feature dimension.
6. Sufficient disk space is available; PECOS text inputs, sparse matrices,
   model files, and SVM outputs can be large.
