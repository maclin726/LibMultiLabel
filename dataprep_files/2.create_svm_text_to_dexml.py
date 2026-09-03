# requirements of DEXML format:
# a raw folder containing:
    # trn_X.txt: each line is a sample text only
    # tst_X.txt
    # val_X.txt (Optional)
    # Y.txt, label desciptions, each line is a label description only
# Y.trn.npz, Y.tst.npz, Y.val.npz (Optional), each is a sparse matrix of shape (n_samples, n_labels)

# Input is svm-text format, each line is: id \t label1 label2 ... \t text
import os
import shutil
import random
import argparse
import json
import numpy as np
import scipy.sparse as sp

random.seed(42)
parser = argparse.ArgumentParser(description='Convert SVM-Text format to DEXML format')
parser.add_argument('--config', '-c', type=str, required=True, help='Path to the config file')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = json.load(f)
    
dataset_path = config['dataset_path']
train_file = config['train_file']
test_file = config['test_file']
label_description_file = config['label_description_file']
percent_zeroshot = config['percent_zeroshot']
n_labels = config['n_labels']

# make sure the dataset_path matches the paths in train_file, test_file, label_description_file
assert dataset_path in train_file
assert dataset_path in test_file
assert dataset_path in label_description_file

train_file = train_file.replace('original', 'zeroshot')
test_file = test_file.replace('original', 'zeroshot')
label_description_file = label_description_file.replace('original', 'zeroshot')

# in dataset folder, create a folder named dexml
dexml_folder = os.path.join(dataset_path, 'dexml')
if not os.path.exists(dexml_folder):
    os.makedirs(dexml_folder)
# in dexml folder, create a folder named raw
raw_folder = os.path.join(dexml_folder, 'raw')
if not os.path.exists(raw_folder):
    os.makedirs(raw_folder)
# copy label description file to raw folder and rename to Y.txt
shutil.copy(label_description_file, os.path.join(raw_folder, 'Y.txt'))

n_train_lines = 0
n_test_lines = 0

# read train file and write to raw/trn_X.txt
print("Reading labels from train and test files...")
train_labels = []
with open(train_file, 'r') as f:
    lines = f.readlines()
    n_train_lines = len(lines)
    with open(os.path.join(raw_folder, 'trn_X.txt'), 'w') as f_out:
        for line in lines:
            parts = line.strip().split('\t')
            text = parts[2]
            labels = parts[1].split()
            train_labels.append(labels)
            f_out.write(text + '\n')
# read test file and write to raw/tst_X.txt
test_labels = []
with open(test_file, 'r') as f:
    lines = f.readlines()
    n_test_lines = len(lines)
    with open(os.path.join(raw_folder, 'tst_X.txt'), 'w') as f_out:
        for line in lines:
            parts = line.strip().split('\t')
            text = parts[2]
            labels = parts[1].split()
            test_labels.append(labels)
            f_out.write(text + '\n')
# read validation file and write to raw/val_X.txt (if it exists)
if 'validation_file' in config:
    val_labels = []
    with open(config['validation_file'], 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            text = parts[2]
            labels = parts[1].split()
            val_labels.append(labels)
        with open(os.path.join(raw_folder, 'val_X.txt'), 'w') as f_out:
            for line in lines:
                parts = line.strip().split('\t')
                text = parts[2]
                f_out.write(text + '\n')
# create Y.trn.npz, Y.tst.npz, Y.val.npz (if validation file exists)
# each is a sparse matrix of shape (n_samples, n_labels)
# we can use scipy.sparse to create sparse matrices

print("Reading completed.")   




def build_sparse_label_matrix(label_sequences, n_labels, dtype=np.int8):
    nnz = sum(len(labels) for labels in label_sequences)

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)

    pos = 0
    for i, labels in enumerate(label_sequences):
        if not labels:
            continue

        labels = np.asarray(labels, dtype=np.int64)

        if np.any(labels < 0) or np.any(labels >= n_labels):
            raise ValueError(
                f"Sample {i} contains invalid labels: {labels.tolist()} "
                f"(valid range: 0 to {n_labels - 1})"
            )

        k = len(labels)
        rows[pos:pos + k] = i
        cols[pos:pos + k] = labels
        pos += k

    data = np.ones(nnz, dtype=dtype)

    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(label_sequences), n_labels),
        dtype=dtype,
    )



print("Processing labels and saving sparse matrices...")
Y_trn = build_sparse_label_matrix(train_labels, n_labels)
print("Y_trn shape:", Y_trn.shape)
Y_tst = build_sparse_label_matrix(test_labels, n_labels)
print("Y_tst shape:", Y_tst.shape)

print("\nSaving sparse matrices to disk...")

sp.save_npz(f"{dexml_folder}/Y.trn.npz", Y_trn)
sp.save_npz(f"{dexml_folder}/Y.tst.npz", Y_tst)

# # get number of samples and labels
# Y_trn = np.zeros((len(train_labels), n_labels), dtype=np.int8)
# label_list = set()
# print(train_labels[0:2])
# for i, labels in enumerate(train_labels):
    # convert to int
    # labels = [int(label) for label in labels]
    # print(labels)
    # break
    # Y_trn[i, labels] = 1
    # print(Y_trn[i, labels])
    
# Y_tst = np.zeros((len(test_labels), n_labels), dtype=np.int8)
# for i, labels in enumerate(test_labels):
#     # convert to int
#     labels = [int(label) for label in labels]
#     Y_tst[i, labels] = 1
    
    
# import scipy.sparse as sp
# Y_trn = sp.csr_matrix(Y_trn)
# Y_tst = sp.csr_matrix(Y_tst)
# sp.save_npz('Y.trn.npz', Y_trn)
# sp.save_npz('Y.tst.npz', Y_tst)