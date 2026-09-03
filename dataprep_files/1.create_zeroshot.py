import json
import os
import random
import shutil
import argparse
from collections import Counter

random.seed(30)

parser = argparse.ArgumentParser(description='Create zero-shot dataset')
parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)

dataset_path = config['dataset_path']
train_file = config['train_file']
test_file = config['test_file']
label_description_file = config['label_description_file']
percent_zeroshot = config['percent_zeroshot']

os.chdir(dataset_path)

print(f"Dataset path: {dataset_path}")
print(f"Targeted zero-shot percentage: {percent_zeroshot * 100:.2f}%")

if not os.path.exists('zeroshot'):
    os.makedirs('zeroshot')

# -----------------------------
# Load train
# -----------------------------
train_samples = []
idx = 0  # manual index to ensure no gaps after filtering
c = 0
with open(train_file, 'r') as f:
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 3 or not parts[2].strip():
            c+=1
            continue
        _, label_str, text = parts[0], parts[1], parts[2]
        labels = label_str.split()
        train_samples.append((idx, labels, text))
        idx += 1
# -----------------------------
# Load test
# -----------------------------
test_samples = []
idx = 0  # manual index
with open(test_file, 'r') as f:
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 3 or not parts[2].strip():
            c += 1
            continue
        _, label_str, text = parts[0], parts[1], parts[2]
        labels = label_str.split()
        test_samples.append((idx, labels, text))
        idx += 1
        
# write test_samples to zeroshot/tst.txt
with open(os.path.join('zeroshot', 'tst.txt'), 'w') as f:
    for idx, labels, text in test_samples:
        f.write(f"{idx}\t{' '.join(labels)}\t{text}\n")
        
# labels appearing in test
test_label_set = set()
for _, labels, _ in test_samples:
    test_label_set.update(labels)

# full label universe from original train + test
full_label_set = set()
for _, labels, _ in train_samples:
    full_label_set.update(labels)
for _, labels, _ in test_samples:
    full_label_set.update(labels)

print(f"Original total unique labels: {len(full_label_set)}")

# frequency of labels in original train
train_label_freq = Counter()
for _, labels, _ in train_samples:
    train_label_freq.update(labels)

# Only labels that appear in test can be zero-shot without disappearing from the task definition
candidate_zeroshot_labels = list(test_label_set)
num_zeroshot = int(percent_zeroshot * len(candidate_zeroshot_labels))
zeroshot_labels = set(random.sample(candidate_zeroshot_labels, num_zeroshot))

print(f"Chosen zero-shot labels: {len(zeroshot_labels)}")

# Greedy safe filtering:
# remove a train sample only if:
# 1. it contains at least one zero-shot label
# 2. removing it will NOT make any non-test label disappear entirely
filtered_train_samples = []
current_freq = train_label_freq.copy()

for idx, labels, text in train_samples:
    labels_set = set(labels)

    if not (labels_set & zeroshot_labels):
        filtered_train_samples.append((idx, labels, text))
        continue

    safe_to_remove = True
    for lbl in labels:
        # zero-shot labels are supposed to be removed from train
        if lbl in zeroshot_labels:
            continue

        # if lbl is not in test, it must remain somewhere in filtered train
        if lbl not in test_label_set and current_freq[lbl] <= 1:
            safe_to_remove = False
            break

    if safe_to_remove:
        for lbl in labels:
            current_freq[lbl] -= 1
    else:
        filtered_train_samples.append((idx, labels, text))

# reindex filtered train
if not os.path.exists('zeroshot'):
    os.makedirs('zeroshot')


# shutil.copy(test_file, 'zeroshot/')
shutil.copy(label_description_file, 'zeroshot/')

with open(os.path.join('zeroshot', 'trn.txt'), 'w') as f:
    for new_idx, (_, labels, text) in enumerate(filtered_train_samples):
        f.write(f"{new_idx}\t{' '.join(labels)}\t{text}\n")

# final stats
filtered_train_label_set = set()
for _, labels, _ in filtered_train_samples:
    filtered_train_label_set.update(labels)

final_union = filtered_train_label_set | test_label_set

lines = [
    f"Original train size: {len(train_samples)}",
    f"Filtered train size: {len(filtered_train_samples)}",
    f"Unique labels in filtered train: {len(filtered_train_label_set)}",
    f"Unique labels in test but not in train: {len(test_label_set - filtered_train_label_set)}",
    f"Final total unique labels: {len(final_union)}",
    f"Final zero-shot percentage in test: {len(test_label_set - filtered_train_label_set) / len(test_label_set) * 100:.2f}%",
    f"Final zero-shot percentage in full train-test label universe: {len(test_label_set - filtered_train_label_set) / len(final_union) * 100:.2f}%",
    f"Missing labels from original universe: {len(full_label_set - final_union)}"
]

# print to console
for line in lines:
    print(line)

# write to file
with open(f"zeroshot/stats.txt", "w") as f:
    f.write("\n".join(lines))
