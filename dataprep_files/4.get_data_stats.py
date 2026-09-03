import os
import argparse

def load_samples(file_path):
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3 or not parts[2].strip():
                continue
            idx, label_str, text = parts[0], parts[1], parts[2]
            labels = label_str.split()
            samples.append((int(idx), labels, text))
    return samples


def get_svm_shape(file_path):
    n_samples = 0
    max_index = 0

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            n_samples += 1

            for item in parts[1:]:
                if ':' not in item:
                    continue
                idx, _ = item.split(':', 1)
                idx = int(idx)
                if idx > max_index:
                    max_index = idx

    return n_samples, max_index


def average_labels_per_sample(samples):
    if len(samples) == 0:
        return 0.0
    total_labels = sum(len(labels) for _, labels, _ in samples)
    return total_labels / len(samples)


def main():
    parser = argparse.ArgumentParser(description="Compute zero-shot dataset statistics")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to zeroshot folder')
    parser.add_argument('--original_train_file', type=str, default=None,
                        help='Optional original train file to compute full label universe')
    args = parser.parse_args()

    trn_path = os.path.join(args.dataset_path, 'trn.txt')
    tst_path = os.path.join(args.dataset_path, 'tst.txt')

    # Load filtered train and test
    train_samples = load_samples(trn_path)
    test_samples = load_samples(tst_path)

    # Label sets
    filtered_train_label_set = set()
    for _, labels, _ in train_samples:
        filtered_train_label_set.update(labels)

    test_label_set = set()
    for _, labels, _ in test_samples:
        test_label_set.update(labels)

    final_union = filtered_train_label_set | test_label_set

    # Full label universe (optional)
    if args.original_train_file:
        full_label_set = set()

        original_train_samples = load_samples(args.original_train_file)
        for _, labels, _ in original_train_samples:
            full_label_set.update(labels)
        for _, labels, _ in test_samples:
            full_label_set.update(labels)
    else:
        full_label_set = final_union

    # Average labels
    avg_labels_train = average_labels_per_sample(train_samples)
    avg_labels_test = average_labels_per_sample(test_samples)

    # Stats
    lines = [
        f"Filtered train size: {len(train_samples)}",
        f"Test size: {len(test_samples)}",
        f"Average labels per train sample: {avg_labels_train:.4f}",
        f"Average labels per test sample: {avg_labels_test:.4f}",
        f"Unique labels in filtered train: {len(filtered_train_label_set)}",
        f"Unique labels in test but not in train: {len(test_label_set - filtered_train_label_set)}",
        f"Final total unique labels: {len(final_union)}",
        f"Final zero-shot percentage in test: "
        f"{len(test_label_set - filtered_train_label_set) / len(test_label_set) * 100:.2f}%",
        f"Missing labels from original universe: {len(full_label_set - final_union)}"
    ]

    # Print text-file stats
    for line in lines:
        print(line)

    # Add .svm file stats
    svm_lines = []
    print("\n--- SVM File Shapes ---")
    svm_lines.append("\n--- SVM File Shapes ---")

    for fname in sorted(os.listdir(args.dataset_path)):
        if fname.endswith(".svm"):
            fpath = os.path.join(args.dataset_path, fname)
            n_samples, n_features = get_svm_shape(fpath)

            print(f"{fname}:")
            print(f"  Samples: {n_samples}")
            print(f"  Features: {n_features}")
            print(f"  Shape: ({n_samples}, {n_features})")

            svm_lines.append(f"{fname}:")
            svm_lines.append(f"  Samples: {n_samples}")
            svm_lines.append(f"  Features: {n_features}")
            svm_lines.append(f"  Shape: ({n_samples}, {n_features})")

    # Save
    output_path = os.path.join(args.dataset_path, "stats.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(lines + svm_lines))


if __name__ == "__main__":
    main()