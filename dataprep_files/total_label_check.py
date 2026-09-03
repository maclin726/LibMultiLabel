def load_data(file_path):
    labels = []
    texts = []
    
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                label = parts[1]
                text = parts[2]
                texts.append(text)
                labels.append(label.split())
    
    return texts, labels

def compute_avg_label_length(labels):
    total = 0
    for label in labels:
        total += len(label)    
    if len(labels) == 0:
        return 0
    return total / len(labels)

def get_different_labels(train_labels, test_labels):
    train_labels = [item for sublist in train_labels for item in sublist]
    test_labels = [item for sublist in test_labels for item in sublist]
    train_set = set(train_labels)
    test_set = set(test_labels)
    print("Total labels = ", len(test_set.union(train_set)))
    print("Different labels = ", len(test_set - train_set))

# train_texts, train_labels = load_data("/home/lvu5/LibMultiLabel/data/AmazonCat-13K/zs/trn_original.txt")
# test_texts, test_labels = load_data("/home/lvu5/LibMultiLabel/data/AmazonCat-13K/zs/tst.txt")

train_texts, train_labels = load_data("/home/lvu5/LibMultiLabel/data/LF-AmazonTitles-131K/zeroshot/trn.txt")
test_texts, test_labels = load_data("/home/lvu5/LibMultiLabel/data/LF-AmazonTitles-131K/zeroshot/tst.txt")

get_different_labels(train_labels, test_labels)