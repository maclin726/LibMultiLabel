import sys
from pathlib import Path
import numpy as np
import argparse
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

sys.path.insert(0, '/home/linh.vu/LibMultiLabel')
import libmultilabel.linear as linear
from pathlib import Path



parser = argparse.ArgumentParser(description='Create SVM-Text format dataset')
# create -c or --config argument to specify the config file
parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)
dataset_path = config['dataset_path']
train_file = config['train_file']
test_file = config['test_file']
label_description_file = config['label_description_file']
percent_zeroshot = config['percent_zeroshot']
MAX_FEATURES = config['MAX_FEATURES']
# change path to dataset folder, zeroshot
to_directory = os.path.join(dataset_path, 'zeroshot')
os.chdir(to_directory)
# the current directory is in zeroshot folder

DATANAME = dataset_path.split('/')[-1]


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
    return test_set - train_set

train_texts, train_labels = load_data("trn.txt")
test_texts, test_labels = load_data("tst.txt")

print("Average label length in training set:", compute_avg_label_length(train_labels))
print("Average label length in test set:", compute_avg_label_length(test_labels))
print("Number of Zeroshot labels:", len(get_different_labels(train_labels, test_labels)))

datasets = linear.load_dataset(data_format="txt",
                               train_path="trn.txt",
                               test_path="tst.txt")

preprocessor = linear.Preprocessor(include_test_labels=True)
datasets = preprocessor.fit_transform(datasets)

np.count_nonzero(datasets['train']['y'].sum(axis=0).A.flatten())

with open("label_mapping.txt", "w") as f:
    for i, label in enumerate(preprocessor.label_mapping):
        print(f'{label}', file=f)
        
with open('label_mapping.txt', 'r') as f:
    labels = list(map(str.strip, f.readlines()))        

labels2id = dict()
for i, label in enumerate(labels):
    labels2id[label] = i
label_dict = {}
with open('Y.txt', 'r', encoding='latin-1') as f:
    for i, line in enumerate(f):
        label_dict[str(i)] = line.strip()
label_desc = []
for key, val in labels2id.items():
    # key is the
    label_desc.append(label_dict[key]) 
    
DATA_DIR = Path('.')
RAW_TEXT_FILE = '{}.txt'
raw_text = {'tst' : list(), 'trn' : list()}
label = {'tst' : list(), 'trn' : list()}
for partition in ['tst','trn']:
    RAW_TEXT_PATH = DATA_DIR / RAW_TEXT_FILE.format(partition)
    with open(str(RAW_TEXT_PATH.resolve()), 'r', encoding='utf-8') as file:
        label[partition], raw_text[partition] = \
            zip(*map(lambda x:x.split('\t')[1:], file.readlines()))
            
print(
    "Number of samples:", len(raw_text['trn']),'\n', 
    "Number of test samples:", len(raw_text['tst']),'\n',
    "Number of train labels:", len(label['trn']),'\n',
    "Number of test labels:", len(label['tst']))

import re
def custom_tokenizer(text):
    return re.findall(r"[A-Za-z0-9&#!?+.-]+", text)


if (DATA_DIR / "tfidf_vectorizer.pkl").exists():
    vectorizer = joblib.load(DATA_DIR / "tfidf_vectorizer.pkl")
else:
    if MAX_FEATURES > 0:
        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, dtype=np.float32)
    else:
        vectorizer = TfidfVectorizer(dtype=np.float32, tokenizer=custom_tokenizer)
# check if there's a vectorizer already exists

tfidf = vectorizer.fit_transform(list(raw_text['trn']) + label_desc)
x_train = tfidf[:len(raw_text['trn'])]
label_tfidf = tfidf[len(raw_text['trn']):]
x_test = vectorizer.transform(raw_text['tst'])

# print feature size
print("Train shape: ",x_train.shape)
print("Test shape: ",x_test.shape)
print("Label description shape: ", label_tfidf.shape)

c = 0
for desc, vec in zip(label_desc, label_tfidf):
    if not np.any(vec.toarray()):
        c += 1
print("Number of zero vectors:", c) 

with open(DATA_DIR / f'{DATANAME}_tfidf_train.svm', 'w') as f:
    for labels,feature,index in zip(
        label['trn'], x_train.tolil().data, x_train.tolil().rows):
        labels = labels.split(' ')
        label_str = ','.join([str(labels2id[l]) for l in labels])
        feature_str = ' '.join([f'{x+1}:{y}' for x,y in zip(index,feature)])
        f.write(label_str + ' ' + feature_str + '\n')

with open(DATA_DIR / f'{DATANAME}_tfidf_test.svm', 'w') as f:
    for labels,feature,index in zip(label['tst'], x_test.tolil().data, x_test.tolil().rows):
        labels = labels.split(' ')
        label_str = ','.join([str(labels2id[l]) for l in labels])
        feature_str = ' '.join([f'{x+1}:{y}' for x,y in zip(index,feature)])
        f.write(label_str + ' ' + feature_str + '\n')
        
with open(DATA_DIR / f'{DATANAME}_tfidf_lf.svm', 'w') as f:
    for tfidf in label_tfidf:
        features = tfidf.tolil().data[0]
        indices = tfidf.tolil().rows[0]
        feature_str = ' '.join([f'{x+1}:{y}' for x,y in zip(indices,features)])
        print(f'\t{feature_str}', file=f)
        
# with open(DATA_DIR / f'{DATANAME}_tfidf_lf.svm', 'w') as f:
#     for row in label_tfidf:
#         row = row.tocsr()
#         indices = row.indices
#         features = row.data

#         if len(indices) == 0:
#             # use one extra dummy feature column
#             feature_str = f'{MAX_FEATURES+1}:1'
#         else:
#             feature_str = ' '.join(f'{x+1}:{y}' for x, y in zip(indices, features))

#         print(feature_str, file=f)