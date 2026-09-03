import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from pathlib import Path
import sys
sys.path.insert(0, '/home/lvu5/LibMultiLabel')
import libmultilabel.linear as linear
print("testing")

DIR = "/home/lvu5/LibMultiLabel/data/LF-AmazonTitles-131K"

os.chdir(DIR)

file_name = "LF-AmazonTitles-131K"

print("Loading dataset...")
datasets = linear.load_dataset(data_format="txt",
                               train_path="trn.txt",
                               test_path="tst.txt")#,
                            #    label_path="Y.txt")


preprocessor = linear.Preprocessor(include_test_labels=True)
datasets = preprocessor.fit_transform(datasets)
print("dataset loaded")
np.count_nonzero(datasets['train']['y'].sum(axis=0).A.flatten())

with open("label_mapping.txt", "w") as f:
    for i, label in enumerate(preprocessor.label_mapping):
        print(f'{label}', file=f)

with open('label_mapping.txt', 'r') as f:
    labels = list(map(str.strip, f.readlines()))        

labels2id = dict()
for i, label in enumerate(labels):
    labels2id[label] = i
print("length of labels:", len(labels2id))
label_dict = {}
with open('Y.txt', 'r', encoding='latin-1') as f:
    for i, line in enumerate(f):
        label_dict[str(i)] = line.strip()
# print first 5 label_dict
print(list(label_dict.items())[:5])
label_desc = []
for key, val in labels2id.items():
    # key is the
    label_desc.append(label_dict[key]) 
print(label_desc[:5])


DATA_DIR = Path('.')
RAW_TEXT_FILE = '{}.txt'
raw_text = {'tst' : list(), 'trn' : list()}
label = {'tst' : list(), 'trn' : list()}
for partition in ['tst','trn']:
    RAW_TEXT_PATH = DATA_DIR / RAW_TEXT_FILE.format(partition)
    with open(str(RAW_TEXT_PATH.resolve()), 'r', encoding='utf-8') as file:
        label[partition], raw_text[partition] = \
            zip(*map(lambda x:x.split('\t')[1:], file.readlines()))

print(len(raw_text['trn']), len(raw_text['tst']), len(label['trn']), len(label['tst']))

vectorizer = TfidfVectorizer(dtype=np.float32)
tfidf = vectorizer.fit_transform(list(raw_text['trn']) + label_desc)
x_train = tfidf[:len(raw_text['trn'])]
label_tfidf = tfidf[len(raw_text['trn']):]
x_test = vectorizer.transform(raw_text['tst'])
print("TF-IDF matrix shape:", x_train.shape, x_test.shape, label_tfidf.shape)



empty_labels = [desc for desc, vec in zip(label_desc, label_tfidf) if not np.any(vec.toarray())]
print(empty_labels)

with open(DATA_DIR / f'{file_name}_tfidf_train_ext.svm', 'w') as f:
    for labels,feature,index in zip(
        label['trn'], x_train.tolil().data, x_train.tolil().rows):
        labels = labels.split(' ')
        label_str = ','.join([str(labels2id[l]) for l in labels])
        feature_str = ' '.join([f'{x+1}:{y}' for x,y in zip(index,feature)])
        f.write(label_str + ' ' + feature_str + '\n')

with open(DATA_DIR / f'{file_name}_tfidf_test_ext.svm', 'w') as f:
    for labels,feature,index in zip(label['tst'], x_test.tolil().data, x_test.tolil().rows):
        labels = labels.split(' ')
        label_str = ','.join([str(labels2id[l]) for l in labels])
        feature_str = ' '.join([f'{x+1}:{y}' for x,y in zip(index,feature)])
        f.write(label_str + ' ' + feature_str + '\n')

with open(DATA_DIR / f'{file_name}_tfidf_lf.svm', 'w') as f:
    for tfidf in label_tfidf:
        features = tfidf.tolil().data[0]
        indices = tfidf.tolil().rows[0]
        feature_str = ' '.join([f'{x+1}:{y}' for x,y in zip(indices,features)])
        print(f'\t{feature_str}', file=f)