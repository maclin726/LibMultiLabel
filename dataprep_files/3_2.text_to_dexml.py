import os

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import argparse
import json
import sys
# sys.path.append("/home/lvu5/LibMultiLabel/ZSLWAN-release")
import subprocess

model_name = "Salesforce/SFR-Embedding-2_R"

model = SentenceTransformer(model_name, model_kwargs={'dtype': torch.bfloat16})

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

dataset_name = dataset_path.split('/')[-1]
# os.chdir to dataset_path/dexml/raw
os.chdir(os.path.join(dataset_path, 'dexml', 'raw'))
# print current working directory
print("Current working directory:", os.getcwd())

label_path= 'Y.txt'
with open(label_path, 'r', encoding='latin-1') as f:
    label_descs = f.readlines() # remove the newline characters
label_descs = [desc.strip() for desc in label_descs]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Encoding label embeddings...")
label_embs = model.encode(label_descs, normalize_embeddings=True,
                          convert_to_tensor=True, device=DEVICE)
print("Encoding FINISHED!")
# torch.save(label_embs, 'lbl_embs_sfr.pt')
label_embs_np = (
    label_embs
    .detach()
    .to(torch.float32)   # convert from bfloat16
    .cpu()
    .numpy()
)
# save to dataset_path, 'dexml' folder as lbl_embs_sfr.npy
print("Saving label embeddings to numpy file...")
np.save(os.path.join(dataset_path, 'dexml', 'lbl_embs_sfr.npy'), label_embs_np)
# np.save("lbl_embs_sfr.npy", label_embs_np)
print("Label embeddings saved to numpy file!")



tokenizer_script = "/home/linh.vu/DEXML/utils/tokenization_utils.py"
# set tf_max_len to 32 if there are the word Title in the dataset name, otherwise set to 128
if "Title" in dataset_name:
    tf_max_len = 32
else:
    tf_max_len = 128
commands = [
    ["python", tokenizer_script,
     "--data-path", "trn_X.txt",
     "--tf-max-len", str(tf_max_len),
     "--tf-token-type", "bert-base-uncased"],

    ["python", tokenizer_script,
     "--data-path", "tst_X.txt",
     "--tf-max-len", str(tf_max_len),
     "--tf-token-type", "bert-base-uncased"],

    ["python", tokenizer_script,
     "--data-path", "Y.txt",
     "--tf-max-len", str(tf_max_len),
     "--tf-token-type", "bert-base-uncased"]
]

for i, cmd in enumerate(commands):
    print(f"Running command {i+1}/{len(commands)}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
print("Tokenization finished!")