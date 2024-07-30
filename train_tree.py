from tqdm import tqdm
import math
import libmultilabel.linear as linear
import time
import pickle
import numpy as np
import os
import random

dataset = 'amazon-670k'

with open(f"data/{dataset}/dataset.pkl", "rb") as f:
    datasets = pickle.load(f)
    print("load pickle succeed")

def metrics_in_batches(model):
    batch_size = 256
    num_instances = datasets["test"]["x"].shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = linear.get_metrics(["P@1", "P@3", "P@5"], num_classes=datasets["test"]["y"].shape[1])

    for i in tqdm(range(num_batches), ncols=79):
        preds = linear.predict_values(model, datasets["test"]["x"][i * batch_size : (i + 1) * batch_size])
        target = datasets["test"]["y"][i * batch_size : (i + 1) * batch_size].toarray()
        metrics.update(preds, target)

    return metrics.compute()


path = 'label_tree_obj/default/'
thresholds = [0.1,]
results = dict()
for thres in thresholds:
    results[thres] = []

tree_names = [
    f'{dataset}_elkan_d5_K100_seed0', 
    f'{dataset}_elkan_d5_K100_seed1', 
    f'{dataset}_elkan_d5_K100_seed2'
]

for tree_name in tree_names:
    if not os.path.isfile(f"model/{tree_name}_s2_unpruned.pkl"):
        tree_model = linear.train_tree(
            datasets["train"]["y"], datasets["train"]["x"], 
            options="-s 2 -m 28", path=path+tree_name+'.pkl')
        with open(f'model/{tree_name}_s2_unpruned.pkl', "wb") as f:
            pickle.dump(tree_model, f, protocol=5)
    else:
        with open(f'model/{tree_name}_s2_unpruned.pkl', "rb") as f:
            tree_model = pickle.load(f)
        # print(f'successfully load tree model {}')

    unpruned = tree_model
    unpruned_weights = unpruned.flat_model.weights
    print(f"shape: {unpruned_weights.shape}, sparsity: {unpruned_weights.nnz/(unpruned_weights.shape[0]*unpruned_weights.shape[1])}")
    # print("Before pruning:", metrics_in_batches(unpruned))
    
    nnz = unpruned_weights.nnz
    
    for thres in thresholds:
        n_pruned = np.count_nonzero(np.abs(unpruned_weights.data) < thres)
        print("Pruning percentage:", n_pruned/nnz)
        unpruned.flat_model.weights.data[np.abs(unpruned_weights.data) < thres] = 0
        unpruned.flat_model.weights.eliminate_zeros()
        pruned_weights = unpruned.flat_model.weights
        pruned_nnz = unpruned.flat_model.weights.nnz
        print(f"shape: {pruned_weights.shape}, sparsity: {pruned_nnz/(pruned_weights.shape[0]*pruned_weights.shape[1])}")
        
        results[thres].append(metrics_in_batches(unpruned))

for thres in thresholds:
    metrics = ['P@1', 'P@3', 'P@5']
    tmp = [
        [scores[metric] for scores in results[thres]] for metric in metrics]
    print(f"The average with threshold {thres}", np.average(tmp, axis=1))