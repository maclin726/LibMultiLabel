import json
import math
import os
import pickle
import random
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

import libmultilabel.linear as linear


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

if __name__  == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("mode",
    #                     choices=['all', 'build_tree', 'load_tree', 'clip_tree'],
    #                     default='all')
    # parser.add_argument("format", help="format")
    parser.add_argument("dataset", help="data set")
    parser.add_argument("tree_root_dir",
                        help="the directory to store label trees")
    parser.add_argument("--n_cores", type=int, default=8)
    # parser.add_argument("--K", type=int, default=100)
    # parser.add_argument("--dmax", type=int, default=10)
    # parser.add_argument("--cluster", default="elkan",
    #                     choices=["elkan", "balanced_spherical", "random"])
    # parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(f"data/{args.dataset}/dataset.pkl", "rb") as f:
        datasets = pickle.load(f)
        print("load pickle succeed")


    types = [1, 2, 3]

    type2name = {0: "LR primal", 
                1: "L2   dual",
                2: "L2 primal",
                3: "L1   dual",
                7: "LR   dual"}

    results = {
        0: [],
        1: [],
        2: [],
        3: [],
        7: []
    }

    tree_names = [
        f'{args.tree_root_dir}/{args.dataset}_elkan_d10_K100_seed0.pkl', 
        # f'{args.tree_root_dir}/{args.dataset}_elkan_d10_K100_seed1.pkl', 
        # f'{args.tree_root_dir}/{args.dataset}_elkan_d10_K100_seed2.pkl'
    ]

    for tree_name in tree_names:
        for type in types:
            options = f"-s {type} -m 1" if type != 7 else f"-s {type}"
            tree_model, train_time, model_nnz = linear.train_tree(
                datasets["train"]["y"], datasets["train"]["x"], 
                options=options, path=tree_name, n_jobs=8)
            w_shape = tree_model.flat_model.weights.shape
            results[type].append(
                (train_time, model_nnz, model_nnz/(w_shape[0]*w_shape[1])*100))
            print(f"{args.dataset}\t{type2name[type]}\t" + \
                f"Time: {train_time:.2f} sec\t" + \
                f"Sparsity: {model_nnz/(w_shape[0]*w_shape[1])*100:.2f}%")

    print("=============== Summary ===============")
    for type in types:
        avg = np.average(np.vstack(results[type]), axis=0)
        print(f"{args.dataset}\t{type2name[type]}\t" + \
                f"Time: {avg[0]:.2f} sec\t" + \
                f"Sparsity: {avg[2]:.2f}%")

    with open(f"logs/{args.dataset}_solver_compare_default.json", "w") as f:
        f.writelines(json.dumps(results))