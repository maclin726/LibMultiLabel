import os.path
import pickle
import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

import libmultilabel.linear as linear
from libmultilabel.linear import Node

parser = ArgumentParser()
parser.add_argument("mode",
                    choices=['all', 'build_tree', 'load_tree', 'clip_tree'],
                    default='all')
parser.add_argument("format", help="format")
parser.add_argument("dataset", help="data set")
parser.add_argument("tree_root_dir",
                    help="the directory to store label trees")
parser.add_argument("--K", type=int, default=100)
parser.add_argument("--dmax", type=int, default=10)
parser.add_argument("--cluster", default="elkan",
                    choices=["elkan", "balanced_spherical", "random"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--clip_depth", type=int, default=10,
                    help="specify the depth you want to compute the model size")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--plot_alphas", action="store_true")
args = parser.parse_args()

sanity_check = False


def load_dataset_pickle(format, dataset):
    if os.path.isfile(f"data/{dataset}/dataset.pkl"):
        with open(f"data/{dataset}/dataset.pkl", "rb") as f:
            datasets = pickle.load(f)
            print("load pickle succeed")
    else:
        datasets = linear.load_dataset(
            args.format,
            f"data/{dataset}/train.{format}",
            f"data/{dataset}/test.{format}")
        preprocessor = linear.Preprocessor()
        datasets = preprocessor.fit_transform(datasets)
        with open(f"data/{dataset}/dataset.pkl", "wb") as f:
            pickle.dump(datasets, f)

    print(f"{dataset} pickle file loaded with shape",
          datasets["train"]["x"].shape,
          datasets["train"]["y"].shape)

    return datasets


def build_tree(datasets, tree_root_dir, dataset_name, K, dmax, cluster, seed):
    np.random.seed(seed)
    root = linear.get_label_tree(
        datasets["train"]["y"],
        datasets["train"]["x"],
        K=K,
        dmax=dmax,
        cluster=cluster
    )

    with open(f"{tree_root_dir}/{dataset_name}_{cluster}_d{dmax}_K{K}_seed{seed}.pkl", "wb") as f:
        pickle.dump(root, f)
        print(
            f"Save tree pickle named {dataset_name}_{cluster}_d{dmax}_K{K}_seed{seed}.pkl")


def load_tree(tree_root_dir, dataset_name, K, dmax, cluster, seed):
    try:
        with open(f"{tree_root_dir}/{dataset_name}_{cluster}_d{dmax}_K{K}_seed{seed}.pkl", "rb") as f:
            root = pickle.load(f)
        return root
    except:
        print("Label tree pickle does not exist.")


def clip_tree(tree_root_dir, dataset_name, K, dmax, cluster, seed, clip_depth):
    root = load_tree(tree_root_dir, dataset_name, K, dmax, cluster, seed)
    
    def clip(node: Node):
        nonlocal clip_depth
        if node.depth == clip_depth:
            node.children = []

    root.dfs(clip)

    with open(f"{tree_root_dir}/{dataset_name}_{cluster}_d{clip_depth}_K{K}_seed{seed}.pkl", "wb") as f:
        pickle.dump(root, file=f)


def get_depthwise_stat(root, clip_depth):
    stat = dict()

    for d in range(args.dmax+2):
        stat[d] = {
            "num_labels": [],
            "num_nnz_feats": [],
            "num_children": [],  # number of internal child nodes
            "num_branches": [],  # number of classifiers to train
            "alphas": [],
            # "num_rel_data_alpha": [],
        }

    def collect_stat(node: Node):
        # global sanity_check
        # if node.depth == 3 and sanity_check is False:
        #     relevant_instances = datasets["train"]["y"][:, node.label_map].getnnz(axis=1) > 0
        #     for i, rel in enumerate(relevant_instances):
        #         check_relevant = False
        #         for j in node.label_map:
        #             if datasets["train"]["y"][i,j] == 1:
        #                 check_relevant = True
        #         if rel == True:
        #             try:
        #                 assert check_relevant == True
        #             except:
        #                 print(rel, i)
        #         else:
        #             try:
        #                 assert check_relevant == False
        #             except:
        #                 print(rel, i)
        #     print("pass")

        #     num_feats = 0
        #     feature_vecs = datasets["train"]["x"][relevant_instances]
        #     for i in tqdm.tqdm(range(feature_vecs.shape[1])):
        #         for j in range(feature_vecs.shape[0]):
        #             if feature_vecs[j,i]!= 0:
        #                 num_feats += 1
        #                 break
        #     assert num_feats == node.num_nnz_feat
        #     print("pass")

        #     sanity_check = True

        stat[node.depth]["num_labels"].append(len(node.label_map))
        stat[node.depth]["num_nnz_feats"].append(node.num_nnz_feat)
        stat[node.depth]["num_children"].append(len(node.children))

        nonlocal clip_depth
        if node.isLeaf() or node.depth == clip_depth:
            stat[node.depth]["num_branches"].append(len(node.label_map))
        else:
            stat[node.depth]["num_branches"].append(len(node.children))
            for child in node.children:
                stat[node.depth+1]["alphas"].append(
                    child.num_nnz_feat/node.num_nnz_feat)

    root.dfs(collect_stat)

    tree_depth = 0
    while len(stat[tree_depth]["num_labels"]) > 0:
        tree_depth += 1

    return stat, min(tree_depth, clip_depth+1)


def get_nnz_for_tree_model(stat, tree_depth):
    tree_nnz = 0
    total_labels = 0
    for d in range(tree_depth):
        n_feat = np.array(stat[d]["num_nnz_feats"])
        n_model = np.array(stat[d]["num_branches"])
        tree_nnz += np.dot(n_feat, n_model)

        # sanity check
        for num_children, num_labels in\
                zip(stat[d]["num_children"], stat[d]["num_labels"]):
            if num_children == 0 or d == tree_depth-1:
                total_labels += num_labels

    L = stat[0]["num_labels"][0]
    assert (total_labels == L)
    return tree_nnz


def get_estimated_model_size(stat, tree_depth):
    return int(get_nnz_for_tree_model(stat, tree_depth) * 12)


def get_tree_OVR_size_ratio(stat, tree_depth):
    n = stat[0]["num_nnz_feats"][0]
    L = stat[0]["num_labels"][0]
    return get_nnz_for_tree_model(stat, tree_depth)*1.5 / (n*L)


def get_depthwise_weighted_alphas(stat, tree_depth):
    avg_alphas = [1, ]
    for d in range(1, tree_depth):
        alphas = np.array(stat[d]["alphas"])
        n_model = np.array(stat[d]["num_branches"])
        avg_alpha = np.dot(alphas, n_model) / np.sum(n_model)
        avg_alphas.append(avg_alpha)

    return avg_alphas


def plot_alphas(stat, tree_depth, dataset_name, K, seed):
    avg_alphas = get_depthwise_weighted_alphas(stat, tree_depth)
    fig, axs = plt.subplots(1, tree_depth-1, figsize=[3*(tree_depth-1), 3.5])
        
    for d in range(1, tree_depth):
        alphas = stat[d]["alphas"]
        if tree_depth > 2:
            axs[d-1].hist(alphas, bins=np.arange(0, 1, 0.02), log=True)
            axs[d-1].set_xlabel(f'Value of $\\alpha$ for depth-{d}', fontsize=12)
            if d == 1:
                axs[d-1].set_ylabel("Count", fontsize=12)
        else:
            axs.hist(alphas, bins=np.arange(0, 1, 0.02), log=True)
            axs.set_xlabel(f'Value of $\\alpha$ for depth-{d}', fontsize=12)
            if d == 1:
                axs.set_ylabel("Count", fontsize=12)
        
        textstr = \
            f'$\\bar{{\\alpha}}$ = {avg_alphas[d]:.4f} '
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        if tree_depth > 2:
            axs[d-1].text(0.95, 0.9, textstr, 
                transform=axs[d-1].transAxes, horizontalalignment='right',
                bbox=props,
                fontsize=14)
        else:
            axs.text(0.95, 0.9, textstr, 
                transform=axs.transAxes, horizontalalignment='right',
                bbox=props,
                fontsize=14)


    # fig.suptitle(name_dict[dataset_name], fontsize=20)
    fig.tight_layout()
    fig.savefig(f"figs/alphas-plot-{dataset_name}-K{K}-seed{seed}.png")


# build tree
if args.mode == 'all' or args.mode == 'build_tree':
    # load datasets
    datasets = load_dataset_pickle(args.format, args.dataset)
    build_tree(
        datasets, args.tree_root_dir, args.dataset, args.K, args.dmax, args.cluster, args.seed)

# load tree
if args.mode == 'all' or args.mode == 'load_tree':
    tree_files = [filename for filename in os.listdir(args.tree_root_dir)
                  if filename.startswith(
                  f"{args.dataset}_{args.cluster}_d{args.dmax}_K{args.K}")]
    # if args.dataset in filename and str(args.K) in filename]
    print(tree_files)
    seeds = [int(file.split("seed")[1].split(".")[0]) for file in tree_files]

    results = {"ratio": [], 
               "model_size": [], 
               "nr_unfinished_nodes": [], 
               "nr_unfinished_labels": [],
               "avg_alphas": []
               }
    for seed in sorted(seeds):
        root = load_tree(args.tree_root_dir, args.dataset,
                         args.K, args.dmax, args.cluster, seed)
        stat, tree_depth = get_depthwise_stat(root, args.clip_depth)

        leafs = np.array(stat[tree_depth-1]["num_branches"])
        sum_labels = 0
        for _ in leafs:
            if _ > args.K:
                sum_labels += _
        tree_OVR_size_ratio = get_tree_OVR_size_ratio(stat, tree_depth)
        
        avg_alphas = get_depthwise_weighted_alphas(stat, tree_depth)
        if args.plot_alphas:
            plot_alphas(stat, tree_depth, args.dataset, args.K, seed)

        n = stat[0]["num_nnz_feats"][0]
        L = stat[0]["num_labels"][0]
        results["model_size"].append(
            get_estimated_model_size(stat, tree_depth))
        results["ratio"].append(round(tree_OVR_size_ratio, 5))
        results["nr_unfinished_nodes"].append(
            int(np.count_nonzero(leafs > args.K)))
        results["nr_unfinished_labels"].append(int(sum_labels))
        results["avg_alphas"].append(avg_alphas)

        if args.verbose:
            print("\n")
            print("Tree depth (include leaves):", tree_depth)
            print("Number of leaf nodes:", np.count_nonzero(leafs > 0))
            print("# leaves with > K labels:", np.count_nonzero(leafs > args.K))
            print("# labels in nodes with > K labels", sum_labels)
            print("larger than 1:", np.count_nonzero(leafs > 1))
            print(f"Average layerwise alphas: {[round(a, 4) for a in avg_alphas]}")
            print(f"seed {seed} size ratio: {tree_OVR_size_ratio:.5f}")

    avg_tree_model_size = \
        sum(results["model_size"])/len(results["model_size"])/(1024**3)
    avg_ratio = avg_tree_model_size / (n*L*8/(1024**3))

    if args.verbose:
        print("\n")
        print(
            f"Avg ratio: Tree({avg_tree_model_size:.3f} GB) / OVR({n*L*8/(1024**3):.3f} GB) = {avg_ratio:.5f}")
    with open(f"logs/{args.dataset}_{args.cluster}_d{args.clip_depth}_K{args.K}.json", "w") as f:
        f.writelines(json.dumps(results))


if args.mode == 'clip_tree':
    tree_files = [filename for filename in os.listdir(args.tree_root_dir)
                  if filename.startswith(
                  f"{args.dataset}_{args.cluster}_d{args.dmax}_K{args.K}")]
    print(tree_files)
    seeds = [int(file.split("seed")[1].split(".")[0]) for file in tree_files]

    for seed in seeds:
        clip_tree(args.tree_root_dir, args.dataset,
                    args.K, args.dmax, args.cluster, seed, args.clip_depth)