import libmultilabel.linear as linear
from libmultilabel.linear import Node
from argparse import ArgumentParser
import os.path
import pickle
import time
import numpy as np
import tqdm

parser = ArgumentParser()
parser.add_argument("mode", 
                    choices=['all', 'build_tree', 'load_tree'], 
                    default='all')
parser.add_argument("format", help="format")
parser.add_argument("dataset", help="data set")
parser.add_argument("root_dir")
parser.add_argument("--K", type=int, default=100)
parser.add_argument("--dmax", type=int, default=10)
parser.add_argument("--cluster", default="elkan", 
                    choices=["elkan", "balanced_spherical", "random"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--clip_depth", type=int, default=10)
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

def build_tree(datasets, root_dir, dataset_name, K, dmax, cluster, seed):
    np.random.seed(seed)
    root = linear.get_label_tree(
            datasets["train"]["y"], 
            datasets["train"]["x"],
            K=K,
            dmax=dmax,
            cluster=cluster
        )

    with open(f"{root_dir}/{dataset_name}_{cluster}_d{dmax}_K{K}_seed{seed}.pkl", "wb") as f:
        pickle.dump(root, f)
        print(f"Save tree pickle named {dataset_name}_{cluster}_d{dmax}_K{K}_seed{seed}.pkl")

def load_tree(root_dir, dataset_name, K, dmax, cluster, seed):
    try:
        with open(f"{root_dir}/{dataset_name}_{cluster}_d{dmax}_K{K}_seed{seed}.pkl", "rb") as f:
            root = pickle.load(f)
        return root
    except:
        print("Label tree pickle does not exist.")

def get_depthwise_stat(root, clip_depth):
    stat = dict()
    
    for d in range(args.dmax+2):
        stat[d] = {
            "num_labels": [],
            "num_nnz_feats": [],
            "num_train": [],    # number of training data in this node
            "num_children": [], # number of internal child nodes
            "num_branches": [],
            "num_nnz_feats_alpha": [],
            "num_rel_data_alpha": [],
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
        # stat[node.depth]["num_train"].append(node.num_rel_data)
        stat[node.depth]["num_nnz_feats"].append(node.num_nnz_feat)
        stat[node.depth]["num_children"].append(len(node.children))
        
        nonlocal clip_depth
        if node.isLeaf() or node.depth == clip_depth:
            stat[node.depth]["num_branches"].append(len(node.label_map))
        else:
            stat[node.depth]["num_branches"].append(len(node.children))

    root.dfs(collect_stat)

    tree_depth = 0
    while len(stat[tree_depth]["num_labels"]) > 0:
        tree_depth += 1

    return stat, min(tree_depth, clip_depth+1)

def get_sparsity(stat, tree_depth):
    tree_size = 0
    total_labels = 0
    for d in range(tree_depth):
        n_feat = np.array(stat[d]["num_nnz_feats"])
        n_model = np.array(stat[d]["num_branches"])
        tree_size += np.dot(n_feat, n_model)

        for num_children, num_labels in\
            zip(stat[d]["num_children"], stat[d]["num_labels"]):
            if num_children == 0 or d == tree_depth-1:
                total_labels += num_labels

    n = stat[0]["num_nnz_feats"][0]
    L = stat[0]["num_labels"][0]
    assert(total_labels == L)
    return tree_size / (n*L)


# load datasets
datasets = load_dataset_pickle(args.format, args.dataset)

# build tree
if args.mode == 'all' or args.mode == 'build_tree':
    build_tree(
        datasets, args.root_dir, args.dataset, args.K, args.dmax, args.cluster, args.seed)

# load tree
if args.mode == 'all' or args.mode == 'load_tree':
    tree_files = [filename for filename in os.listdir(args.root_dir)\
              if filename.startswith(
                  f"{args.dataset}_{args.cluster}_d{args.dmax}_K{args.K}")]
    
    seeds = [int(file.split("seed")[1].split(".")[0]) for file in tree_files]
    
    results = []
    for seed in seeds:
        root = load_tree(args.root_dir, args.dataset, args.K, args.dmax, args.cluster, seed)
        stat, tree_depth = get_depthwise_stat(root, args.clip_depth)

        print("\n")
        print("tree depth:", tree_depth)
        leafs = np.array(stat[tree_depth-1]["num_branches"])
        print("leaf nodes:", np.count_nonzero(leafs > 0))
        print("larger than K:", np.count_nonzero(leafs > args.K))
        print("larger than 1:", np.count_nonzero(leafs > 1))
        sparsity = get_sparsity(stat, tree_depth)
        results.append(sparsity)
        print(f"seed {seed} sparsity: {sparsity:.5f}")
    print(f"{sum(results)/len(results):.5f}")

# results = dict()
# for d in range(args.dmax+1):
#     results[d] = {"n_models": [],
#                   "avg_nnz_feats_per_model": [],
#                   "nnz_alpha": [],
#                  }
#     if d == 0:
#         results[d]["nnz_alpha"] = [1] * len(seeds)
# results["model_size_reduction"] = []



    # total_model_size = 0
    # total_labels = 0
    # for d in range(tree_depth):
    #     n_feat = np.array(stat[d]["num_nnz_feat"])
    #     n_model = np.array(stat[d]["num_branches"])
    #     print(n_model)
    #     model_size = np.dot(n_feat, n_model)
    #     results[d]["n_models"].append(sum(stat[d]["num_branches"]))
    #     results[d]["avg_nnz_feats_per_model"].append(model_size/sum(stat[d]["num_branches"]))
    #     if d > 0:
    #         results[d]["nnz_alpha"].append(
    #             results[d]["avg_nnz_feats_per_model"][-1]/results[d-1]["avg_nnz_feats_per_model"][-1])
    #     total_model_size += model_size

    #     # compute number of total labels from child nodes
    #     for num_children, num_labels in zip(stat[d]["num_children"], stat[d]["num_label_map"]):
    #         if num_children == 0:
    #             total_labels += num_labels

    # print(f"total_labels: {total_labels}")

    # n = stat[0]["num_nnz_feat"][0]
    # L = stat[0]["num_label_map"][0]
    # results["model_size_reduction"].append(total_model_size/(n*L))
    # print(f"Model size reduction: {total_model_size} / {n*L} = {total_model_size / (n*L): .5f}")
    
# # Statistics of several label trees
# time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
# f = open(f'logs/{time_str}-{args.dataset}-{args.cluster}-{args.K}--{args.dmax}.txt', "w")
# print(f"The information of constructed tree after {len(seeds)} times of exps")
# print("\\begin{center}\n\\begin{tabular}{c c c c}", file=f)
# print("layer ($i$) & $c_i$ & $\\bar{n}_i$ &$\\alpha_i$\\\\", file=f)
# for d in range(args.dmax+1):
#     if len(results[d]["n_models"]) == 0:
#         break
#     print("\hline", file=f)
#     out = ""
#     n_models = np.array(results[d]["n_models"])
#     n_models_avg = np.average(n_models)
#     n_models_std = np.std(n_models)
#     avg_nnz_feats_per_model = np.array(results[d]["avg_nnz_feats_per_model"])
#     avg_nnz_feats_per_model_avg = np.average(avg_nnz_feats_per_model)
#     avg_nnz_feats_per_model_std = np.std(avg_nnz_feats_per_model)
#     nnz_alpha = np.array(results[d]["nnz_alpha"])
#     nnz_alpha_avg = np.average(nnz_alpha)
#     nnz_alpha_std = np.std(nnz_alpha)
#     out += f'{d+1} & ${n_models_avg:.0f}$'
#     out += f'& ${avg_nnz_feats_per_model_avg:.0f}$'
#     out += f'& ${nnz_alpha_avg*100:.2f}\\%$\\\\'
#     # out += f'{d+1} & ${n_models_avg:.0f}\pm {n_models_std:.0f}$'
#     # out += f'& ${avg_nnz_feats_per_model_avg:.0f}\pm {avg_nnz_feats_per_model_std:.0f}$'
#     # out += f'& ${nnz_alpha_avg*100:.2f}\\% \pm {nnz_alpha_std*100:.2f}\\%$\\\\'
#     print(out, file=f)

# reductions = np.array(results["model_size_reduction"])
# reductions_avg = np.average(reductions)
# reductions_std = np.std(reductions)
# print('\\multicolumn{4}{c}{', file=f, end='')
# print(f"Reduction rate: ${reductions_avg*100:.2f}\\%$", file=f, end='')
# print('}', file=f)
# print("\\end{tabular}\n\\end{center}", file=f)
# print(f"The model is reduced to ${reductions_avg*100:.2f}\% \pm {reductions_std*100:.2f}\%$ compared to 1-vs-rest.", file=f)
# f.close()
