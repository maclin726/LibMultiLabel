import libmultilabel.linear as linear
from libmultilabel.linear import Node
from argparse import ArgumentParser
import os.path
import pickle
import numpy as np

parser = ArgumentParser()
parser.add_argument("format", help="format")
parser.add_argument("dataset", help="data set")
args = parser.parse_args()

if os.path.isfile(f"data/{args.dataset}/dataset.pkl"):
    with open(f"data/{args.dataset}/dataset.pkl", "rb") as f:
        datasets = pickle.load(f)
        print("load pickle succeed")
else:
    datasets = linear.load_dataset(
        args.format, 
        f"data/{args.dataset}/train.{args.format}", 
        f"data/{args.dataset}/test.{args.format}")
    preprocessor = linear.Preprocessor()
    datasets = preprocessor.fit_transform(datasets)
    with open(f"data/{args.dataset}/dataset.pkl", "wb") as f:
        pickle.dump(datasets, f)

seeds = list(range(5))

results = dict()
for d in range(20):
    results[d] = {"n_models": [],
                  "avg_nnz_feats_per_model": [],
                  "nnz_alpha": [],
                 }
    if d == 0:
        results[d]["nnz_alpha"] = [1] * len(seeds)
results["model_size_reduction"] = []

print(datasets["train"]["x"].shape, datasets["train"]["y"].shape)

for seed in seeds:
    np.random.seed(seed)
    K = 10
    dmax=10
    # the train_tree method for fast training on data with many labels
    root = linear.get_label_tree(
        datasets["train"]["y"], 
        datasets["train"]["x"],
        # K=K,
        # dmax=dmax
    )
    stat = dict()
    
    for d in range(20):
        stat[d] = dict()
        stat[d]["num_label_map"] = list()
        stat[d]["num_nnz_feat"] = list()
        stat[d]["num_rel_data"] = list()
        stat[d]["num_children"] = list()
        stat[d]["num_branches"] = list()
        stat[d]["num_nnz_feat_alpha"] = list()
        stat[d]["num_rel_data_alpha"] = list()

    def collect_stat(node: Node):
        stat[node.depth]["num_nnz_feat"].append(node.num_nnz_feat)
        stat[node.depth]["num_label_map"].append(len(node.label_map))
        stat[node.depth]["num_rel_data"].append(node.num_rel_data)
        stat[node.depth]["num_children"].append(len(node.children))
        
        if node.isLeaf():
            stat[node.depth]["num_branches"].append(len(node.label_map))
        else:
            stat[node.depth]["num_branches"].append(len(node.children))

    root.dfs(collect_stat)

    tree_depth = 0
    while len(stat[tree_depth]["num_rel_data"]) > 0:
        tree_depth += 1
    print(tree_depth)

    # for i in range(1, tree_depth):
    #     for j in range(len(stat[i]["num_rel_data"])):
    #         stat[i]["num_rel_data_alpha"].append(
    #             stat[i]["num_rel_data"][j] / stat[i-1]["num_rel_data"][j//K]
    #         )
    #     for j in range(len(stat[i]["num_nnz_feat"])):
    #         stat[i]["num_nnz_feat_alpha"].append(
    #             stat[i]["num_nnz_feat"][j] / stat[i-1]["num_nnz_feat"][j//K]
    #         )

    # for d in range(1, tree_depth):
    #     rel_data_alphas = np.array(stat[d]["num_rel_data_alpha"])
    #     macro_data_alpha = np.average(rel_data_alphas)
    #     num_nnz_feat_alphas = np.array(stat[d]["num_nnz_feat_alpha"])
    #     macro_nnz_feat_alpha = np.average(num_nnz_feat_alphas)
        
    #     results[d]["macro_nnz_feat_alpha"].append(macro_nnz_feat_alpha)
    #     results[d]["macro_rel_data_alpha"].append(macro_data_alpha)

    #     micro_data_alpha = \
    #         np.average(np.array(stat[i]["num_rel_data"])) / \
    #         np.average(np.array(stat[i-1]["num_rel_data"]))
    #     micro_nnz_feat_alpha = \
    #         np.average(np.array(stat[i]["num_nnz_feat"])) / \
    #         np.average(np.array(stat[i-1]["num_nnz_feat"]))
    #     results[i]["micro_nnz_feat_alpha"].append(micro_nnz_feat_alpha)
    #     results[i]["micro_rel_data_alpha"].append(micro_data_alpha)

    total_model_size = 0
    total_labels = 0
    for d in range(tree_depth):
        n_feat = np.array(stat[d]["num_nnz_feat"])
        n_model = np.array(stat[d]["num_branches"])
        model_size = np.dot(n_feat, n_model)
        results[d]["n_models"].append(sum(stat[d]["num_branches"]))
        results[d]["avg_nnz_feats_per_model"].append(model_size/sum(stat[d]["num_branches"]))
        if d > 0:
            results[d]["nnz_alpha"].append(
                results[d]["avg_nnz_feats_per_model"][-1]/results[d-1]["avg_nnz_feats_per_model"][-1])
        total_model_size += model_size

        # compute number of total labels from child nodes
        for num_children, num_labels in zip(stat[d]["num_children"], stat[d]["num_label_map"]):
            if num_children == 0:
                total_labels += num_labels

    print(f"total_labels: {total_labels}")

    n = stat[0]["num_nnz_feat"][0]
    L = stat[0]["num_label_map"][0]
    results["model_size_reduction"].append(total_model_size/(n*L))
    print(f"Model size reduction: {total_model_size} / {n*L} = {total_model_size / (n*L): .5f}")
    
# Statistics of several label trees
print(f"The information of constructed tree after {len(seeds)} times of exps")
for d in range(10):
    if len(results[d]["n_models"]) == 0:
        break
    print("\hline")
    out = ""
    n_models = np.array(results[d]["n_models"])
    n_models_avg = np.average(n_models)
    n_models_std = np.std(n_models)
    avg_nnz_feats_per_model = np.array(results[d]["avg_nnz_feats_per_model"])
    avg_nnz_feats_per_model_avg = np.average(avg_nnz_feats_per_model)
    avg_nnz_feats_per_model_std = np.std(avg_nnz_feats_per_model)
    nnz_alpha = np.array(results[d]["nnz_alpha"])
    nnz_alpha_avg = np.average(nnz_alpha)
    nnz_alpha_std = np.std(nnz_alpha)
    out += f'{d+1} & ${n_models_avg:.0f}\pm {n_models_std:.0f}$'
    out += f'& ${avg_nnz_feats_per_model_avg:.0f}\pm {avg_nnz_feats_per_model_std:.0f}$'
    out += f'& ${nnz_alpha_avg*100:.2f}\\% \pm {nnz_alpha_std*100:.2f}\\%$\\\\'
    print(out)

reductions = np.array(results["model_size_reduction"])
reductions_avg = np.average(reductions)
reductions_std = np.std(reductions)
print(f"The model is reduced to {reductions_avg*100:.2f}\% \pm {reductions_std*100:.2f}\% compared to 1-vs-rest.")
    # print(data_alphas, feat_alphas)