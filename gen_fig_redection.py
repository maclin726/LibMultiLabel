import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import subprocess
import numpy as np

# fixed-K
configs = {
    "eurlex": {
        "name": "EUR-Lex",
        "format": "txt",
        "sparsity": [16.03, 14.84, 14.84, 14.84, 14.84]
    },
    "amazoncat-13k": {
        "name": "AmazonCat-13k",
        "format": "svm",
        "sparsity": [26.40, 14.91, 14.76, 14.76, 14.76],
    },
    "wiki10-31k": {
        "name": "Wiki10-31k",
        "format": "svm", 
        "sparsity": [60.56, 28.72, 23.91, 23.70, 23.69]
    },
    "wiki-500k": {
        "name": "Wiki-500k",
        "format": "svm",
        "sparsity": [29.62, 4.19, 1.82, 1.69, 1.69],
    },
    "amazon-670k": {
        "name": "Amazon-670k",
        "format": "svm",
        "sparsity": [41.59, 13.04, 4.83, 3.04, 2.75]
    },
    "amazon-3m": {
        "name": "Amazon-3m",
        "format": "svm",
        "sparsity": [38.31, 13.79, 5.50, 4.36, 3.95],
    },
}

# controlled-depth
# configs = {
#     "eurlex": {"name": "EUR-Lex",
#                "format": "txt",
#                "K": [64, 64, 16, 8],
#                "sparsity": [18.01, 13.11, 8.79, 8.26]},
#     "amazoncat-13k": {"name": "AmazonCat-13k",
#                       "format": "svm",
#                       "K": [116, 116, 24, 11],
#                       "sparsity": [25.32, 15.75, 10.55, 9.47],},
#     "wiki10-31k": {"name": "Wiki10-31k",
#                    "format": "svm", 
#                    "K": [176, 176, 32, 14],
#                    "sparsity": [54.54, 28.20, 24.72, 25.25]},
#     "wiki-500k": {
#         "name": "Wiki-500k",
#         "format": "svm",
#         "K": [708, 708, 80, 27],
#         "sparsity": [-1, 4.30, 1.86, 1.97]
#     },
#     "amazon-670k": {"name": "Amazon-670k",
#                     "format": "svm",
#                     "K": [819, 819, 88, 29],
#                     "sparsity": [19.34, 7.98, 5.26, 5.16]},
#     "amazon-3m": {
#         "name": "Amazon-3m",
#         "format": "svm",
#         "K": [1677, 1677, 142, 41],
#         "sparsity": [12.02, 5.20, 5.88, 5.82],
#         # "K": [1677, 1677, 142, 41, 20, 12],
#         # "sparsity": [12.02, 5.20, 5.88, 5.82, 4.02, 3.14],
#     }
# }

compute_datasets = []
datasets = ["eurlex", "amazoncat-13k", "wiki10-31k", "wiki-500k", "amazon-670k", "amazon-3m"]

for dataset in compute_datasets:
    # controlled-depth
    for i, K in enumerate(configs["dataset"]["K"]):
        if i == 0:
            continue
        cmd = [
            'python3', '-W ignore', 'tree_space.py', 'load_tree', 
            '--cluster', 'elkan',
            '--K', str(K), 
            '--dmax', i+1, 
            configs[dataset]["format"], dataset, 'label_tree_obj/bonsai'
        ]
        results = subprocess.run(cmd, stdout=subprocess.PIPE)
        avg = float(results.stdout.decode("utf-8").split("\n")[-2])
        configs[dataset]["sparsity"].append(avg*100)
        print(dataset, i+1, avg)

    # for depth in range(1, 6):
    #     cmd = [
    #         'python3', '-W ignore', 'tree_space.py', 'load_tree', 
    #         '--cluster', 'elkan',
    #         '--K', '100', 
    #         '--dmax', '10', '--clip_depth', str(depth), 
    #         configs[dataset]["format"], dataset, 'label_tree_obj/default'
    #     ]
    #     results = subprocess.run(cmd, stdout=subprocess.PIPE)
    #     avg = float(results.stdout.decode("utf-8").split("\n")[-2])
    #     configs[dataset]["sparsity"].append(avg*100)
    #     print(dataset, depth+1, avg)

# controlled-depth
# fig, axs = plt.subplots(1, 6, figsize=[24, 5], dpi=160)
# plt.subplots_adjust(left=0.04, right=0.99, top=0.9, bottom=0.15)

# for i, dataset in enumerate(datasets):
#     axs[i].plot(list(range(3, len(configs[dataset]["sparsity"])+2)),
#                     100-np.array(configs[dataset]["sparsity"][1:])*1.5,
#                     "-o")
#     axs[i].set_xticks(range(3, len(configs[dataset]["sparsity"])+2))

#     axs[i].tick_params("x", labelsize=22)
#     axs[i].tick_params("y", labelsize=20)
#     axs[i].set_title(configs[dataset]["name"], fontsize=25)
#     if i == 0:
#         axs[i].set_ylabel("Reduction Rate (%)", fontsize=20)
#     axs[i].set_xlabel('Tree Depth', fontsize=20)
#     axs[i].grid()
        
# fig.savefig(f"figs/reduction-controlled-depth.png")

# print four figures
for i, dataset in enumerate(datasets):
    fig, axs = plt.subplots(1, 1)
    axs.plot(list(range(2, len(configs[dataset]["sparsity"])+2)),
                    100-np.array(configs[dataset]["sparsity"])*1.5,
                    "-o")
    if dataset in ["eurlex", "amazoncat-13k", "wiki10-31k"]:
        rho = 100-configs[dataset]["sparsity"][1]*1.5
        print(rho)
        if dataset == "eurlex":
            axs.annotate(f"{rho:.01f}",xy=(3, rho-0.1), size="large")
        else:
            axs.annotate(f"{rho:.01f}",xy=(3, rho-3), size="large")
    else:
        rho = 100-configs[dataset]["sparsity"][2]*1.5
        axs.annotate(f"{rho:.01f}",xy=(4,rho-5), size="large")
    axs.set_xticks(range(2, len(configs[dataset]["sparsity"])+2))
    axs.tick_params("x", labelsize=30)
    axs.tick_params("y", labelsize=15)
    axs.set_title(configs[dataset]["name"], fontsize=30)
    if i % 3 == 0:
        axs.set_ylabel("Reduction Rate (%)", fontsize=15)
    axs.set_xlabel('Tree Depth', fontsize=25)
    axs.grid()
    fig.tight_layout()
    fig.savefig(f"figs/fixed-K-{dataset}.png")

# fig, axs = plt.subplots(1, 1)
# axs.set_xticks([2,3,4,5,6])
# axs.tick_params("x", labelsize=30)
# axs.tick_params("y", labelsize=15)
# axs.set_xlabel('Tree Depth', fontsize=25)
# axs.grid()
# axs.set_ylabel("Reduction Rate (%)", fontsize=15)
# for i, dataset in enumerate(datasets):
#     axs.plot(list(range(2, len(configs[dataset]["sparsity"])+2)),
#                     configs[dataset]["sparsity"],
#                     "-o")
# fig.savefig(f"figs/reduction-rate-vs-depths-K100-allin1.png")