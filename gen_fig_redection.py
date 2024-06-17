import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import subprocess
import numpy as np
from matplotlib import ticker as mtick

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
#                "K": [64, 16, 8], # 6, 4
#                "sparsity": [18.01, 12.32, 10.48]}, # 18.01, 13.11, 8.79, 8.26
#     "amazoncat-13k": {"name": "AmazonCat-13k",
#                       "format": "svm",
#                       "K": [116, 24, 11], # 7, 5
#                       "sparsity": [25.32, 15.95, 12.78],}, # 25.32, 15.75, 10.55, 9.47
#     "wiki10-31k": {"name": "Wiki10-31k",
#                    "format": "svm", 
#                    "K": [176, 32, 14], # 8, 6
#                    "sparsity": [54.54, 38.57, 33.81]}, # 54.54, 28.20, 24.72, 25.25
#     "wiki-500k": {
#         "name": "Wiki-500k",
#         "format": "svm",
#         "K": [708, 80, 27, 14, 9],
#         "sparsity": [9.02, 5.53, 5.23, 5.31, 4.61]
#         # [-1, 4.30, 1.86, 1.97]
#     },
#     "amazon-670k": {"name": "Amazon-670k",
#                     "format": "svm",
#                     "K": [819, 88, 29], # 15, 10
#                     "sparsity": [19.34, 13.95, 11.61]}, #[19.34, 7.98, 5.26, 5.16]
#     "amazon-3m": {
#         "name": "Amazon-3m",
#         "format": "svm",
#         "K": [1677, 142, 41, 20, 12],
#         "sparsity": [12.02, 11.20, 8.78, 5.95, 5.95], # 12.02, 5.20, 5.88, 5.82
#         "max": [12.26, 13.16, 10.50],
#         "min": [11.97, 10.02, 7.30]
#         # "K": [1677, 1677, 142, 41, 20, 12],
#         # "sparsity": [12.02, 5.20, 5.88, 5.82, 4.02, 3.14],
#     }
# }

compute_datasets = []
datasets = ["eurlex", "amazoncat-13k", "wiki10-31k", "wiki-500k", "amazon-670k", "amazon-3m"]

for dataset in compute_datasets:
    # controlled-depth
    for i, K in enumerate(configs[dataset]["K"]):
        cmd = [
            'python3', '-W ignore', 'tree_space.py', 'load_tree', 
            '--cluster', 'elkan',
            '--K', str(K),
            '--dmax', str(i+2), 
            '--clip_depth', str(i+1),
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

# fmt = '%.2f'  # as per no of zero(or other) you want after decimal point
# yticks = mtick.FormatStrFormatter(fmt)
# fig, axs = plt.subplots(1, 6, figsize=[25, 5], dpi=160)
# plt.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.15)

# for i, dataset in enumerate(datasets):
#     axs[i].plot(list(range(2, 5)),
#                     np.array(configs[dataset]["sparsity"][:3])*1.5*0.01,
#                     "-o")
#     axs[i].set_xticks(range(2, 5))

#     axs[i].tick_params("x", labelsize=22)
#     axs[i].tick_params("y", labelsize=18)
#     axs[i].set_title(configs[dataset]["name"], fontsize=25)
#     if i == 0:
#         axs[i].set_ylabel("Model-size: Tree / OVR", fontsize=20)
#     axs[i].set_xlabel('Tree depth', fontsize=20)
#     axs[i].grid()
#     axs[i].yaxis.set_major_formatter(yticks)
        
# fig.savefig(f"figs/ratio-controlled-depth.png")

# print six figures
fmt = '%.2f'  # as per no of zero(or other) you want after decimal point
yticks = mtick.FormatStrFormatter(fmt)
for i, dataset in enumerate(datasets):
    fig, axs = plt.subplots(1, 1)
    axs.plot(list(range(2, len(configs[dataset]["sparsity"])+2)),
                    np.array(configs[dataset]["sparsity"])*1.5*0.01,
                    "-o")
    # if dataset in ["eurlex", "amazoncat-13k", "wiki10-31k"]:
    #     rho = configs[dataset]["sparsity"][1]*1.5*0.01
    #     print(rho)
    #     if dataset == "eurlex":
    #         axs.annotate(f"{rho:.01f}",xy=(3, rho-0.1), size="large")
    #     else:
    #         axs.annotate(f"{rho:.01f}",xy=(3, rho-3), size="large")
    # else:
    #     rho = configs[dataset]["sparsity"][2]*1.5*0.01
    #     axs.annotate(f"{rho:.01f}",xy=(4,rho-5), size="large")
    axs.set_xticks(range(2, len(configs[dataset]["sparsity"])+2))
    axs.tick_params("x", labelsize=30)
    axs.tick_params("y", labelsize=15)
    axs.set_title(configs[dataset]["name"], fontsize=30)
    if i % 3 == 0:
        axs.set_ylabel("Model-size: Tree / OVR", fontsize=15)
    axs.set_xlabel('Tree depth', fontsize=25)
    axs.grid()
    axs.yaxis.set_major_formatter(yticks)
    fig.tight_layout()
    fig.savefig(f"figs/ratio-fixed-K-{dataset}.png")

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