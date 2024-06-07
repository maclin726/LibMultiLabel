import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import subprocess

configs = {
    "eurlex": {"name": "EUR-Lex",
               "format": "txt",
               "K": [64, 64, 16, 8],
               "sparsity": [18.01, 13.11, 8.79, 8.26]},
    "amazoncat-13k": {"name": "AmazonCat-13k",
                      "format": "svm",
                      "K": [116, 116, 24, 11],
                      "sparsity": [25.32, 15.75, 10.55, 9.47],},
    "wiki10-31k": {"name": "Wiki10-31k",
                   "format": "svm", 
                   "K": [176, 176, 32, 14],
                   "sparsity": [54.54, 28.20, 24.72, 25.25]},
    "amazon-670k": {"name": "Amazon-670k",
                    "format": "svm",
                    "K": [819, 819, 88, 29],
                    "sparsity": [19.34, 7.98, 5.26, 5.16]},
    "amazon-3m": {
        "name": "Amazon-3m",
        "format": "svm",
        "K": [1677, 1677, 142, 41, 20, 12],
        "sparsity": [12.02, 5.20, 5.88, 5.82, 4.02, 3.14],
    }
}

compute_datasets = []
datasets = ["eurlex", "amazoncat-13k", "wiki10-31k", "amazon-670k", "amazon-3m"]

for dataset in compute_datasets:
    for depth, k in enumerate(configs[dataset]["K"]):
        cmd = [
            'python3', '-W ignore', 'tree_space.py', 'load_tree', 
            '--cluster', 'elkan', '--K', str(k), '--dmax', str(depth+1), 
            configs[dataset]["format"], dataset, 'label_tree_obj/bonsai'
        ]
        results = subprocess.run(cmd, stdout=subprocess.PIPE)
        avg = float(results.stdout.decode("utf-8").split("\n")[-2])
        configs[dataset]["sparsity"].append(avg*100)
        print(dataset, depth+1, k, avg)

# print a 1*4 figure
fig, axs = plt.subplots(1, 5, figsize=[20, 5], dpi=160)
plt.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.15)

for i, dataset in enumerate(datasets):
    axs[i].plot(list(range(2, len(configs[dataset]["sparsity"])+2)),
                    configs[dataset]["sparsity"],
                    "-o")
    axs[i].set_xticks(range(2, len(configs[dataset]["sparsity"])+2))

    axs[i].tick_params("x", labelsize=22)
    axs[i].tick_params("y", labelsize=20)
    axs[i].set_title(configs[dataset]["name"], fontsize=25)
    if i == 0:
        axs[i].set_ylabel("Reduction Rate (%)", fontsize=20)
    axs[i].set_xlabel('Tree Depth', fontsize=20)
    axs[i].grid()
        
fig.savefig(f"figs/reduction-rate-vs-depths.png")

# print four figures
for i, dataset in enumerate(datasets):
    fig, axs = plt.subplots(1, 1)
    axs.plot(list(range(2, len(configs[dataset]["sparsity"])+2)),
                    configs[dataset]["sparsity"],
                    "-o")
    axs.set_xticks(range(2, len(configs[dataset]["sparsity"])+2))

    axs.tick_params("x", labelsize=30)
    axs.tick_params("y", labelsize=15)
    axs.set_title(configs[dataset]["name"], fontsize=30)
    axs.set_ylabel("Reduction Rate (%)", fontsize=15)
    axs.set_xlabel('Tree Depth', fontsize=25)
    axs.grid()
    fig.tight_layout()
    fig.savefig(f"figs/reduction-rate-vs-depths-{dataset}.png")
