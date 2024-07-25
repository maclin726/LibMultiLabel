import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import subprocess
import json
import numpy as np
import os.path
import math
from matplotlib import ticker as mtick

datasets = [
    ("eurlex", "txt", 3993), 
    ("amazoncat-13k", "svm", 13330), 
    ("wiki10-31k", "svm", 30928), 
    ("wiki-500k", "svm", 501070), 
    ("amazon-670k", "svm", 670091), 
    ("amazon-3m", "svm", 2812281)
]

dataset_name = [
    "EUR-Lex", 
    "AmazonCat-13k", 
    "Wiki10-31k", 
    "Wiki-500k",
    "Amazon-670k",
    "Amazon-3m"
]

def gen_fixed_K_logs():
    for dataset in datasets:
        for depth in range(1, 6):
            if os.path.isfile(f"logs/{dataset[0]}_elkan_d{depth}_K100.json"):
                continue
            
            print(f"File {dataset[0]}_elkan_d{depth}_K100.json not found. Generating a new json log...")
            cmd = [
                'python3', '-W ignore', 'tree_space.py', 'load_tree', 
                '--cluster', 'elkan',
                '--K', '100', 
                '--dmax', '10', 
                '--clip_depth', str(depth),
                dataset[1], dataset[0],
                'label_tree_obj/default'
            ]
            subprocess.run(cmd)
            # subprocess.run(cmd, stdout=subprocess.PIPE)
        
def plot_fixed_K(start_d, end_d):
    fmt = '%.3f'  # as per no of zero(or other) you want after decimal point
    yticks = mtick.FormatStrFormatter(fmt)
    for i, dataset in enumerate(datasets):

        logs = dict()
        for d in range(start_d, end_d):
            with open(f"logs/{dataset[0]}_elkan_d{d}_K100.json", "rb") as f:
                logs[d] = json.load(f)

        fig, axs = plt.subplots(1, 1)

        ratio_avg = [
            np.average(logs[d]["ratio"]) for d in range(start_d, end_d)
        ]

        tree_size_avg = [ np.average(logs[d]["model_size"])
            for d in range(start_d, end_d) ]
        ovr_size = tree_size_avg[-1]/ratio_avg[-1]

        tree_size_unit = "GB"
        ovr_size_unit = "TB" if ovr_size > 1024**4 else "GB"
        ovr_power = 4 if ovr_size > 1024**4 else 3
        textstr = '\n'.join(
            [f'Tree ($d_{{max}}=6$): {tree_size_avg[-1]/(1024**3):.2f} ({tree_size_unit})',
            f'OVR: {ovr_size/(1024**ovr_power):.2f} ({ovr_size_unit})',
            f'Model size ratio: {ratio_avg[-1]:.3f}']
        )
        
        axs.plot(list(range(start_d+1, end_d+1)), ratio_avg, "-o")
        axs.set_xticks(range(start_d+1, end_d+1))
        # axs2 = axs.twinx()
        # axs2.plot(list(range(start_d+1, end_d+1)), size_avg, "-o")
        axs.tick_params("x", labelsize=30)
        axs.tick_params("y", labelsize=15)
        axs.set_title(dataset_name[i], fontsize=30)
        if i % 3 == 0:
            axs.set_ylabel("Model-size ratio: Tree / OVR", fontsize=15)
        # if i % 3 == 2:
        #     axs2.set_ylabel("Tree model size (GB)", fontsize=15)
        axs.set_xlabel('Tree depth', fontsize=20)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axs.text(0.98, 0.65, textstr, 
                 transform=axs.transAxes, horizontalalignment='right',
                 bbox=props,
                 fontsize=20)
        axs.grid()
        axs.yaxis.set_major_formatter(yticks)
        fig.tight_layout()
        fig.savefig(f"figs/ratio-fixed-K-{dataset[0]}.png")

def gen_varied_K_logs():
    for dataset in datasets:
        for depth in range(1, 4):
            K = math.ceil(dataset[2]**(1/(depth+1)))
            if os.path.isfile(f"logs/{dataset[0]}_elkan_d{depth}_K{K}.json"):
                continue
            
            print(f"File {dataset[0]}_elkan_d{depth}_K{K}.json not found. Generating a new json log...")
            cmd = [
                'python3', '-W ignore', 'tree_space.py', 'load_tree', 
                '--cluster', 'elkan',
                '--K', str(K), 
                '--dmax', str(depth), 
                dataset[1], dataset[0],
                'label_tree_obj/bonsai'
            ]
            subprocess.run(cmd)
            # subprocess.run(cmd, stdout=subprocess.PIPE)

def plot_varied_K(start_d, end_d):
    
    fmt = '%.2f'  # as per no of zero(or other) you want after decimal point
    yticks = mtick.FormatStrFormatter(fmt)
    fig, axs = plt.subplots(1, 6, figsize=[25, 5], dpi=160)
    plt.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.15)

    for i, dataset in enumerate(datasets):
        logs = dict()
        for d in range(start_d, end_d):
            K = math.ceil(dataset[2]**(1/(d+1)))
            with open(f"logs/{dataset[0]}_elkan_d{d}_K{K}.json", "rb") as f:
                logs[d] = json.load(f)

        ratio_avg = [
            np.average(logs[d]["ratio"]) for d in range(start_d, end_d)
        ]
        
        axs[i].plot(list(range(start_d+1, end_d+1)), ratio_avg, "-o")
        axs[i].set_xticks(range(start_d+1, end_d+1))

        axs[i].tick_params("x", labelsize=22)
        axs[i].tick_params("y", labelsize=18)
        axs[i].set_title(dataset_name[i], fontsize=25)
        if i == 0:
            axs[i].set_ylabel("Model-size: Tree / OVR", fontsize=20)
        axs[i].set_xlabel('Tree depth', fontsize=20)
        axs[i].grid()
        axs[i].yaxis.set_major_formatter(yticks)
    
    fig.tight_layout()
    fig.savefig(f"figs/ratio-varied-K.png")

gen_fixed_K_logs()
plot_fixed_K(1, 6)

gen_varied_K_logs()
plot_varied_K(1, 4)