import math
import numpy as np
import matplotlib.pyplot as plt

def compute_ratio_fixed_K(L, K, a, start_d, end_d):
    rate = []
    for d in range(start_d, end_d):
        rate.append((K/L)*( ((K*a)**(d-1)-1) / (K*a-1) ) + a**(d-1))
    return np.array(rate)

def compute_ratio_controlled_depth(L, a):
    rate = []
    for d in range(2, 6):
        K = math.ceil(math.pow(L, 1/d))
        # print(K)
        # rate.append((a / math.pow(K, 1/3))**(d-1) + \
        #     K/L*((1-(K*a/math.pow(K, 1/3)**(d-1)))/(1-(K*a/math.pow(K, 1/3)))))
        if a != 1:
            rate.append((K*(1-a**d))/(L*(1-a)))
        if a == 1:
            rate.append(K/L*d)
    return np.array(rate) * 100

def plot_ratio_fixed_K(L, K, start_d, end_d, fig_name):
    fig, axs = plt.subplots(1, 1)
    axs.set_xticks([2,3,4,5])
    axs.tick_params("x", labelsize=20)
    axs.tick_params("y", labelsize=15)
    axs.grid()
    for a in [0.3, 0.4, 0.5, 0.6]:
        X = list(range(start_d, end_d))
        Y = compute_ratio_fixed_K(200000000, 100, a, start_d, end_d)
        axs.plot(X, Y, "-o", label=f"$\\alpha$={a}")
        for x, y in zip(X, Y):
            axs.annotate(f"{y:.03f}",xy=(x,y+0.01))
    axs.legend()
    axs.set_xlabel('Tree depth', fontsize=20)
    axs.set_ylabel("# non-zeros in weight matrix: Tree/OVR", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"figs/{fig_name}")

plot_ratio_fixed_K(2*(10**8), 100, 2, 6, 
                   "fixed-K-with-different-alpha-paper.png")
plot_ratio_fixed_K(2*(10**8), 100, 2, 5, 
                   "fixed-K-with-different-alpha-slides.png")