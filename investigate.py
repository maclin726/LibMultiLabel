# import numpy as np
# import pickle
# import scipy.sparse as sparse

# name = "wiki10-31k"
# with open(f"data/{name}/dataset.pkl", "rb") as f:
#     dataset = pickle.load(f)

# y = dataset["train"]["y"].T.tocsr()
# L = y.shape[0]

# print(np.sort(np.array(y.sum(axis=1)).reshape(-1))[::-1][:10])

# hash_val = []
# for j in range(L):
#     pos = y.indices[y.indptr[j]:y.indptr[j+1]]
#     weights = np.array(list(range(1, len(pos)+1)))
#     hash_val.append(sum(pos*weights))

# print(np.unique(np.array(hash_val)).shape)
# print(L)

import math
import numpy as np
import matplotlib.pyplot as plt
def compute_ratio_fixed_K(L, K, a):
    rate = []
    D = math.ceil(math.log(L, K))
    for d in range(2, D+1):
        rate.append(((a)**(d-1)-1/L) / (K*a-1) + a**(d-1))
    return np.array(rate) * 100

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

fig, axs = plt.subplots(1, 1)
axs.set_xticks([2,3,4,5])
axs.tick_params("x", labelsize=20)
axs.tick_params("y", labelsize=15)
axs.grid()
for a in [1, 3, 5, 7, 9]:
# for a in [0.3, 0.4, 0.5]:
    # X = list(range(2,5))
    # Y = compute_ratio_fixed_K(3000000, 100, a)
    X = list(range(2, 6))
    Y = compute_ratio_controlled_depth(3000000, a)
    axs.plot(X, Y, "-o", label=f"$\\alpha$={a}")
    for x, y in zip(X, Y):
        axs.annotate(f"{y:.01f}",xy=(x,y+1))
axs.legend()
axs.set_xlabel('Tree Depth', fontsize=20)
axs.set_ylabel("Reduction Rate (%)", fontsize=15)
fig.tight_layout()
fig.savefig(f"figs/controlled-depth-with-different-alpha.png")
# fig.savefig(f"figs/fixed-K-with-different-alpha.png")