import libmultilabel.linear as linear
from libmultilabel.linear import Node
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os.path
import pickle
import time
import numpy as np

parser = ArgumentParser()
parser.add_argument("format", help="format")
parser.add_argument("dataset", help="data set")
# parser.add_argument("--K", type=int, default=100)
# parser.add_argument("--dmax", type=int, default=10)
# parser.add_argument("--cluster", default="elkan", choices=["elkan", "balanced_spherical"])
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

X = datasets["train"]["x"].tocsc()
feat_dist = np.sort(np.array([X[:,i].count_nonzero() for i in range(X.shape[1])]))[::-1]
feat_id = [i for i in range(len(feat_dist))]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
line, = ax.plot(feat_id, feat_dist, '-')
ax.set_yscale('log')
ax.set_ylabel('# nonzero instances')
ax.set_xlabel('feature rank')
plt.grid()
fig.savefig(f"data/{args.dataset}/feature_dist.png")