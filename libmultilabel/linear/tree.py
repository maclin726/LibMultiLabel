from __future__ import annotations

from typing import Callable

import pickle
import matplotlib.pyplot as plt

import numpy as np
import scipy.sparse as sparse
import sklearn.cluster
import sklearn.preprocessing
from tqdm import tqdm

from . import linear
# from . import kmeans

__all__ = ["train_tree", "get_label_tree", "Node"]


class Node:
    def __init__(
        self,
        label_map: np.ndarray,
        children: list[Node],
    ):
        """
        Args:
            label_map (np.ndarray): The labels under this node.
            children (list[Node]): Children of this node. Must be an empty list if this is a leaf node.
        """
        self.label_map = label_map
        self.children = children

    def isLeaf(self) -> bool:
        return len(self.children) == 0

    def dfs(self, visit: Callable[[Node], None] | Callable[[Node, int], None], depth=None):
        if depth == None:
            visit(self)
        else:
            visit(self, depth)
        # Stops if self.children is empty, i.e. self is a leaf node
        for child in self.children:
            if depth == None:
                child.dfs(visit)
            else:
                child.dfs(visit, depth+1)


class TreeModel:
    def __init__(
        self,
        root: Node,
        flat_model: linear.FlatModel,
        weight_map: np.ndarray,
    ):
        self.name = "tree"
        self.root = root
        self.flat_model = flat_model
        self.weight_map = weight_map
        self.multiclass = False

    def predict_values(
        self,
        x: sparse.csr_matrix,
        beam_width: int = 10,
    ) -> np.ndarray:
        """Calculates the decision values associated with x.

        Args:
            x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
            beam_width (int, optional): Number of candidates considered during beam search. Defaults to 10.

        Returns:
            np.ndarray: A matrix with dimension number of instances * number of classes.
        """
        # number of instances * number of labels + total number of metalabels
        all_preds = linear.predict_values(self.flat_model, x)
        return np.vstack([self._beam_search(all_preds[i], beam_width) for i in range(all_preds.shape[0])])

    def _beam_search(self, instance_preds: np.ndarray, beam_width: int) -> np.ndarray:
        """Predict with beam search using cached decision values for a single instance.

        Args:
            instance_preds (np.ndarray): A vector of cached decision values of each node, has dimension number of labels + total number of metalabels.
            beam_width (int): Number of candidates considered.

        Returns:
            np.ndarray: A vector with dimension number of classes.
        """
        cur_level = [(self.root, 0.0)]  # pairs of (node, score)
        next_level = []
        while True:
            num_internal = sum(map(lambda pair: not pair[0].isLeaf(), cur_level))
            if num_internal == 0:
                break

            for node, score in cur_level:
                if node.isLeaf():
                    next_level.append((node, score))
                    continue
                slice = np.s_[self.weight_map[node.index] : self.weight_map[node.index + 1]]
                pred = instance_preds[slice]
                children_score = score - np.maximum(0, 1 - pred) ** 2
                next_level.extend(zip(node.children, children_score.tolist()))

            cur_level = sorted(next_level, key=lambda pair: -pair[1])[:beam_width]
            next_level = []

        num_labels = len(self.root.label_map)
        scores = np.full(num_labels, -np.inf)
        for node, score in cur_level:
            slice = np.s_[self.weight_map[node.index] : self.weight_map[node.index + 1]]
            pred = instance_preds[slice]
            scores[node.label_map] = np.exp(score - np.maximum(0, 1 - pred) ** 2)
        return scores


def train_tree(
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    options: str = "",
    clustering: str = "spherical",
    K=100,
    dmax=10,
    verbose: bool = True,
    path: str = None,
) -> tuple[TreeModel, float]:
    """Trains a linear model for multiabel data using a divide-and-conquer strategy.
    The algorithm used is based on https://github.com/xmc-aalto/bonsai.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        clustering (str): Clustering algorithm, one of {spherical, balanced_spherical, elkan}. Defaults to spherical.
        K (int, optional): Maximum degree of nodes in the tree. Defaults to 100.
        dmax (int, optional): Maximum depth of the tree. Defaults to 10.
        verbose (bool, optional): Output extra progress information. Defaults to True.

    Returns:
        A model which can be used in predict_values.
    """
    if clustering not in {"spherical", "balanced_spherical", "elkan", "random"}:
        raise ValueError(f"invalid clustering {clustering}")

    if path is not None:
        with open(path, "rb") as f:
            root = pickle.load(f)
        print(f"Succesfully load tree {path}")
    else:
        label_representation = (y.T * x).tocsr()
        label_representation = sklearn.preprocessing.normalize(label_representation, norm="l2", axis=1)
        root = _build_tree(label_representation, np.arange(y.shape[1]), clustering, 0, K, dmax)

    num_nodes = 0

    def count(node):
        nonlocal num_nodes
        num_nodes += 1

    root.dfs(count)

    pbar = tqdm(total=num_nodes, disable=not verbose, ncols=79)

    def visit(node):
        relevant_instances = y[:, node.label_map].getnnz(axis=1) > 0
        _train_node(y[relevant_instances], x[relevant_instances], options, node)
        pbar.update()

    root.dfs(visit)
    pbar.close()

    flat_model, weight_map = _flatten_model(root)
    
    tree_model = TreeModel(root, flat_model, weight_map)
    train_time = pbar.format_dict["elapsed"]
    model_nnz = tree_model.flat_model.weights.nnz
    return tree_model, train_time, model_nnz
    # return TreeModel(root, flat_model, weight_map)


def _build_tree(
    label_representation: sparse.csr_matrix, label_map: np.ndarray, clustering: str, d: int, K: int, dmax: int
) -> Node:
    """Builds the tree recursively by kmeans clustering.

    Args:
        label_representation (sparse.csr_matrix): A matrix with dimensions number of classes under this node * number of features.
        label_map (np.ndarray): Maps 0..label_representation.shape[0] to the original label indices.
        clustering (str): Clustering algorithm, one of {spherical, balanced_spherical, elkan}. Defaults to spherical.
        d (int): Current depth.
        K (int): Maximum degree of nodes in the tree.
        dmax (int): Maximum depth of the tree.

    Returns:
        Node: root of the (sub)tree built from label_representation.
    """
    if d >= dmax or label_representation.shape[0] <= K:
        return Node(label_map=label_map, children=[])

    if clustering == "elkan":
        metalabels = (
            sklearn.cluster.KMeans(
                K,
                random_state=np.random.randint(2**31 - 1),
                n_init=1,
                max_iter=300,
                tol=0.0001,
                algorithm="elkan",
            )
            .fit(label_representation)
            .labels_
        )
    # elif clustering == "spherical":
    #     metalabels = kmeans.spherical(label_representation, K, max_iter=300, tol=0.0001)
    # elif clustering == "balanced_spherical":
    #     metalabels = kmeans.balanced_spherical(label_representation, K, max_iter=300, tol=0.0001)
    # elif clustering == "random":
    #     metalabels = kmeans.random_clustering(label_representation, K)

    children = []
    for i in range(K):
        child_representation = label_representation[metalabels == i]
        child_map = label_map[metalabels == i]
        child = _build_tree(child_representation, child_map, clustering, d + 1, K, dmax)
        children.append(child)

    return Node(label_map=label_map, children=children)


def _train_node(y: sparse.csr_matrix, x: sparse.csr_matrix, options: str, node: Node):
    """If node is internal, computes the metalabels representing each child and trains
    on the metalabels. Otherwise, train on y.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        node (Node): Node to be trained.
    """
    if node.isLeaf():
        node.model = linear.train_1vsrest(y[:, node.label_map], x, False, options, False)
    else:
        # meta_y[i, j] is 1 if the ith instance is relevant to the jth child.
        # getnnz returns an ndarray of shape number of instances.
        # This must be reshaped into number of instances * 1 to be interpreted as a column.
        meta_y = [y[:, child.label_map].getnnz(axis=1)[:, np.newaxis] > 0 for child in node.children]
        meta_y = sparse.csr_matrix(np.hstack(meta_y))
        node.model = linear.train_1vsrest(meta_y, x, False, options, False)

    node.model.weights = sparse.csc_matrix(node.model.weights)


def _flatten_model(root: Node) -> tuple[linear.FlatModel, np.ndarray]:
    """Flattens tree weight matrices into a single weight matrix. The flattened weight
    matrix is used to predict all possible values, which is cached for beam search.
    This pessimizes complexity but is faster in practice.
    Consecutive values of the returned map denotes the start and end indices of the
    weights of each node. Conceptually, given root and node:
        flat_model, weight_map = _flatten_model(root)
        slice = np.s_[weight_map[node.index]:
                      weight_map[node.index+1]]
        node.model.weights == flat_model.weights[:, slice]

    Args:
        root (Node): Root of the tree.

    Returns:
        tuple[linear.FlatModel, np.ndarray]: The flattened model and the ranges of each node.
    """
    index = 0
    weights = []
    bias = root.model.bias

    def visit(node):
        assert bias == node.model.bias
        nonlocal index
        node.index = index
        index += 1
        weights.append(node.model.__dict__.pop("weights"))

    root.dfs(visit)

    

    model = linear.FlatModel(
        name="flattened-tree",
        weights=sparse.hstack(weights, "csr"),
        bias=bias,
        thresholds=0,
        multiclass=False,
    )

    # w.shape[1] is the number of labels/metalabels of each node
    weight_map = np.cumsum([0] + list(map(lambda w: w.shape[1], weights)))

    return model, weight_map


def get_label_tree(
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    options: str = "",
    K=100,
    dmax=10,
    verbose: bool = True,
    cluster: str = "elkan",
) -> Node:
    """
    Get the information of the constructed label tree (nnz_feature, num_rel_data)
    The algorithm used is based on https://github.com/xmc-aalto/bonsai.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        K (int, optional): Maximum degree of nodes in the tree. Defaults to 100.
        dmax (int, optional): Maximum depth of the tree. Defaults to 10.
        verbose (bool, optional): Output extra progress information. Defaults to True.
    """
    label_representation = (y.T * x).tocsr()
    label_representation = sklearn.preprocessing.normalize(label_representation, norm="l2", axis=1)
    root = _build_tree(
        label_representation, np.arange(y.shape[1]), cluster, 0, K, dmax)

    num_nodes = 0

    def count(node):
        nonlocal num_nodes
        num_nodes += 1

    root.dfs(count)

    pbar = tqdm(total=num_nodes, disable=not verbose, ncols=79)

    def visit(node, depth):
        node.depth = depth
        # relexvant_instances = y[:, node.label_map].getnnz(axis=1) > 0
        # x[relevant_instances].nonzero()[1] extracts the column indices with nz elements
        # node.num_nnz_feat = np.unique(x[relevant_instances].nonzero()[1]).shape[0]
        # node.num_nnz_feat = np.count_nonzero(x[relevant_instances].sum(axis=0))
        # node.num_rel_data = np.count_nonzero(relevant_instances)
        
        node.num_nnz_feat = np.count_nonzero(label_representation[node.label_map,:].sum(axis=0))
        pbar.update()

    root.dfs(visit, 0)
    pbar.close()
    
    return root