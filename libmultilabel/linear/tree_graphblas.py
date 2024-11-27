# import os
# os.environ["OMP_NUM_THREADS"] = "8"

from __future__ import annotations
from typing import Callable

import sklearn
import numpy as np
import scipy.sparse as sparse
import time
from tqdm import tqdm
import sys

from .tree import TreeModel, _train_node, _flatten_model
# from .sparse_cluster import SparseKMeans

import numbers
from sklearn.utils.extmath import row_norms, stable_cumsum
import sparse_dot_mkl as sdm

import graphblas as gb
from graphblas import Matrix, Vector, Scalar
from graphblas import dtypes
from graphblas import semiring

class Node:
    def __init__(
        self,
        label_map: np.ndarray,
        children: list[Node],
        depth: int
    ):
        """
        Args:
            label_map (np.ndarray): The labels under this node.
            children (list[Node]): Children of this node. Must be an empty list if this is a leaf node.
        """
        self.label_map = label_map
        self.children = children
        self.depth = depth

    def isLeaf(self) -> bool:
        return len(self.children) == 0

    def dfs(self, visit: Callable[[Node], None]):
        visit(self)
        # Stops if self.children is empty, i.e. self is a leaf node
        for child in self.children:
            child.dfs(visit)

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


class GraphBLASKMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = check_random_state(random_state)
        
        self.isDense = False

    def fit(self, X: sparse.csr_matrix):
        
        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)

        samples = X.shape[0]
        print("Total Samples: ", samples)

        # Get the square norm of each sample in x
        x_squared_norms = row_norms(X, squared=True)

        # Convert X to GraphBLAS sparse format
        X = gb.io.from_scipy_sparse(X)

        # Initialize centroids from the data points with K-Means++
        # self.centroids is Matrix.sparse.csr (graphblas format)
        self.centroids, indices = self._init_centroids_graph(X, x_squared_norms,random_state=self.random_state)
        c_squared_norms = x_squared_norms[indices]

        # initialize labels assigned to each cluster
        self.labels_ = np.zeros((X.shape[0], 1))
        for i in range(self.max_iter):
            # print(f"Start iteration {i}")

            start_iter = time.time()
            # Step 1: Assign clusters
            XC = Matrix(dtypes.FP64, nrows=samples, ncols=self.n_clusters)
            XC << X.mxm(self.centroids.T)        # mxm
            XC = 2 * XC.to_dense(fill_value=0)
            
            distances_to_clusters = x_squared_norms[:, np.newaxis] - XC + c_squared_norms[np.newaxis, :]
            distances_to_clusters = np.clip(distances_to_clusters, 0, None)

            # Assign sample to its closest centroids.
            labels = np.argmin(distances_to_clusters, axis=1)
            
            # Step 2: Save old centroid and calculate
            old_centroids = self.centroids
            self.centroids, mask = self._update_centroids(X, labels)

            # Step 3: Check if the dense should be set to True (if the centroids is dense)
            if i == 0:
                centroids_sparsity = self.centroids.nvals / (self.centroids.nrows * self.centroids.ncols)
                self.density_ = centroids_sparsity
                # Setting threshold to 0.1 here, convert from sparse -> dense -> full format
                if centroids_sparsity > 0.1:
                    print("Centroids Sparsity by Node: ", centroids_sparsity)
                    self.isDense = True
                    self.centroids = self.centroids.to_dense(fill_value=0)
                    self.centroids = Matrix.from_dense(self.centroids)
                    self.centroids = self.centroids.ss.export("fullc")
                    self.centroids = gb.Matrix.ss.import_fullc(**self.centroids)

            # Getting the old centroid for cluster whose number of samples is 0
            self.centroids[mask, :] << old_centroids[mask, :]

            # Recalculate c_squared_norms for updated centroids
            c_squared_norms_old = c_squared_norms
            c_squared_norms = self.gb_row_norms(self.centroids)

            self.labels_ = labels

            # Step 3: Check for convergence
            if self._converged(old_centroids, c_squared_norms_old, c_squared_norms):
                break

            end_iter = time.time()
            print(f"Time to conduct iteration: {i}", end_iter - start_iter)
            

        return self
    
    def _init_centroids_graph(self, X, x_squared_norms, random_state, sample_weight=None):
        # X is already saved in GraphBLAS sparse format
        # Randomly select the first centroid
        n_samples, n_features = X.shape
        if not sample_weight:
            sample_weight = np.ones(n_samples)

        # Define centroids in hypersparse matrix (0 non-zero entry)
        centroids = Matrix(dtype=dtypes.FP64, nrows=self.n_clusters, ncols=n_features)

        # Pick first center randomly and track index of point
        centroid_id = random_state.choice(n_samples, p=sample_weight / sum(sample_weight))
        indices = np.full(self.n_clusters, -1, dtype=int)

        # Assign first centroid and first index
        centroids[0, :] << X[centroid_id, :]
        indices[0] = centroid_id

        # Scikit-learn Kmeans++ (Greedy Kmeans++) (Able to reproduce results from K-means Scikit-learn)
        distances = Matrix(dtype=dtypes.FP64, nrows=n_samples, ncols=1)
        distances = X[centroid_id, :].vxm(X.T)
        closest_dist_sq = x_squared_norms[indices[0]] - 2 * distances.to_dense(fill_value=0) + x_squared_norms
        current_pot = closest_dist_sq @ sample_weight
        n_local_trials = 2 + int(np.log(self.n_clusters))

        # Scikit-learn Kmeans++ implementation
        for c in range(1, self.n_clusters):
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            rand_vals = random_state.uniform(size=n_local_trials) * current_pot

            candidate_ids = np.searchsorted(
                stable_cumsum(sample_weight * closest_dist_sq), rand_vals
            )
            # XXX: numerical imprecision can result in a candidate_id out of range
            np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

            # Compute distances to center candidates
            candidates = Matrix(dtype=dtypes.FP64, nrows=n_local_trials, ncols=n_features)
            candidates << X[candidate_ids, :]

            distances = Matrix(dtype=dtypes.FP64, nrows=n_local_trials, ncols=n_samples)
            distances << candidates.mxm(X.T)

            # This one slowing down the process
            # distances = Matrix(dtype=dtypes.FP64, nrows=n_samples, ncols=n_local_trials)
            # distances << X.mxm(candidates.T)

            distance_to_candidates = x_squared_norms[:, np.newaxis] - 2 * distances.T.to_dense(fill_value=0) + x_squared_norms[candidate_ids][np.newaxis, :]
            distance_to_candidates = distance_to_candidates.T

            # update closest distances squared and potential for each candidate
            # Broadcasting here for closest_dist_sq
            np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
            candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

            # Decide which candidate is the best
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            # Permanently add best center candidate found in local tries
            centroids[c, :] << X[best_candidate, :]
            indices[c] = best_candidate

        return centroids, indices


    def _update_centroids(self, X, labels):
        n_features = X.shape[1]

        # Binary matrix with weight of a sample for each cluster (for calculating new centroids purpose)
        binary_matrix = np.zeros((self.centroids.shape[0], X.shape[0]))

        # Define a mask to check centroid whose number of samples is 0
        mask = np.ones((self.centroids.shape[0], 1))

        for i in range(self.n_clusters):
            idx = np.where(labels == i)[0]
            if len(idx) > 0:
                binary_matrix[i][idx] = 1 / len(idx)tr
            else:
                mask[i] = 0

        mask = np.where(mask == 0)[0]

        # Calculating new centroids
        binary_matrix = Matrix.from_dense(binary_matrix, missing_value=0)
        centroids = Matrix(dtypes.FP64, nrows=self.n_clusters, ncols=n_features)
        centroids << binary_matrix.mxm(X)

        # Check whether the dense flag is to convert the centroid to Full format
        if self.isDense:
            centroids = centroids.to_dense(fill_value=0)
            centroids = Matrix.from_dense(centroids)
            centroids = centroids.ss.export("fullc")
            centroids = gb.Matrix.ss.import_fullc(**centroids)
            
        return centroids, mask

    def _converged(self, old_centroids, c_squared_norms_old, c_squared_norm):
        OldNew = Matrix(dtypes.FP64, nrows=self.n_clusters, ncols=self.centroids.shape[1])
        OldNew << self.centroids.ewise_mult(old_centroids, op="*[float]")        # mxm
        diff = Vector(float, self.n_clusters)
        diff << OldNew.reduce_rowwise("+[float]")
        centroidsDiff = 2 * diff.to_dense(fill_value=0)
        tol = c_squared_norms_old[:, np.newaxis] - centroidsDiff[:, np.newaxis] + c_squared_norm[:, np.newaxis]
        tol = np.clip(tol, 0, None)
        tol = np.sum(tol)
        return tol <= self.tol 

    def gb_row_norms(self, gbX):
        squared = Matrix(dtypes.FP64, gbX.shape[0], gbX.shape[1])
        squared << gbX.ewise_mult(gbX, op="*[float]")  # method style

        squared_norms = Vector(float, gbX.shape[0])
        squared_norms << squared.reduce_rowwise("+[float]")
        return squared_norms.to_dense()

    def predict(self, X):
        return 


def _build_tree_graphblas(label_representation: sparse.csr_matrix, label_map: np.ndarray,
                                  d: int, K: int, dmax: int, exec_dict: dict, counter_dict: dict, density_dict: dict) -> Node:
    """Builds the tree recursively by kmeans clustering.

    Args:
        label_representation (sparse.csr_matrix): A matrix with dimensions number of classes under this node * number of features.
        label_map (np.ndarray): Maps 0..label_representation.shape[0] to the original label indices.
        d (int): Current depth.
        K (int): Maximum degree of nodes in the tree.
        dmax (int): Maximum depth of the tree.

    Returns:
        Node: root of the (sub)tree built from label_representation.
    """
    samples = label_representation.shape[0]

    if d >= dmax or samples <= K:
        return Node(label_map=label_map, children=[], depth = d)

    start = time.time()

    kmeans = GraphBLASKMeans(
        K,
        random_state=np.random.randint(2**31 - 1),
        max_iter=300,
        tol=0.0001,
        )
    kmeans.fit(label_representation)
    metalabels = kmeans.labels_

    end = time.time()
    
    exec_time = end - start
    print(f"Total time for clustering node at level {d}: ", exec_time)
    if d not in exec_dict:
        exec_dict[d] = 0

    if d not in counter_dict:
        counter_dict[d] = []

    if d not in density_dict:
        density_dict[d] = []

    exec_dict[d] += exec_time
    counter_dict[d].append(samples)
    density_dict[d].append(kmeans.density_)

    children = []
    for i in range(K):
        child_representation = label_representation[metalabels == i]
        child_map = label_map[metalabels == i]
            
        child = _build_tree_graphblas(child_representation, child_map, d + 1, K, dmax, exec_dict, counter_dict, density_dict)
        children.append(child)

    return Node(label_map=label_map, children=children, depth=d)