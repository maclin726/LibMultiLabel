# import os
# os.environ["OMP_NUM_THREADS"] = "8"

import sklearn
import numpy as np
import scipy.sparse as sparse
import time
from tqdm import tqdm
import sys

# from .tree import Node, TreeModel, _train_node, _flatten_model
from typing import Tuple

import numbers
from sklearn.utils.extmath import row_norms, stable_cumsum
import sparse_dot_mkl as sdm

from scipy.sparse import vstack

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


class MixSparseKMeans:
    def __init__(self,
                 d, 
                 n_clusters=3, 
                 max_iter=100, 
                 tol=1e-4, 
                 random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = []
        self.random_state = check_random_state(random_state)
        
        self.isDense = False
        self.d = d

    def fit(self, X: sparse.csr_matrix):
        
        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)

        print("Total Samples: ", X.shape[0])

        # Get the square norm of each sample in x
        x_squared_norms = row_norms(X, squared=True)

        # Randomly initialize centroids from the data points
        # This will always return self.centroids as sparse.csr no matter what is self.isDense config
        self.centroids, indices = self._init_centroids_csr(X, x_squared_norms,random_state=self.random_state)
        c_squared_norms = x_squared_norms[indices]
        
        # Convert X to sparse column matrix
        X_csr = X # For updating centroids
        X = sparse.csc_matrix(X) # For dot product efficient calculation

        # initialize labels assigned to each cluster
        self.labels_ = np.zeros((X.shape[0], 1))
        for i in range(self.max_iter):
            # print(f"Start iteration {i}")

            start_iter = time.time()
            # Step 1: Assign clusters
            # Calculate the distance from points to clusters (X*X.T - 2X*C + C*C.T)
            # This one is for scipy verision
            # XC = (2 * X.dot(self.centroids.T))

            # Multi-core
            if self.isDense:
                X_csr.indices = X_csr.indices.copy()
                self.centroids = self.centroids.copy()
                XC = 2 * sdm.dot_product_mkl(X_csr, self.centroids.T, dense=True)
                distances_to_clusters = x_squared_norms[:, np.newaxis] - XC + c_squared_norms[np.newaxis, :]
            else:
                X.indices = X.indices.copy()
                self.centroids = self.centroids.copy()
                XC = 2 * sdm.dot_product_mkl(X, self.centroids.T, dense=False)
                distances_to_clusters = x_squared_norms[:, np.newaxis] - XC.A + c_squared_norms[np.newaxis, :]
            
            distances_to_clusters = np.clip(distances_to_clusters, 0, None)

            # Assign sample to its closest centroids.
            labels = np.argmin(distances_to_clusters, axis=1)

            # Step 2: Calculate new centroids
            # If self.isDense True, we save the centroids in dense format (need to convert from the )
            old_centroids = self.centroids
            # self.centroids, mask = self._update_centroids(X_csr, labels)
            self.centroids, mask = self._update_centroids_fixsegment(X_csr, labels)

            # check centroids_sparsity to consider which format should be use
            if i == 0 and self.d == 0:
                self.isDense = True
                # Change the flat and conduct centroids (which is saved in sparse format by default to dense ones)
                old_centroids = old_centroids.A
                self.centroids = self.centroids.A
            
            # print(self.isDense, type(old_centroids), type(self.centroids))
            self.centroids[mask==0, :] = old_centroids[mask==0, :]

            # Recalculate c_squared_norms for updated centroids
            c_squared_norms_old = c_squared_norms
            c_squared_norms = row_norms(self.centroids, squared=True)
            self.labels_ = labels

            # Check for convergence
            if self._converged(old_centroids, c_squared_norms_old, c_squared_norms):
                break

            end_iter = time.time()
            print(f"Time to conduct iteration: {i}", end_iter - start_iter)
            

        return self

    def _init_centroids(self, 
                        X, 
                        x_squared_norms, 
                        random_state, 
                        sample_weight=None) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Old version for _init_centroids
        """
        # Randomly select the first centroid
        n_samples, n_features = X.shape
        if not sample_weight:
            sample_weight = np.ones(n_samples)

        # Create an empty sparse matrix in LIL format (List of List) for easier row filling
        centroids = np.zeros((self.n_clusters, n_features), dtype=X.dtype)

        # Pick first center randomly and track index of point
        centroid_id = random_state.choice(n_samples, p=sample_weight / sum(sample_weight))
        indices = np.full(self.n_clusters, -1, dtype=int)

        # Assign first centroid and first index
        centroids[0] = X[centroid_id].A
        indices[0] = centroid_id

        # Scikit-learn Kmeans++ (Greedy Kmeans++) (Able to reproduce results from K-means Scikit-learn)
        # closest_dist_sq = x_squared_norms[indices[0]] - (2 * X[centroid_id].dot(X.T)).A + x_squared_norms
        closest_dist_sq = x_squared_norms[indices[0]] - 2 * sdm.dot_product_mkl(X[centroid_id], (X.T)).A + x_squared_norms
        current_pot = closest_dist_sq @ sample_weight
        n_local_trials = 2 + int(np.log(self.n_clusters))

        X_csc = sparse.csc_matrix(X)

        # Scikit-learn Kmeans++ implementation
        for i, c in enumerate(range(1, self.n_clusters)):
            start_iter = time.time()
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            rand_vals = random_state.uniform(size=n_local_trials) * current_pot

            candidate_ids = np.searchsorted(
                stable_cumsum(sample_weight * closest_dist_sq), rand_vals
            )
            # XXX: numerical imprecision can result in a candidate_id out of range
            np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

            # Compute distances to center candidates
            # This one slows down computing process
            # candidates = sparse.csc_matrix(X[candidate_ids])
            # distance_to_candidates = x_squared_norms[candidate_ids][:, np.newaxis] - (2 * candidates.dot(X.T)).A + x_squared_norms[np.newaxis, :]

            # We need to conduct X.candidates instead of candidates.X (both stored in csc)
            # candidates.X is much faster compared to X.candidates
            # The scipy performance on these 2 operation is difference, possibly due to the loop for columns in latter matrix
            candidates = X[candidate_ids]
            # distance_to_candidates = x_squared_norms[:, np.newaxis] - (2 * X_csc.dot(candidates.T)).A + x_squared_norms[candidate_ids][np.newaxis, :]
            X_csc.indices = X_csc.indices.copy()
            candidates.indices = candidates.indices.copy()
            distance_to_candidates = x_squared_norms[:, np.newaxis] - 2 * sdm.dot_product_mkl(X_csc, candidates.T).A + x_squared_norms[candidate_ids][np.newaxis, :]
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
            centroids[c] = X[best_candidate].A
            indices[c] = best_candidate

            end_iter = time.time()
            print(f"Time to init iteration: {i}", end_iter - start_iter)

        return sparse.csr_matrix(centroids), indices

    def _init_centroids_csr(self, X, x_squared_norms, random_state, sample_weight=None):
        """
        Latest version for init centroids.
        """
        # Randomly select the first centroid
        n_samples, n_features = X.shape
        if not sample_weight:
            sample_weight = np.ones(n_samples)

        # Create an empty sparse matrix in LIL format (List of List) for easier row filling
        # centroids = sparse.csr_matrix((self.n_clusters, n_features), dtype=X.dtype)
        centroids = []

        # Pick first center randomly and track index of point
        centroid_id = random_state.choice(n_samples, p=sample_weight / sum(sample_weight))
        indices = np.full(self.n_clusters, -1, dtype=int)

        # Assign first centroid and first index
        # centroids[0] = X[centroid_id]
        centroids.append(X[centroid_id])
        indices[0] = centroid_id

        # Scikit-learn Kmeans++ (Greedy Kmeans++) (Able to reproduce results from K-means Scikit-learn)

        # Initialize list of closest distances and calculate current potential
        # closest_dist_sq = x_squared_norms[indices[0]] - (2 * X[centroid_id].dot(X.T)).A + x_squared_norms
        closest_dist_sq = x_squared_norms[indices[0]] - 2 * sdm.dot_product_mkl(X[centroid_id], (X.T)).A + x_squared_norms
        current_pot = closest_dist_sq @ sample_weight
        n_local_trials = 2 + int(np.log(self.n_clusters))

        X_csc = sparse.csc_matrix(X)

        # Scikit-learn Kmeans++ implementation
        for i, c in enumerate(range(1, self.n_clusters)):
            start_iter = time.time()
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            rand_vals = random_state.uniform(size=n_local_trials) * current_pot
            candidate_ids = np.searchsorted(
                stable_cumsum(sample_weight * closest_dist_sq), rand_vals
            )
            # XXX: numerical imprecision can result in a candidate_id out of range
            np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

            # Compute distances to center candidates
            # This one slows down computing process
            # candidates = sparse.csc_matrix(X[candidate_ids])
            # distance_to_candidates = x_squared_norms[candidate_ids][:, np.newaxis] - (2 * candidates.dot(X.T)).A + x_squared_norms[np.newaxis, :]

            # We need to conduct X.candidates instead of candidates.X (both stored in csc)
            # candidates.X is much faster compared to X.candidates
            # The scipy performance on these 2 operation is difference, possibly due to the loop for columns in latter matrix
            candidates = X[candidate_ids]
            # distance_to_candidates = x_squared_norms[:, np.newaxis] - (2 * X_csc.dot(candidates.T)).A + x_squared_norms[candidate_ids][np.newaxis, :]

            distance_to_candidates = x_squared_norms[:, np.newaxis] - 2 * sdm.dot_product_mkl(X_csc, candidates.T).A + x_squared_norms[candidate_ids][np.newaxis, :]
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
            # centroids[c] = X[best_candidate]
            centroids.append(X[best_candidate])
            indices[c] = best_candidate

            end_iter = time.time()
            # print(f"Time to init iteration: {i}", end_iter - start_iter)

        return vstack(centroids), indices

    def _update_centroids(self, X_csr, labels):
        binary_matrix = np.zeros((self.centroids.shape[0], X_csr.shape[0]))
        mask = np.ones(self.centroids.shape[0])

        for i in range(self.n_clusters):
            idx = np.where(labels == i)[0]
            if len(idx) > 0:
                binary_matrix[i][idx] = 1 / len(idx)
            else:
                mask[i] = 0

        binary_matrix = sparse.csr_matrix(binary_matrix)

        centroids = sdm.dot_product_mkl(binary_matrix, X_csr, dense=self.isDense)
        return centroids, mask


    def _update_centroids_fixsegment(self, X_csr, labels):
        binary_matrix = np.zeros((self.centroids.shape[0], X_csr.shape[0]))
        mask = np.ones(self.centroids.shape[0])

        for i in range(self.n_clusters):
            idx = np.where(labels == i)[0]
            if len(idx) > 0:
                binary_matrix[i][idx] = 1 / len(idx)
            else:
                mask[i] = 0

        binary_matrix = sparse.csr_matrix(binary_matrix)
        binary_matrix.indices = binary_matrix.indices.copy()
        X_csr.indices = X_csr.indices.copy()

        centroids = sdm.dot_product_mkl(X_csr.T, binary_matrix.T, dense=self.isDense)
        return centroids.T, mask

    def _converged(self, old_centroids, c_squared_norms_old, c_squared_norm):
        if self.isDense:
            centroidsDiff = 2 * (self.centroids * old_centroids).sum(axis=1)[:, np.newaxis]
        else:
            centroidsDiff = 2 * self.centroids.multiply(old_centroids).sum(axis=1)

        tol = c_squared_norms_old[:, np.newaxis] - centroidsDiff + c_squared_norm[:, np.newaxis]
        tol = np.clip(tol, 0, None)
        tol = np.sum(tol)
        return tol <= self.tol 

    def predict(self, X):
        return 


# def _build_tree_mix(label_representation: sparse.csr_matrix, label_map: np.ndarray, d: int, K: int, dmax: int, exec_dict: dict, counter_dict: dict) -> Node:
#     """Builds the tree recursively by kmeans clustering.

#     Args:
#         label_representation (sparse.csr_matrix): A matrix with dimensions number of classes under this node * number of features.
#         label_map (np.ndarray): Maps 0..label_representation.shape[0] to the original label indices.
#         d (int): Current depth.
#         K (int): Maximum degree of nodes in the tree.
#         dmax (int): Maximum depth of the tree.

#     Returns:
#         Node: root of the (sub)tree built from label_representation.
#     """
#     samples = label_representation.shape[0]

#     if d >= dmax or samples <= K:
#         return Node(label_map=label_map, children=[])

#     # stack_size = len(inspect.stack())
#     # print(f"Current stack size: {stack_size} frames, Approx. Memory: {stack_size * sys.getsizeof(n)} bytes")

#     start = time.time()
#     metalabels = (
#         MixSparseKMeans(
#             d,
#             K,
#             random_state=np.random.randint(2**31 - 1),
#             max_iter=300,
#             tol=0.0001
#         )
#         .fit(label_representation)
#         .labels_
#     )
#     # metalbales should be np.ndarray
#     end = time.time()
    
#     exec_time = end - start
#     print(f"Total time for clustering node at level {d}: ", exec_time)
#     if d not in exec_dict:
#         exec_dict[d] = 0

#     if d not in counter_dict:
#         counter_dict[d] = []

#     exec_dict[d] += exec_time
#     counter_dict[d].append(samples)

#     children = []
#     for i in range(K):
#         # Why indexing cause numpy.matrix
#         child_representation = label_representation[metalabels == i]
#         child_map = label_map[metalabels == i]

#         # if d == 0 and child_representation.shape[0] > 100:
#         #     # Save the CSR matrix to a .npz file
#         #     sparse.save_npz(f'./data/extract/child_representation_{child_representation.shape[0]}_{i}.npz', child_representation)
            
#         child = _build_tree_mix(child_representation, child_map, d + 1, K, dmax, exec_dict, counter_dict)
#         children.append(child)

#     return Node(label_map=label_map, children=children)