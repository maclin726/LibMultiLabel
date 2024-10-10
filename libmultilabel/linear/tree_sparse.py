import sklearn
import numpy as np
import scipy.sparse as sparse
import time
from tqdm import tqdm

from .tree import Node, TreeModel, _train_node, _flatten_model
# from .sparse_cluster import SparseKMeans

import numbers
from sklearn.utils.extmath import row_norms, stable_cumsum

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


class SparseKMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = []
        self.random_state = check_random_state(random_state)

    def fit(self, X: sparse.csr_matrix):
        
        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)

        print("Total Samples: ", X.shape[0])

        # Get the square norm of each sample in x
        x_squared_norms = row_norms(X, squared=True)

        # Randomly initialize centroids from the data points
        self.centroids, indices = self._init_centroids(X, x_squared_norms,random_state=self.random_state)
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
            
            # Calculate X*C.T in advance
            if not isinstance(self.centroids, sparse.csr_matrix):
                self.centroids = sparse.csr_matrix(self.centroids)

            # Calculate the distance from points to clusters (X*X.T - 2X*C + C*C.T)
            XC = (2 * X.dot(self.centroids.T))

            distances_to_clusters = x_squared_norms[:, np.newaxis] - XC.A + c_squared_norms[np.newaxis, :]
            distances_to_clusters = np.clip(distances_to_clusters, 0, None)

            # Assign sample to its closest centroids.
            labels = np.argmin(distances_to_clusters, axis=1)

            # Step 2: Calculate new centroids
            old_centroids = self.centroids
            self.centroids, mask = self._update_centroids(X_csr, labels)

            self.centroids[mask==0] = old_centroids[mask==0]

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

    def _init_centroids(self, X, x_squared_norms, random_state, sample_weight=None):
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
        closest_dist_sq = x_squared_norms[indices[0]] - (2 * X[centroid_id].dot(X.T)).A + x_squared_norms
        current_pot = closest_dist_sq @ sample_weight
        n_local_trials = 2 + int(np.log(self.n_clusters))

        X_csc = sparse.csc_matrix(X)

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
            # This one slows down computing process
            # candidates = sparse.csc_matrix(X[candidate_ids])
            # distance_to_candidates = x_squared_norms[candidate_ids][:, np.newaxis] - (2 * candidates.dot(X.T)).A + x_squared_norms[np.newaxis, :]

            # We need to conduct X.candidates instead of candidates.X (both stored in csc)
            # candidates.X is much faster compared to X.candidates
            # The scipy performance on these 2 operation is difference, possibly due to the loop for columns in latter matrix
            candidates = X[candidate_ids]
            distance_to_candidates = x_squared_norms[:, np.newaxis] - (2 * X_csc.dot(candidates.T)).A + x_squared_norms[candidate_ids][np.newaxis, :]
            distance_to_candidates = distance_to_candidates.T

        #     distance_to_candidates = _euclidean_distances(
        #     X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
        # )

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

        return sparse.csr_matrix(centroids), indices


    def _update_centroids(self, X_csr, labels):
        binary_matrix = np.zeros((self.centroids.shape[0], X_csr.shape[0]))
        mask = np.ones((self.centroids.shape[0], 1))

        for i in range(self.n_clusters):
            idx = np.where(labels == i)[0]
            if len(idx) > 0:
                binary_matrix[i][idx] = 1 / len(idx)
            else:
                mask[idx] = 0

        binary_matrix = sparse.csr_matrix(binary_matrix)
        centroids = X_csr.T.dot(binary_matrix.T)

        return centroids.T, mask

    def _converged(self, old_centroids, c_squared_norms_old, c_squared_norm):
        tol = c_squared_norms_old[:, np.newaxis] - 2 * self.centroids.multiply(old_centroids).sum(axis=1) + c_squared_norm[:, np.newaxis]
        # print(tol.shape)
        tol = np.clip(tol, 0, None)
        tol = np.sum(tol)
        return tol <= self.tol 

    def predict(self, X):
        return 


def _build_tree_sparse(label_representation: sparse.csr_matrix, label_map: np.ndarray, d: int, K: int, dmax: int) -> Node:
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
    if d >= dmax or label_representation.shape[0] <= K:
        return Node(label_map=label_map, children=[])

    start = time.time()
    metalabels = (
        SparseKMeans(
            K,
            random_state=np.random.randint(2**31 - 1),
            max_iter=300,
            tol=0.0001
        )
        .fit(label_representation)
        .labels_
    )
    # metalbales should be np.ndarray
    end = time.time()
    if d == 1:
        print(f"Total time for clustering node at level {d}: ", end - start)

    children = []
    for i in range(K):
        # Why indexing cause numpy.matrix
        child_representation = label_representation[metalabels == i]
        child_map = label_map[metalabels == i]

        # if d == 0 and child_representation.shape[0] > 100:
        #     # Save the CSR matrix to a .npz file
        #     sparse.save_npz(f'./data/extract/child_representation_{child_representation.shape[0]}_{i}.npz', child_representation)
            
        child = _build_tree_sparse(child_representation, child_map, d + 1, K, dmax)
        children.append(child)

    return Node(label_map=label_map, children=children)

def train_tree_sparse(
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    options: str = "",
    K=100,
    dmax=10,
    verbose: bool = True,
) -> TreeModel:
    """Trains a linear model for multiabel data using a divide-and-conquer strategy.
    The algorithm used is based on https://github.com/xmc-aalto/bonsai.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        K (int, optional): Maximum degree of nodes in the tree. Defaults to 100.
        dmax (int, optional): Maximum depth of the tree. Defaults to 10.
        verbose (bool, optional): Output extra progress information. Defaults to True.

    Returns:
        A model which can be used in predict_values.
    """
    label_representation = (y.T * x).tocsr()
    label_representation = sklearn.preprocessing.normalize(label_representation, norm="l2", axis=1)

    # Build a tree
    # In case the node have more than K labels => it continues to cluster else stop
    # Note that the number of labels per node is not the same.
    start = time.time()
    root = _build_tree_sparse(label_representation, np.arange(y.shape[1]), 0, K, dmax)
    end = time.time()
    print("Clustering time: {:10.2f}\n".format(end - start))

    num_nodes = 0

    def count(node):
        nonlocal num_nodes
        num_nodes += 1

    root.dfs(count)

    pbar = tqdm(total=num_nodes, disable=not verbose)

    def visit(node):
        relevant_instances = y[:, node.label_map].getnnz(axis=1) > 0
        _train_node(y[relevant_instances], x[relevant_instances], options, node)
        pbar.update()

    root.dfs(visit)
    pbar.close()

    flat_model, weight_map = _flatten_model(root)
    return TreeModel(root, flat_model, weight_map)