"""
Usage:
    model_predict.py --model_path=<model_path> \
    --train_instance_data_path=<train_data> \
    --test_instance_data_path=<test_data> \
    --label_feature_path=<label_data> \
    --run_name=<run_name_str> \
    --strategy=<strategy_str>

Options:
    --model_path=<model_path>                Path to the model file (string).
    --train_instance_data_path=<train_data>  Path to the training instance data file (string).
    --test_instance_data_path=<test_data>    Path to the testing instance data file (string).
    --label_feature_path=<label_features>    Path to the label data file (string).
    --run_name=<run_name_str>                Name of the run (string).
    --strategy=<strategy_str>                Strategy for prediction (string).
"""
import json
import sys
import os
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import math
import pickle
import numpy as np
import scipy.sparse as sparse
import libmultilabel.linear as linear

from time import time
from functools import partial
from docopt import docopt
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.nn.functional as F

def load_svm_data(file_path, /, *args, **keywords):
    """
    A wrapper of load_svmlight_file with arguments with changed default values:
        - multilabel=True
        - zero_based=False

    Note: If you pass new values to above arguments, the function's behavior will follow the new values you provided.
    """
    _wrapped_func = partial(load_svmlight_file, multilabel=True, zero_based=False)
    return _wrapped_func(file_path, *args, **keywords)

def predict_values_by_tfidf(instance_tfidf, label_tfidf):
    """Calculates the similarity scores between instance and label tfidf vectors.

        Args:
            instance_tfidf (sparse.csr_matrix): A matrix of shape (#instances, #features).
            label_tfidf (sparse.csr_matrix): A matrix of shape (#labels, #features).

        Returns:
            np.array: A matrix of shape (#instances, #labels).
    """
    
    return (instance_tfidf @ label_tfidf.T).toarray()

def predict_values_by_tfidf_gpu(instance_tfidf, label_tfidf):
    """
    Calculates the similarity scores between instance and label tfidf vectors using GPU.

    Args:
        instance_tfidf (scipy.sparse.csr_matrix): Shape (#instances, #features).
        label_tfidf (scipy.sparse.csr_matrix): Shape (#labels, #features).

    Returns:
        np.ndarray: Similarity score matrix of shape (#instances, #labels).
    """
    # Convert SciPy sparse matrices to CuPy sparse
    instance_gpu = cpx_sparse.csr_matrix(instance_tfidf)
    label_gpu = cpx_sparse.csr_matrix(label_tfidf)

    # Sparse matrix multiplication on GPU
    score_gpu = instance_gpu @ label_gpu.T
    score_dense_gpu = score_gpu.toarray()     
    # Bring result back to CPU (as numpy array)
    return cp.asnumpy(score_dense_gpu)


def metrics_in_batches(X, y, predictor, unseen_labels, metric_list, **kargs_for_predictors):
    batch_size = 256
    num_instances = X.shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = linear.get_metrics(
        metric_list, 
        num_classes=y.shape[1],
        unseen_labels=unseen_labels
    )

    for i in range(num_batches):
        # orignal
        preds = predictor.predict_on_all_label(
            X[i * batch_size : (i + 1) * batch_size], 
            **kargs_for_predictors
        )
        
        
        target = y[i * batch_size : (i + 1) * batch_size].toarray()
        # compare
        metrics.update(preds, target)
    
    return metrics.compute()

class MixedPredictor:
    """A mixed predictor can predict on seen and unseen labels."""

    def __init__(
            self,
            all_label_map,
            seen_labels,
            supervised_model,
            label_feature,
            strategy='raw'
        ):
        self.all_label_map = all_label_map
        self.seen_labels = seen_labels
        self.supervised_model = supervised_model
        self.label_feature = label_feature
        self.strategy = strategy
        self.unseen_labels = np.setdiff1d(all_label_map, seen_labels)
        assert (
            self.unseen_labels.shape[0] + self.seen_labels.shape[0] 
            == self.all_label_map.shape[0]
        ), "The set of seen labels should be a subset of all labels."

        
        self.seen_label_feature = label_feature[self.seen_labels]
        self.unseen_label_feature = label_feature[self.unseen_labels]
        self.label_neighbors = self.get_kneighbors() # (n_labels, n_neighbors)
        

    def get_kneighbors(self, n_neighbors=5):
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(self.seen_label_feature)
        return neigh.kneighbors(self.label_feature, return_distance=False)

    def predict_values_on_seen_label(self, x):
        preds = self.supervised_model.predict_values(x)
        return preds

    def predict_values_on_unseen_label(self, x):
        preds = predict_values_by_tfidf_gpu(x, self.unseen_label_feature)
        return preds
    
    def predict_values_on_all_label(self, x):
        # predict on all labels but just use tfidf
        preds = np.zeros((x.shape[0], self.all_label_map.shape[0]))
        all_label = np.concatenate(
            (self.seen_label_feature, self.unseen_label_feature), axis=0
        )
        all_label = sparse.vstack(all_label)
        all_label = all_label.toarray()
        preds = predict_values_by_tfidf_gpu(x, all_label)
        return preds
        
        

    def predict_on_all_label(self, x, alpha, beta, proxy_type):
        """
        Predict the values for all labels based on the unified framework,
        given as
            scores_seen = alpha * s_hat_seen + (1-alpha) * label_doc_sim_seen
            scores_unseen = beta * proxy + (1-beta) * label_doc_sim_unseen

            Args:
                alpha (float): a number in [0, 1]
                beta (float): a number in [0, 1]
                proxy_type (str): could be one of the following
                    "zero": 
                    "insert_closest":
                    "avg":
                    "min":
                    "period":

        """
        preds = np.zeros((x.shape[0], self.all_label_map.shape[0]))
        seen_label_doc_sim = \
            predict_values_by_tfidf_gpu(x, self.seen_label_feature)
        unseen_label_doc_sim = \
            predict_values_by_tfidf_gpu(x, self.unseen_label_feature)
            
        s_hat_seen = self.predict_values_on_seen_label(x)
        
        proxy = np.zeros((x.shape[0], self.unseen_labels.shape[0]))
        # print(type(s_hat_seen), type(seen_label_doc_sim), type(alpha))
        # score1 = alpha * s_hat_seen
        # score2 = (1 - alpha) * seen_label_doc_sim
        # print("score1:", type(score1), isinstance(score1, np.ndarray))
        # print("score2:", type(score2), isinstance(score2, np.ndarray))
        # combined_score = score1 + score2
        preds[:,self.seen_labels] = \
            alpha * s_hat_seen + (1-alpha) * seen_label_doc_sim
        if proxy_type == "zero":
            pass
        elif proxy_type == "insert_closest":
            nearest_seen_label = self.label_neighbors[self.unseen_labels, 0]
            sign = np.sign(
                (x @ (self.label_feature[self.unseen_labels] - \
                self.label_feature[nearest_seen_label]).T).toarray()
            )
            proxy = preds[:,nearest_seen_label] + sign * 1e-8            
        elif proxy_type == "avg":
            nearest_seen_labels = self.label_neighbors[self.unseen_labels, :3]
            # shape: (n_instances, n_unseen labels, n_nearest neighbors)
            proxy = np.average(preds[:,nearest_seen_labels], axis=2)           
        elif proxy_type == "attention":
            D_proj = 1024
            key = nn.Linear(self.seen_label_feature.shape[1], D_proj, bias=False)
            query = nn.Linear(self.unseen_label_feature.shape[1], D_proj, bias=False)
            seen_label_tensor_dense = self.seen_label_feature.toarray()
            unseen_label_tensor_dense = self.unseen_label_feature.toarray()
            seen_label_tensor = torch.tensor(seen_label_tensor_dense, dtype=torch.float32)
            unseen_label_tensor = torch.tensor(unseen_label_tensor_dense, dtype=torch.float32)
            k_attention = key(seen_label_tensor)
            q = query(unseen_label_tensor)
            
            weight = q @ k_attention.T
            weight = torch.softmax(weight, dim=-1)
            weight = weight.detach().numpy()

            proxy = preds[:,self.seen_labels] @ weight.T # goal is (256, 74)           
            
        elif proxy_type == "min":
            # bad performance
            nearest_seen_labels = self.label_neighbors[self.unseen_labels, :3]
            proxy = np.min(preds[:,nearest_seen_labels], axis=2)
        else:
            raise ValueError("Unknown proxy type for unseen labels")
        
        if self.strategy == 'rank_rrf':
            k = 60
            ranks_s_hat_seen = np.argsort(np.argsort(-s_hat_seen, axis=1), axis=1)
            ranks_seen_label_doc_sim = np.argsort(np.argsort(-seen_label_doc_sim, axis=1), axis=1)
            ranks_unseen_label_doc_sim = np.argsort(np.argsort(-unseen_label_doc_sim, axis=1), axis=1)
            # prediction for seen labels
            preds[:, self.seen_labels] = (
            alpha * (1 / (ranks_s_hat_seen + k)) + (1 - alpha) * (1 / (ranks_seen_label_doc_sim + k)))
            ranks_proxy = np.argsort(np.argsort(-proxy, axis=1), axis=1)
            preds[:,self.unseen_labels] = \
                beta * (1 / (ranks_proxy + k)) + (1-beta) * (1 /  (ranks_unseen_label_doc_sim + k))
                
        elif self.strategy == 'rank_normal':
            ranks_s_hat_seen = np.argsort(np.argsort(-s_hat_seen, axis=1), axis=1)
            ranks_seen_label_doc_sim = np.argsort(np.argsort(-seen_label_doc_sim, axis=1), axis=1)
            ranks_unseen_label_doc_sim = np.argsort(np.argsort(-unseen_label_doc_sim, axis=1), axis=1)
            
            combined_ranks_seen = alpha * ranks_s_hat_seen + (1 - alpha) * ranks_seen_label_doc_sim
            # prediction for seen labels
            preds[:, self.seen_labels] = -combined_ranks_seen
            
            ranks_proxy = np.argsort(np.argsort(-proxy, axis=1), axis=1)
            combined_ranks_unseen = beta * ranks_proxy + (1 - beta) * ranks_unseen_label_doc_sim
            # prediction for unseen labels
            preds[:, self.unseen_labels] = -combined_ranks_unseen
            
        elif self.strategy == 'raw':
            # prediction for unseen labels
            preds[:,self.unseen_labels] = \
            beta * proxy + (1-beta) * unseen_label_doc_sim
        
        
        return preds


def main():
    # Parse command-line arguments
    args = docopt(__doc__)

    # Accessing the arguments
    model_path = args['--model_path']
    train_data_path = args['--train_instance_data_path']
    test_data_path = args['--test_instance_data_path']
    label_feature_path = args['--label_feature_path']
    run_name = args['--run_name']
    strategy = args['--strategy'] if '--strategy' in args else 'raw'
    
    # Print the parsed arguments (for debugging purposes)
    print(f"Model Path: {model_path}")
    print(f"Train Instance Data Path: {train_data_path}")
    print(f"Test Instance Data Path: {test_data_path}")
    print(f"Label Feature Path: {label_feature_path}")
    
    print("Start processing...")

    # Load models and data
    with open(model_path, "rb") as _F:
        model = pickle.load(_F)['model']

    # model.flat_model.weights.shape: (n_features, n_classifiers)
    if model.name == '1vsrest':
        X_train, y_train = load_svm_data(
            train_data_path, n_features=model.weights.shape[0])
    else:
        X_train, y_train = load_svm_data(
            train_data_path, n_features=model.flat_model.weights.shape[0])

    X_test, y_test = load_svm_data(test_data_path, n_features=X_train.shape[1])
    X_label, _ = load_svm_data(label_feature_path, n_features=X_train.shape[1])

    binarizer = MultiLabelBinarizer(
        classes=np.arange(X_label.shape[0], dtype="float"), sparse_output=True)
    # binarizer = MultiLabelBinarizer(sparse_output=True)
    binarizer.fit(y_train + y_test)
    y_train = binarizer.transform(y_train)
    y_test = binarizer.transform(y_test)
    seen_labels = np.nonzero(np.sum(y_train, axis=0)[0])[1]
    unseen_labels = np.setdiff1d(
        np.arange(X_label.shape[0], dtype="int"), seen_labels)

    # Init a mixed predictor
    mixed_predictor = MixedPredictor(
        np.arange(X_label.shape[0], dtype="int"),
        seen_labels,
        model,
        X_label,
        strategy
    )
    
    # a grid search
    proxy_types = ["zero","attention","avg","insert_closest"]
    
    alphas = np.arange(0, 1.1, 0.1)
    betas = np.arange(0, 1.1, 0.1)
    
    metric_list = [
        "P@1", "P@3", "P@5", 
        "R@10", "R@20", "R@50",
        "ZSR@10", "ZSR@20", "ZSR@50"]
    
    for proxy_type in proxy_types:
        logs = {'alpha': [], 'beta': [], 'full_metrics': []}
        for alpha in alphas:
            for beta in betas:
                if alpha < beta:
                    continue
                metric_dict = metrics_in_batches(
                    X_test, y_test, mixed_predictor, unseen_labels, metric_list,
                    alpha=alpha, beta=beta, proxy_type=proxy_type)
                
                # store the metrics to a list then append to full_metrics
                full_metrics = {}
                for metric in metric_list:
                    full_metrics[metric] = metric_dict[metric]
                    
                logs['alpha'].append(alpha)
                logs['beta'].append(beta)
                logs['full_metrics'].append(full_metrics)
                
                print(linear.tabulate_metrics(
                        metric_dict, 
                        f"a={alpha} b={beta}, proxy={proxy_type} Test"), flush=True)
        # save logs as json
        
        with open(f'logs_{proxy_type}_{run_name}.json', 'w') as f:
            json.dump(logs, f)

if __name__ == "__main__":
    main()

