"""
Usage:
    model_predict.py --model_path=<model_path> \
    --train_instance_data_path=<train_data> \
    --test_instance_data_path=<test_data> \
    --label_feature_path=<label_data> \
    --run_name=<run_name_str>

Options:
    --model_path=<model_path>                Path to the model file (string).
    --train_instance_data_path=<train_data>  Path to the training instance data file (string).
    --test_instance_data_path=<test_data>    Path to the testing instance data file (string).
    --label_feature_path=<label_features>    Path to the label data file (string).
    --run_name=<run_name_str>                Name of the run (string).
"""
import json
import sys
import os

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


def metrics_in_batches(X, y, predictor, unseen_labels, metric_list, label_pos_count, num_instances_train, **kargs_for_predictors):
    batch_size = 256
    num_instances = X.shape[0]
    num_batches = math.ceil(num_instances / batch_size)
    
    metrics = linear.get_metrics(
        metric_list, 
        num_classes=y.shape[1],
        unseen_labels=unseen_labels,
        label_pos_counts=label_pos_count,
        num_instances=num_instances_train
    )

    for i in range(num_batches):
        # tfidf only
        preds = predictor.predict_values_on_all_label(
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
            label_feature,
        ):
        self.all_label_map = all_label_map
        self.seen_labels = seen_labels
        self.label_feature = label_feature
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
        # preds = self.supervised_model.predict_values(x)
        # return preds
        pass

    def predict_values_on_unseen_label(self, x):
        preds = predict_values_by_tfidf(x, self.unseen_label_feature)
        return preds
    
    def predict_values_on_all_label(self, x):
        # predict on all labels but just use tfidf
        preds = predict_values_by_tfidf(x, self.label_feature)
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
            predict_values_by_tfidf(x, self.seen_label_feature)
        unseen_label_doc_sim = \
            predict_values_by_tfidf(x, self.unseen_label_feature)
            
        s_hat_seen = self.predict_values_on_seen_label(x)
        
        k = 60
        ranks_s_hat_seen = np.argsort(np.argsort(-s_hat_seen, axis=1), axis=1)
        ranks_seen_label_doc_sim = np.argsort(np.argsort(-seen_label_doc_sim, axis=1), axis=1)
        ranks_unseen_label_doc_sim = np.argsort(np.argsort(-unseen_label_doc_sim, axis=1), axis=1)
        
        # filter ranks for seen labels
        # ranks_seen_filtered = ranks_seen[:, self.seen_labels]
        combined_ranks_seen = alpha * ranks_s_hat_seen + (1 - alpha) * ranks_seen_label_doc_sim

# Re-rank the combined scores to get final ranks
        final_ranks_seen = np.argsort(np.argsort(combined_ranks_seen, axis=1), axis=1)
        preds[:, self.seen_labels] = -final_ranks_seen
        # preds[:, self.seen_labels] = (
        #     alpha * ranks_s_hat_seen + (1 - alpha) * ranks_seen_label_doc_sim
        # )
        
        # preds[:, self.seen_labels] = (
        #     alpha * (1 / (ranks_s_hat_seen + k)) + (1 - alpha) * (1 / (ranks_seen_label_doc_sim + k))
        # )
        
        proxy = np.zeros((x.shape[0], self.unseen_labels.shape[0]))
        
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
            # value = nn.Linear(self.seen_label_feature.shape[1], D_proj, bias=False)
            
            seen_label_tensor_dense = self.seen_label_feature.toarray()
            unseen_label_tensor_dense = self.unseen_label_feature.toarray()
            # x_dense = x.toarray()
            
            seen_label_tensor = torch.tensor(seen_label_tensor_dense, dtype=torch.float32)
            unseen_label_tensor = torch.tensor(unseen_label_tensor_dense, dtype=torch.float32)
            # x = torch.tensor(x_dense, dtype=torch.float32)
            
            k_attention = key(seen_label_tensor)
            q = query(unseen_label_tensor)
            
            weight = q @ k_attention.T
            weight = torch.softmax(weight, dim=-1)
            weight = weight.detach().numpy()

            proxy = preds[:,self.seen_labels] @ weight.T # goal is (256, 74)
            
        elif proxy_type == "attention_test":
            D_proj = 1024
            num_heads = 8
            head_size = D_proj // num_heads
            key_proj = nn.Linear(self.seen_label_feature.shape[1], D_proj, bias=False)
            query_proj = nn.Linear(self.unseen_label_feature.shape[1], D_proj, bias=False)
            seen_dense = self.seen_label_feature.toarray()
            unseen_dense = self.unseen_label_feature.toarray()
            seen_label_tensor = torch.tensor(seen_dense, dtype=torch.float32)
            unseen_label_tensor = torch.tensor(unseen_dense, dtype=torch.float32)
            
            k = key_proj(seen_label_tensor)
            q = query_proj(unseen_label_tensor)
            k = k.view(-1, num_heads, head_size).transpose(0, 1)
            q = q.view(-1, num_heads, head_size).transpose(0, 1)
            
            weight = q @ k.transpose(-2, -1) / math.sqrt(head_size)
            weight = torch.softmax(weight, dim=-1)
            weight = weight.mean(dim=0)
            weight = weight.detach().numpy()
            # print(weight.shape)
            # print(preds[:,self.seen_labels].shape)
            proxy = preds[:,self.seen_labels] @ weight.T
            
        elif proxy_type == "min":
            # bad performance
            nearest_seen_labels = self.label_neighbors[self.unseen_labels, :3]
            proxy = np.min(preds[:,nearest_seen_labels], axis=2)
        else:
            raise ValueError("Unknown proxy type for unseen labels")
        
        ranks_proxy = np.argsort(np.argsort(-proxy, axis=1), axis=1)
        
        # preds[:,self.unseen_labels] = \
        #     beta * (1 / (ranks_proxy + k)) + (1-beta) * (1 /  (ranks_unseen_label_doc_sim + k))
            
        # preds[:,self.unseen_labels] = \
        #     beta * ranks_proxy + (1-beta) * ranks_unseen_label_doc_sim
        # get average of preds[:,self.unseen_labels] and preds[:,self.seen_labels] and plot them
        
        combined_ranks_unseen = beta * ranks_proxy + (1 - beta) * ranks_unseen_label_doc_sim

        # Re-rank the combined values to get final ranks
        final_ranks_unseen = np.argsort(np.argsort(combined_ranks_unseen, axis=1), axis=1)

        # Assign scores as negative ranks so lower ranks = higher scores
        preds[:, self.unseen_labels] = -final_ranks_unseen
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
    
    # Print the parsed arguments (for debugging purposes)
    print(f"Model Path: {model_path}")
    print(f"Train Instance Data Path: {train_data_path}")
    print(f"Test Instance Data Path: {test_data_path}")
    print(f"Label Feature Path: {label_feature_path}")
    
    print("Start processing...")

    # Load models and data
    # with open(model_path, "rb") as _F:
    #     model = pickle.load(_F)['model']

    # model.flat_model.weights.shape: (n_features, n_classifiers)
    # if model.name == '1vsrest':
    #     X_train, y_train = load_svm_data(
    #         train_data_path, n_features=model.weights.shape[0])
    # else:
    #     X_train, y_train = load_svm_data(
    #         train_data_path, n_features=model.flat_model.weights.shape[0])
        
    
    X_train, y_train = load_svm_data(
        train_data_path, n_features=3065363)

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
    print("seen labels", seen_labels.shape, 
        "\nunseen labels", unseen_labels.shape)
    # Init a mixed predictor
    mixed_predictor = MixedPredictor(
        np.arange(X_label.shape[0], dtype="int"),
        seen_labels,
        X_label
    )
    total_num_labels = seen_labels.shape[0] + unseen_labels.shape[0]
    print(X_label.shape)
    print(total_num_labels)
    exit()
    
    lj = np.zeros((total_num_labels,), dtype=int)
    # at the jth label, how many instances
    y_train_coo = y_train.tocoo()
    for i, j, v in zip(y_train_coo.row, y_train_coo.col, y_train_coo.data):
        lj[j] += 1
    
    num_instances = X_train.shape[0]
    metric_list = [
        "P@1", "P@3", "P@5",
        "R@1", "R@3", "R@5",
        "NDCG@1", "NDCG@3", "NDCG@5",
         "ZSR@10", "ZSR@50", "ZSR@100",
         "PSP@1", "PSP@3", "PSP@5"]
    print("hello")
    metric_dict = metrics_in_batches(
        X_test, y_test, mixed_predictor, unseen_labels, label_pos_count=lj, num_instances_train=num_instances,
        metric_list=metric_list
    )
    print("hello2")

    # store the metrics to a list then append tto full_metrics

    print(linear.tabulate_metrics(
            metric_dict, "MIMIC-TFIDF"), flush=True)
        # save logs as json

if __name__ == "__main__":
    main()

