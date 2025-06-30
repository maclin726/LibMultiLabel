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

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import math
import pickle
import numpy as np
import scipy.sparse as sparse
import libmultilabel.linear as linear
from liblinear.liblinearutil import predict

from time import time
from functools import partial
from docopt import docopt
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import minimize
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, feature_in, head_size):
        super().__init__()
        self.key = nn.Linear(feature_in, head_size, bias=False)
        self.query = nn.Linear(feature_in, head_size, bias=False)
        self.value = nn.Linear(feature_in, head_size, bias=False)
        self.head_size = head_size

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weight = q @ k.T
        weight = torch.softmax(weight, dim=-1)
        weight = weight.detach().numpy()
        return weight
    

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

# def metrics_in_batches(X, y, preds, unseen_labels, metric_list, save_path, **kargs_for_predictors):
#     batch_size = 128
#     num_instances = X.shape[0]
#     num_batches = math.ceil(num_instances / batch_size)
#     metrics = linear.get_metrics(
#         metric_list, 
#         num_classes=y.shape[1],
#         unseen_labels=unseen_labels
#     )
    
#     for i in range(num_batches):
#         target = y[i * batch_size : (i + 1) * batch_size].toarray()
#         metrics.update(preds[i], target)
#     return metrics.compute()
        

def metrics_in_batches(X, y, predictor, unseen_labels, metric_list, save_path, **kargs_for_predictors):
    batch_size = 16
    num_instances = X.shape[0]
    num_batches = math.ceil(num_instances / batch_size)
    print("testing")
    metrics = linear.get_metrics(
        metric_list, 
        num_classes=y.shape[1],
        unseen_labels=unseen_labels
    )
    # print(num_batches)
    # exit()
    for i in range(num_batches):
        # orignal
        preds = predictor.predict_on_all_label(
            X[i * batch_size : (i + 1) * batch_size], save_path=save_path, num_batches=num_batches, batch_num = i,
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
        self.strategy = strategy
        self.supervised_model = supervised_model
        self.label_feature = label_feature
        self.unseen_labels = np.setdiff1d(all_label_map, seen_labels)
        assert (
            self.unseen_labels.shape[0] + self.seen_labels.shape[0] 
            == self.all_label_map.shape[0]
        ), "The set of seen labels should be a subset of all labels."

        
        self.seen_label_feature = label_feature[self.seen_labels]
        self.unseen_label_feature = label_feature[self.unseen_labels]
        # print("seen label feature:",self.seen_label_feature.shape)
        self.label_neighbors = self.get_kneighbors() # (n_labels, n_neighbors)
        

    def get_kneighbors(self, n_neighbors=5):
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(self.seen_label_feature)
        return neigh.kneighbors(self.label_feature, return_distance=False)

    def predict_values_on_seen_label(self, x):
        preds = self.supervised_model.predict_values(x)
        return preds

    def predict_values_on_unseen_label(self, x):
        preds = predict_values_by_tfidf(x, self.unseen_label_feature)
        return preds

    def predict_on_all_label(self, x, alpha, beta, proxy_type, save_path=None, num_batches=None, batch_num=None):
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
        
        save_path = os.path.dirname(save_path)
        n = num_batches
        # np.save(f"{save_path}/s_hat_seen.npy", s_hat_seen)
        # np.save(f"{save_path}/seen_label_doc_sim.npy", seen_label_doc_sim)
        # np.save(f"{save_path}/unseen_label_doc_sim.npy", unseen_label_doc_sim)
        # check if the files exists
        # for i in range(n-1, -1, -1):
        #     
        #     if os.path.exists(f"{save_path}/s_hat_seen_{i}.npy"):
        #         continue
        #     else:
        #         break
        
        # check if the last files exsist, if all of them exists, good, load them
        if os.path.exists(f"{save_path}/s_hat_seen_{batch_num}.npy"):
            print("loading found")
            print(f"loading the {batch_num} batch")
            s_hat_seen = np.load(f"{save_path}/s_hat_seen_{batch_num}.npy")
            seen_label_doc_sim = np.load(f"{save_path}/seen_label_doc_sim_{batch_num}.npy")
            unseen_label_doc_sim = np.load(f"{save_path}/unseen_label_doc_sim_{batch_num}.npy")
        else:
            start_time = time.time()
            s_hat_seen = self.predict_values_on_seen_label(x)
            s_hat_seen = 1 / (1 + np.exp(-s_hat_seen))
            seen_label_doc_sim = predict_values_by_tfidf(x, self.seen_label_feature)
            unseen_label_doc_sim = predict_values_by_tfidf(x, self.unseen_label_feature)
            end_time = time.time()
            print(f"predicting on batch {batch_num} took {end_time - start_time:.2f} seconds")
            # save the results
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.save(f"{save_path}/s_hat_seen_{batch_num}.npy", s_hat_seen)
                np.save(f"{save_path}/seen_label_doc_sim_{batch_num}.npy", seen_label_doc_sim)
                np.save(f"{save_path}/unseen_label_doc_sim_{batch_num}.npy", unseen_label_doc_sim)
            
            
        # s_hat_seen = self.predict_values_on_seen_label(x)
        # s_hat_seen = 1 / (1 + np.exp(-s_hat_seen))
        # seen_label_doc_sim = predict_values_by_tfidf(x, self.seen_label_feature)
        # unseen_label_doc_sim = predict_values_by_tfidf(x, self.unseen_label_feature)
            # np.save(f"{save_path}/s_hat_seen.npy", s_hat_seen)
            # np.save(f"{save_path}/seen_label_doc_sim.npy", seen_label_doc_sim)
            # np.save(f"{save_path}/unseen_label_doc_sim.npy", unseen_label_doc_sim)
            
        # seen_label_doc_sim = \
        #     predict_values_by_tfidf(x, self.seen_label_feature)
        # unseen_label_doc_sim = \
        #     predict_values_by_tfidf(x, self.unseen_label_feature)
        # s_hat_seen = self.predict_values_on_seen_label(x)
        # s_hat_seen = 1 / (1 + np.exp(-s_hat_seen))
        
        # print shapes
        print(f"s_hat_seen shape: {s_hat_seen.shape}"
              f", seen_label_doc_sim shape: {seen_label_doc_sim.shape}"
              f", unseen_label_doc_sim shape: {unseen_label_doc_sim.shape}"
              f", x shape: {x.shape}")
        preds[:,self.seen_labels] = \
            alpha * s_hat_seen + (1-alpha) * seen_label_doc_sim
            
        
        proxy = np.zeros((x.shape[0], self.unseen_labels.shape[0]))
        
        if proxy_type == "zero":
            # preds[:,self.unseen_labels] = 0
            # return preds
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
            
        elif proxy_type == "weighted_avg":
            nearest_seen_labels = self.label_neighbors[self.unseen_labels, :3]
            distances, _ = NearestNeighbors(n_neighbors=3).fit(self.seen_label_feature).kneighbors(self.unseen_label_feature)
            similarities = 1 / (distances + 1e-10)
            weight_sum = np.sum(similarities, axis=1, keepdims=True)
            weights = similarities / weight_sum
            proxy = np.sum(preds[:,nearest_seen_labels] * weights[None, :, :], axis=2)
            
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
            
            k = key(seen_label_tensor)
            q = query(unseen_label_tensor)
            
            weight = q @ k.T
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

        preds[:,self.unseen_labels] = \
            beta * proxy + (1-beta) * unseen_label_doc_sim
        # get average of preds[:,self.unseen_labels] and preds[:,self.seen_labels] and plot them
        
        
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

    start_time = time.time()
    # Load models and data
    with open(model_path, "rb") as _F:
        model = pickle.load(_F)['model']
    end_time = time.time()
    print(f"Model loaded in {end_time - start_time:.2f} seconds.")

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
    # proxy_types = ["attention"]
    proxy_types = ["zero","avg","insert_closest"]
    
    alphas = np.arange(0, 1.1, 0.1)
    betas = np.arange(0, 1.1, 0.1)
    # alphas = [1]
    # betas = [1]
    
    metric_list = [
        "P@1", "P@3", "P@5",
        "R@1", "R@3", "R@5",
        "NDCG@1", "NDCG@3", "NDCG@5",
         "ZSR@10", "ZSR@50", "ZSR@100"]
    
    for proxy_type in proxy_types:
        logs = {'alpha': [], 'beta': [], 'full_metrics': []}
        for alpha in alphas:
            for beta in betas:
                if alpha < beta:
                    continue
                metric_dict = metrics_in_batches(
                    X_test, y_test, mixed_predictor, unseen_labels, metric_list, model_path,
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

