"""
Usage:
    model_predict.py --model_path=<model_path> --train_instance_data_path=<train_data> --test_instance_data_path=<test_data> --label_data_path=<label_data>

Options:
    --model_path=<model_path>                Path to the model file (string).
    --train_instance_data_path=<train_data>  Path to the training instance data file (string).
    --test_instance_data_path=<test_data>    Path to the testing instance data file (string).
    --label_data_path=<label_data>           Path to the label data file (string).
"""

import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import pickle
import libmultilabel.linear as linear
import numpy as np

from functools import partial
from docopt import docopt
from sklearn.datasets import load_svmlight_file


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

class MixedPredictor:
    """A mixed predictor can predict on seen and unseen labels."""

    def __init__(
            self,
            all_label_map,
            seen_label_map,
            supervised_model,
            label_feature,
        ):
        self.all_label_map = all_label_map
        self.seen_label_map = seen_label_map
        self.supervised_model = supervised_model
        self.label_feature = label_feature

        self.unseen_label_map = np.setdiff1d(all_label_map, seen_label_map)
        self.unseen_label_feature = label_feature[self.unseen_label_map]

        assert self.unseen_label_map.shape[0] + self.seen_label_map.shape[0] == self.all_label_map.shape[0], "The set of seen labels should be a subset of all labels."

    def predict_values_on_seen_label(self, x):
        preds = self.supervised_model.predict_values(x)
        return preds

    def predict_values_on_unseen_label(self, x):
        preds = predict_values_by_tfidf(x, self.unseen_label_feature)
        return preds

    def predict_on_all_label(self, x):
        raise NotImplementedError

def main():
    # Parse command-line arguments
    args = docopt(__doc__)
    
    # Accessing the arguments
    model_path = args['--model_path']
    train_data_path = args['--train_instance_data_path']
    test_data_path = args['--test_instance_data_path']
    label_data_path = args['--label_data_path']
    
    # Print the parsed arguments (for debugging purposes)
    print(f"Model Path: {model_path}")
    print(f"Train Instance Data Path: {train_data_path}")
    print(f"Test Instance Data Path: {test_data_path}")
    print(f"Label Data Path: {label_data_path}")
    
    print("Start processing...")

    # Load models and data
    X_train, y_train = load_svm_data(train_data_path)
    X_test, y_test = load_svm_data(test_data_path, n_features=X_train.shape[1])
    X_label, _ = load_svm_data(label_data_path, n_features=X_train.shape[1])
    with open(model_path, "rb") as _F:
        model = pickle.load(_F)['model']

    # Init a mixed predictor
    mixed_predictor = MixedPredictor(
            np.arange(X_label.shape[0], dtype="int"),
            model.root.label_map,
            model,
            X_label,
            )

    # Predict seen and unseen labels on the test data
    preds_on_seen_label = mixed_predictor.predict_values_on_seen_label(X_test)
    preds_on_unseen_label = mixed_predictor.predict_values_on_unseen_label(X_test)
    print(preds_on_seen_label.shape)
    print(preds_on_unseen_label.shape)

if __name__ == "__main__":
    main()

