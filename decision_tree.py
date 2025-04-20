import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple

"""
Notice:
    1) You can't add any additional package
    2) You can add or remove any function "except" fit, _build_tree, predict
    3) You can ignore the suggested data type if you want
"""

class ConvNet(nn.Module): # Don't change this part!
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        x = self.model(x)
        return x
    
class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.tree = self._build_tree(X, y)
        self.progress.close()

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        # (TODO) Grow the decision tree and return it
        raise NotImplementedError

    def predict(self, X: pd.DataFrame)->np.ndarray:
        # (TODO) Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
        raise NotImplementedError

    def _predict_tree(self, x, tree_node):
        # (TODO) Recursive function to traverse the decision tree
        raise NotImplementedError

    def _split_data(X: pd.DataFrame, y: np.ndarray, feature_index: int, threshold: float):
        # (TODO) split one node into left and right node 
        raise NotImplementedError
        return left_dataset_X, left_dataset_y, right_dataset_X, right_dataset_y

    def _best_split(X: pd.DataFrame, y: np.ndarray):
        # (TODO) Use Information Gain to find the best split for a dataset
        raise NotImplementedError
        return best_feature_index, best_threshold

    def _entropy(y: np.ndarray)->float:
        # (TODO) Return the entropy
        raise NotImplementedError

def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and labels
    raise NotImplementedError
    return features, labels

def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and path of the images
    raise NotImplementedError
    return features, paths