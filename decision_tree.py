import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple, Optional

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        return self.model(x)

class DecisionTree:
    def __init__(self, max_depth=7, min_gain=0.01):
        self.max_depth = max_depth
        self.min_gain = min_gain  # Minimum information gain to allow a split
        self.tree = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        # Stopping conditions: max depth reached or pure node
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return self._create_leaf_node(y)
        feature_idx, threshold = self._best_split(X, y)
        # If no valid split is found (gain too low or none possible), create leaf
        if feature_idx is None:
            return self._create_leaf_node(y)
        X_left, y_left, X_right, y_right = self._split_data(X, y, feature_idx, threshold)
        # Avoid empty splits
        if len(y_left) == 0 or len(y_right) == 0:
            return self._create_leaf_node(y)
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        return {'feature_idx': feature_idx, 'threshold': threshold, 'left': left_subtree, 'right': right_subtree}

    def _create_leaf_node(self, y: np.ndarray):
        counts = np.bincount(y, minlength=self.n_classes)
        return {'class': np.argmax(counts)}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._predict_tree(x, self.tree) for x in X.to_numpy()])

    def _predict_tree(self, x, node):
        if 'class' in node:
            return node['class']
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_tree(x, node['left'])
        else:
            return self._predict_tree(x, node['right'])

    def _split_data(self, X: pd.DataFrame, y: np.ndarray, feature_idx: int, threshold: float):
        left_mask = X.iloc[:, feature_idx] <= threshold
        return X[left_mask], y[left_mask], X[~left_mask], y[~left_mask]

    def _best_split(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        best_gain = -1
        best_feature, best_threshold = None, None
        parent_entropy = self._entropy(y)
        n_samples = len(y)

        # Iterate over all features (0 to 299)
        for feature_idx in range(X.shape[1]):
            values = X.iloc[:, feature_idx].values
            # Use percentiles as thresholds to reduce computation (e.g., 25th, 50th, 75th)
            thresholds = np.percentile(values, [25, 50, 75])
            for threshold in thresholds:
                left_mask = X.iloc[:, feature_idx] <= threshold
                y_left = y[left_mask]
                y_right = y[~left_mask]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                # Corrected information gain calculation
                child_entropy = (len(y_left) / n_samples * self._entropy(y_left) + 
                                len(y_right) / n_samples * self._entropy(y_right))
                gain = parent_entropy - child_entropy
                if gain > best_gain and gain >= self.min_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        # Only return a split if gain exceeds min_gain
        if best_gain >= self.min_gain:
            return best_feature, best_threshold
        return None, None

    def _entropy(self, y: np.ndarray) -> float:
        counts = np.bincount(y)
        probabilities = counts / len(y)
        probabilities = probabilities[probabilities > 0]  # Avoid log(0)
        return -np.sum(probabilities * np.log2(probabilities))

def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device) -> Tuple[List, List]:
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Extracting features and labels"):
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            features.extend(outputs)
            labels.extend(targets.numpy())
    return np.array(features), np.array(labels)

def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device) -> Tuple[List, List]:
    model.eval()
    features = []
    paths = []
    with torch.no_grad():
        for inputs, names in tqdm(dataloader, desc="Extracting features and paths"):
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            features.extend(outputs)
            paths.extend(names)
    return np.array(features), paths