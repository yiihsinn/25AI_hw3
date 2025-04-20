import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        # (TODO) Design your CNN, it can only be less than 3 convolution layers
        super(CNN, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        # (TODO) Forward the model
        raise NotImplementedError
        return x

def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device)->float:
    # (TODO) Train the model and return the average loss of the data, we suggest use tqdm to know the progress
    raise NotImplementedError
    return avg_loss


def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    # (TODO) Validate the model and return the average loss and accuracy of the data, we suggest use tqdm to know the progress
    raise NotImplementedError
    return avg_loss, accuracy

def test(model: CNN, test_loader: DataLoader, criterion, device):
    # (TODO) Test the model on testing dataset and write the result to 'CNN.csv'
    raise NotImplementedError
    print(f"Predictions saved to 'CNN.csv'")
    return