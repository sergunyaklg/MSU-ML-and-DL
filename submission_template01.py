import numpy as np
import torch
from torch import nn

def create_model():
    # Linear layer mapping from 784 features, so it should be 784->256->16->10

    # your code here
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.Linear(256, 16),
        nn.Linear(16, 10)
    )
    # return model instance (None is just a placeholder)
    
    return model

def count_parameters(model):
    # your code here
    # return integer number (None is just a placeholder)
    
    return sum(p.numel() for p in model.parameters())