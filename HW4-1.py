

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Определение слоев сети
        # Сверточный слой 1: принимает 3 канала, имеет 3 фильтра размером 5x5
        self.conv1 = nn.Conv2d(3, 3, (5, 5))
        # Макс-пулинг слой 1: имеет ядро размером 2x2
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        # Сверточный слой 2: принимает 3 канала, имеет 5 фильтров размером 3x3
        self.conv2 = nn.Conv2d(3, 5, (3, 3))
        # Макс-пулинг слой 2: имеет ядро размером 2x2
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        
        # Слой для выравнивания
        self.flatten = nn.Flatten()

        # Полносвязный слой 1: принимает на вход вектор размером 5*6*6 и имеет 100 нейронов на выходе
        self.fc1 = nn.Linear(5 * 6 * 6, 100)
        # Полносвязный слой 2: принимает на вход вектор размером 100 и имеет 10 нейронов на выходе
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # Размерность х ~ [32, 3, 32, 32]

        # Применение сверточного слоя 1
        x = self.conv1(x)
        # Применение функции активации ReLU
        x = F.relu(x)
        # Применение макс-пулинга слоя 1
        x = self.pool1(x)
        
        # Применение сверточного слоя 2
        x = self.conv2(x)
        # Применение функции активации ReLU
        x = F.relu(x)
        # Применение макс-пулинга слоя 2
        x = self.pool2(x)
        
        # Выравнивание тензора
        x = self.flatten(x)

        # Применение полносвязного слоя 1
        x = self.fc1(x)
        # Применение функции активации ReLU
        x = F.relu(x)
        # Применение полносвязного слоя 2
        x = self.fc2(x)
        
        return x
    
def create_model():
    return ConvNet()