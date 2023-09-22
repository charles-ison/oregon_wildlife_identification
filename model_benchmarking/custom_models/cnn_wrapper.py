import torch
import torchvision.models as models
import torch.nn as nn

class CNNWrapper(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn
        out_features = self.cnn.fc.out_features
        self.fc1 = nn.Linear(out_features, 2 * out_features)
        self.fc2 = nn.Linear(2 * out_features, 2 * out_features)
        self.fc3 = nn.Linear(2 * out_features, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return torch.squeeze(x, dim=0)