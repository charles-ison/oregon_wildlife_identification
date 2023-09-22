import torch
import torchvision.models as models
import torch.nn as nn

class AggregatingCNN(nn.Module):
    def __init__(self, max_batch_size, embedding_size, cnn):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.embedding_size = embedding_size
        self.cnn = cnn.module
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(in_features, self.embedding_size)
        self.fc1 = nn.Linear(self.embedding_size, 2 * self.embedding_size)
        self.fc2 = nn.Linear(2 * self.embedding_size, 1)
        self.fc3 = nn.Linear(self.max_batch_size, 1)
        self.relu = nn.ReLU()

    def forward(self, input_list):
        results = []
        for x in input_list:
            if x.shape[0] > self.max_batch_size:
                raise Exception("Number of images in batch: " + str(x.shape[0]) + " is greater than max_batch_size: " + str(self.max_batch_size))
        
            x = self.cnn(x)
            x = nn.functional.pad(x, pad=(0, 0, 0, self.max_batch_size - x.shape[0]))
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = torch.squeeze(x, dim=1)
            x = self.fc3(x)
            x = torch.squeeze(x, dim=0)
            results.append(x)
        return torch.stack(results)