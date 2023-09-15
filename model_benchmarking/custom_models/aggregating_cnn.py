import torch
import torchvision.models as models
import torch.nn as nn

class AggregatingCNN(nn.Module):
    def __init__(self, max_batch_size):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.cnn = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
        self.cnn_out_features = self.cnn.fc.out_features
        self.fc1 = nn.Linear(self.cnn_out_features, 2 * self.cnn_out_features)
        self.fc2 = nn.Linear(2 * self.cnn_out_features, 1)
        self.fc3 = nn.Linear(self.max_batch_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        embeddings = []
        
        if x.shape[0] > self.max_batch_size:
            raise Exception("Number of images in batch: " + str(x.shape[0]) + " is greater than max_batch_size: " + str(self.max_batch_size))
        
        for index, image in enumerate(x):
            image = torch.unsqueeze(image, dim=0)
            embedding = self.cnn(image)
            embedding = torch.squeeze(embedding, dim=0)
            embeddings.append(embedding)
        
        embeddings = torch.stack(embeddings, dim=0)
        embeddings = nn.functional.pad(embeddings, pad=(0, 0, 0, self.max_batch_size - embeddings.shape[0]))
        x = self.fc1(embeddings)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = torch.squeeze(x, dim=1)
        x = self.fc3(x)
        x = torch.squeeze(x, dim=0)
        return x.round()