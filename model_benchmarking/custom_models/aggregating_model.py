import torch
import torchvision.models as models
import torch.nn as nn

class AggregatingModel(nn.Module):
    def __init__(self, max_batch_size):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.cnn = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        self.cnn_out_features = self.cnn.fc.out_features
        self.fc1 = nn.Linear(self.max_batch_size, 1)
        self.fc2 = nn.Linear(self.cnn_out_features, 1)

    def forward(self, x):
        embeddings = []
        for index, image in enumerate(x):
            image = torch.unsqueeze(image, dim=0)
            embedding = self.cnn(image)
            embeddings.append(embedding)
        
        embeddings = torch.stack(embeddings, dim=2)
        embeddings = nn.functional.pad(embeddings, pad=(self.max_batch_size - embeddings.shape[2], 0, 0, 0, 0, 0))
        
        x = self.fc1(embeddings)
        x = x.flatten(start_dim = 1)
        return self.fc2(x)