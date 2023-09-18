import torch
import math
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        
    def forward(self, predictions, labels):
        if predictions.size() == torch.Size([]) and labels.size() == torch.Size([]):
            return self.get_weighted_mse(predictions, labels)
        
        weighted_mse_list = []
        for prediction, label in zip(predictions, labels):
            weighted_mse = self.get_weighted_mse(prediction, label)
            weighted_mse_list.append(weighted_mse)
        
        weighted_mse_tensor = torch.tensor(weighted_mse_list, requires_grad=True)
        return torch.mean(weighted_mse_tensor)
        
    
    def get_weighted_mse(self, prediction, label):
        mse = torch.square(prediction - label)
        weight = self.weights[int(label.item())]
        return weight * mse
        