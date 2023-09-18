import random
from torch.utils.data import Dataset

class ImageDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'data': self.data[index], 'label': self.labels[index]}
        
    def shuffle(self):
        dataset_tuples = []
        for image, label in zip(self.data, self.labels):
            dataset_tuples.append((image, label))
        
        random.shuffle(dataset_tuples)
        
        data, labels = [], []
        for element in dataset_tuples:
            data.append(element[0])
            labels.append(element[1])
            
        self.data = data
        self.labels = labels