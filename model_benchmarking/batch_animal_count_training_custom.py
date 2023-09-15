import os
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import utilities
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def set_device_for_list_of_dicts(some_list, device):
    for some_dict in some_list:
        some_dict["boxes"] = some_dict["boxes"].to(device)
        some_dict["labels"] = some_dict["labels"].to(device)
        

def get_info_from_batch(batch):
    data, targets = batch['data'], batch['label']
    utilities.set_device_for_list_of_tensors(data, device)
    set_device_for_list_of_dicts(targets, device)
    return data, targets


def print_validation_analysis(all_labels, all_predictions, title, data_dir, saving_dir):
    subplot = plt.subplot()

    cf_matrix = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cf_matrix = np.flip(cf_matrix, axis=0)
    sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues')

    subplot.set_xlabel('Predictions')
    subplot.set_ylabel('Labels')
    subplot.set_title(title + ' Validation Confusion Matrix')
    subplot.xaxis.set_ticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    subplot.yaxis.set_ticklabels([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    plot_file_name = saving_dir + title + "_Confusion_Matrix.png"
    plt.savefig(plot_file_name)
    plt.show()

    accuracy = accuracy_score(all_labels, all_predictions)
    print(title + " Accuracy: " + str(accuracy))

    precision, recall, f_score, support = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    print(title + " Precision: " + str(precision))
    print(title + " Recall: " + str(recall))
    print(title + " F-Score: " + str(f_score))

def train(model, training_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    num_correct = 0
    for batch in training_loader:
        data, labels = batch['data'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        output = model(data).flatten()

        loss = criterion(output, labels)
        running_loss += loss.item()
        num_correct += (output.round() == labels).sum().item()
        loss.backward()
        optimizer.step()

    loss = running_loss/len(training_loader.dataset)
    accuracy = num_correct/len(training_loader.dataset)
    return loss, accuracy

def validation(model, validation_loader, criterion, print_incorrect_images, data_dir, device):
    model.eval()
    running_loss = 0.0
    num_correct = 0
    all_labels, all_predictions = [], []

    for i, batch in enumerate(validation_loader):
        data, labels = batch['data'].to(device), batch['label'].to(device)
        output = model(data).flatten()

        loss = criterion(output, labels)
        running_loss += loss.item()
        for index, prediction in enumerate(output.round()):
            all_predictions.append(prediction.cpu().item())
            if(prediction == labels[index]):
                num_correct += 1
            elif(print_incorrect_images):
                utilities.print_image(data[index], prediction, data_dir, i)

        all_labels.extend(labels.cpu())

    loss = running_loss/len(validation_loader.dataset)
    accuracy = num_correct/len(validation_loader.dataset)
    return loss, accuracy, all_labels, all_predictions

def batch_validation(model, batch_validation_loader, criterion, print_incorrect_images, data_dir, device):
    model.eval()
    num_correct = 0
    running_loss = 0.0
    all_labels, all_predictions = [], []

    for batch in batch_validation_loader:
        data, labels = torch.squeeze(batch['data'], dim=0).to(device), batch['label'].to(device)

        # This is to prevent cuda memory issues for large batches
        max_prediction = 0
        for image in data:
            image = torch.unsqueeze(image, dim=0)
            output = model(image).flatten()
            max_prediction = max(max_prediction, output.round().item())

        max_prediction = torch.tensor(max_prediction).to(device)
        max_label = torch.max(labels)
        
        loss = criterion(max_prediction, max_label)
        running_loss += loss.item()

        if max_prediction == max_label:
            num_correct += 1

        all_predictions.append(max_prediction.cpu())
        all_labels.append(max_label.cpu())

    loss = running_loss/len(batch_validation_loader.dataset)
    accuracy = num_correct/len(batch_validation_loader.dataset)
    return loss, accuracy, all_labels, all_predictions

def train_and_validate(num_epochs, model, model_name, training_loader, validation_loader, batch_validation_loader, device, criterion, data_dir, saving_dir):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    highest_batch_validation_accuracy = 0.0
    saving_dir = saving_dir + "batch_count_" + model_name + "/"

    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch))

        training_loss, training_accuracy = train(model, training_loader, criterion, optimizer, device)
        print("training loss: " + str(training_loss) + " and training accuracy: " + str(training_accuracy))

        validation_loss, validation_accuracy, _, _ = validation(model, validation_loader, criterion, False, data_dir, device)
        print("validation loss: " + str(validation_loss) + " and validation accuracy: " + str(validation_accuracy))

        batch_validation_loss, batch_validation_accuracy, batch_labels, batch_predictions = batch_validation(model, batch_validation_loader, criterion, False, data_dir, device)
        print("batch validation loss (MSE): " + str(batch_validation_loss) + " and batch validation accuracy: "+ str(batch_validation_accuracy))

        if highest_batch_validation_accuracy < batch_validation_accuracy:
            print("Highest batch validation accuracy achieved, saving weights")
            highest_batch_validation_accuracy = batch_validation_accuracy
            torch.save(model.module.state_dict(), saving_dir + model_name + ".pt")
            print_validation_analysis(batch_labels, batch_predictions, model_name, data_dir, saving_dir)


# Declaring Constants
num_epochs = 5
batch_size = 5
json_file_name = "animal_count_key.json"
data_dir = "/nfs/stak/users/isonc/hpc-share/saved_data/2022_Cottonwood_Eastface_bounding_boxes/"
saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/"

print(torch.__version__)
print(torchvision.__version__)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
criterion = nn.MSELoss()

training_data, validation_data, training_labels, validation_labels, batch_training_data, batch_training_labels, batch_validation_data, batch_validation_labels = utilities.fetch_data(data_dir, json_file_name, False, True, True)
training_data_set = utilities.image_data_set(training_data, training_labels)
validation_data_set = utilities.image_data_set(validation_data, validation_labels)
batch_validation_data_set = utilities.image_data_set(batch_validation_data, batch_validation_labels)
training_loader = DataLoader(dataset = training_data_set, batch_size = batch_size, shuffle = True)
validation_loader = DataLoader(dataset = validation_data_set, batch_size = batch_size, shuffle = True)
batch_validation_loader = DataLoader(dataset = batch_validation_data_set, batch_size = 1, shuffle = True)

class CustomModel(nn.Module):
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

max_batch_size = 100
custom_model = CustomModel(max_batch_size)
custom_model.to(device)

for batch in batch_validation_data_set:
    data, targets = batch['data'].to(device), batch['label']
    output = custom_model(data)

if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    custom_model = nn.DataParallel(custom_model)

print("\nTraining and Validating Custom Model")
train_and_validate(num_epochs, custom_model, "CustomModel", training_loader, validation_loader, batch_validation_loader, device, criterion, data_dir, saving_dir)

