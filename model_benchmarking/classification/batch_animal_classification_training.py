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
from operator import itemgetter
from datetime import datetime
from pycocotools.coco import COCO

def print_validation_analysis(all_labels, all_predictions, title, data_dir, saving_dir):
    subplot = plt.subplot()

    cf_matrix = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    cf_matrix = np.flip(cf_matrix, axis=0)
    sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues')

    subplot.set_xlabel('Predictions')
    subplot.set_ylabel('Labels')
    subplot.set_title(title + ' Validation Confusion Matrix')
    subplot.xaxis.set_ticklabels([0, 1])
    subplot.yaxis.set_ticklabels([1, 0])

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
        output = model(data)

        loss = criterion(output, labels)
        running_loss += loss.item()
        
        _, predictions = torch.max(output.data, 1)
        num_correct += (predictions == labels).sum().item()
        
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
        output = model(data)
        
        loss = criterion(output, labels)
        running_loss += loss.item()
        
        _, predictions = torch.max(output.data, 1)
        for index, prediction in enumerate(predictions):
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
    all_labels, all_predictions = [], []

    for batch in batch_validation_loader:
        data, labels = torch.squeeze(batch['data'], dim=0).to(device), batch['label'].to(device)

        # This is to prevent cuda memory issues for large batches
        batch_prediction = 0
        for image in data:
            image = torch.unsqueeze(image, dim=0)
            output = model(image)
            _, prediction = torch.max(output.data, 1)
            if prediction == 1:
                batch_prediction = 1
        
        batch_label = 0
        # TODO: We have to use a 0 index here because batch size is 1, should fix 
        for label in labels[0]:
            if label == 1:
                batch_label = 1

        if batch_prediction == batch_label:
            num_correct += 1

        all_predictions.append(batch_prediction)
        all_labels.append(batch_label)

    accuracy = num_correct/len(batch_validation_loader.dataset)
    return accuracy, all_labels, all_predictions

def train_and_validate(num_epochs, model, model_name, training_loader, validation_loader, batch_validation_loader, device, criterion, data_dir, saving_dir):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    highest_batch_validation_accuracy = 0.0
    saving_dir = saving_dir + "batch_classification_" + model_name + "/"

    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch))

        training_loss, training_accuracy = train(model, training_loader, criterion, optimizer, device)
        print("training loss: " + str(training_loss) + " and training accuracy: " + str(training_accuracy))

        validation_loss, validation_accuracy, _, _ = validation(model, validation_loader, criterion, False, data_dir, device)
        print("validation loss: " + str(validation_loss) + " and validation accuracy: " + str(validation_accuracy))

        batch_validation_accuracy, batch_labels, batch_predictions = batch_validation(model, batch_validation_loader, criterion, False, data_dir, device)
        print("batch validation accuracy: " + str(batch_validation_accuracy))

        if highest_batch_validation_accuracy < batch_validation_accuracy:
            print("Highest batch validation accuracy achieved, saving weights")
            highest_batch_validation_accuracy = batch_validation_accuracy
            torch.save(model.module.state_dict(), saving_dir + model_name + ".pt")
            print_validation_analysis(batch_labels, batch_predictions, model_name, data_dir, saving_dir)


# Declaring Constants
num_epochs = 5
num_classes = 2
batch_size = 10
json_file_name = "animal_count_key.json"
data_dir = "/nfs/stak/users/isonc/hpc-share/saved_data/2022_Cottonwood_Eastface_and_Repelcam/"
saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/"

print(torch.__version__)
print(torchvision.__version__)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
criterion = nn.CrossEntropyLoss()

training_data, validation_data, training_labels, validation_labels, batch_validation_data, batch_validation_labels = utilities.fetch_data(data_dir, json_file_name, True, False, True)
training_data_set = utilities.image_data_set(training_data, training_labels)
validation_data_set = utilities.image_data_set(validation_data, validation_labels)
batch_validation_data_set = utilities.image_data_set(batch_validation_data, batch_validation_labels)
training_loader = DataLoader(dataset = training_data_set, batch_size = batch_size, shuffle = True)
validation_loader = DataLoader(dataset = validation_data_set, batch_size = batch_size, shuffle = True)
batch_validation_loader = DataLoader(dataset = batch_validation_data_set, batch_size = 1, shuffle = True)

# Declaring Models
resnet50 = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
in_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(in_features, num_classes)

resnet152 = models.resnet152(weights = models.ResNet152_Weights.DEFAULT)
in_features = resnet152.fc.in_features
resnet152.fc = nn.Linear(in_features, num_classes)

vit_l_16 = models.vit_l_16(weights = models.ViT_L_16_Weights.DEFAULT)
in_features = vit_l_16.heads[0].in_features
vit_l_16.heads[0] = nn.Linear(in_features, num_classes)

if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    resnet50 = nn.DataParallel(resnet50)
    resnet152 = nn.DataParallel(resnet152)
    vit_l_16 = nn.DataParallel(vit_l_16)

# Training
print("\nTraining and Validating ResNet50")
train_and_validate(num_epochs, resnet50, "ResNet50", training_loader, validation_loader, batch_validation_loader, device, criterion, data_dir, saving_dir)

print("\nTraining and Validating ResNet152")
train_and_validate(num_epochs, resnet152, "ResNet152", training_loader, validation_loader, batch_validation_loader, device, criterion, data_dir, saving_dir)




