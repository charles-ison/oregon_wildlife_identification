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
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from operator import itemgetter
from datetime import datetime
from pycocotools.coco import COCO

def print_testing_analysis(all_labels, all_predictions, title, data_dir, saving_dir):
    subplot = plt.subplot()

    cf_matrix = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cf_matrix = np.flip(cf_matrix, axis=0)
    sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues')

    subplot.set_xlabel('Predictions')
    subplot.set_ylabel('Labels')
    subplot.set_title(title + ' Testing Confusion Matrix')
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

def test(model, testing_loader, criterion, print_incorrect_images, data_dir, device):
    model.eval()
    running_loss = 0.0
    num_correct = 0
    all_labels, all_predictions = [], []

    for i, batch in enumerate(testing_loader):
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

    loss = running_loss/len(testing_loader.dataset)
    accuracy = num_correct/len(testing_loader.dataset)
    return loss, accuracy, all_labels, all_predictions

def test_batch(model, batch_testing_loader, criterion, print_incorrect_images, data_dir, device):
    model.eval()
    num_correct = 0
    running_loss = 0.0
    all_labels, all_predictions = [], []

    for batch in batch_testing_loader:
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

    loss = running_loss/len(batch_testing_loader.dataset)
    accuracy = num_correct/len(batch_testing_loader.dataset)
    return loss, accuracy, all_labels, all_predictions

def train_and_test(num_epochs, model, model_name, training_loader, testing_loader, batch_testing_loader, device, criterion, data_dir, saving_dir):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    highest_batch_testing_accuracy = 0.0
    saving_dir = saving_dir + "batch_count_" + model_name + "/"

    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch))

        training_loss, training_accuracy = train(model, training_loader, criterion, optimizer, device)
        print("training loss: " + str(training_loss) + " and training accuracy: " + str(training_accuracy))

        testing_loss, testing_accuracy, _, _ = test(model, testing_loader, criterion, False, data_dir, device)
        print("testing loss: " + str(testing_loss) + " and testing accuracy: " + str(testing_accuracy))

        batch_testing_loss, batch_testing_accuracy, batch_labels, batch_predictions = test_batch(model, batch_testing_loader, criterion, False, data_dir, device)
        print("batch testing loss (MSE): " + str(batch_testing_loss) + " and batch testing accuracy: "+ str(batch_testing_accuracy))

        if highest_batch_testing_accuracy < batch_testing_accuracy:
            print("Highest batch testing accuracy achieved, saving weights")
            highest_batch_testing_accuracy = batch_testing_accuracy
            torch.save(model.module.state_dict(), saving_dir + model_name + ".pt")
            print_testing_analysis(batch_labels, batch_predictions, model_name, data_dir, saving_dir)


# Declaring Constants
num_epochs = 5
batch_size = 10
json_file_name = "animal_count_key.json"
data_dir = "/nfs/stak/users/isonc/hpc-share/saved_data/animal_count_manually_labeled_wildlife_data/"
saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/"

print(torch.__version__)
print(torchvision.__version__)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
criterion = nn.MSELoss()

training_data, testing_data, training_labels, testing_labels, batch_testing_data, batch_testing_labels = utilities.get_data_sets(data_dir, json_file_name, False)
training_data_set = utilities.image_data_set(training_data, training_labels)
testing_data_set = utilities.image_data_set(testing_data, testing_labels)
batch_testing_data_set = utilities.image_data_set(batch_testing_data, batch_testing_labels)
training_loader = DataLoader(dataset = training_data_set, batch_size = batch_size, shuffle = True)
testing_loader = DataLoader(dataset = testing_data_set, batch_size = batch_size, shuffle = True)
batch_testing_loader = DataLoader(dataset = batch_testing_data_set, batch_size = 1, shuffle = True)

# Declaring Models
resnet50 = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
in_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(in_features, 1)

resnet152 = models.resnet152(weights = models.ResNet152_Weights.DEFAULT)
in_features = resnet152.fc.in_features
resnet152.fc = nn.Linear(in_features, 1)

vit_l_16 = models.vit_l_16(weights = models.ViT_L_16_Weights.DEFAULT)
in_features = vit_l_16.heads[0].in_features
vit_l_16.heads[0] = nn.Linear(in_features, 1)

if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    resnet50 = nn.DataParallel(resnet50)
    resnet152 = nn.DataParallel(resnet152)
    vit_l_16 = nn.DataParallel(vit_l_16)

# Training
print("\nTraining and Testing ResNet50")
train_and_test(num_epochs, resnet50, "ResNet50", training_loader, testing_loader, batch_testing_loader, device, criterion, data_dir, saving_dir)

print("\nTraining and Testing ResNet152")
train_and_test(num_epochs, resnet152, "ResNet152", training_loader, testing_loader, batch_testing_loader, device, criterion, data_dir, saving_dir)



