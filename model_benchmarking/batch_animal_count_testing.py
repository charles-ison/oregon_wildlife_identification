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
from pytorch_grad_cam import FullGrad

def test_batch(model, batch_testing_loader, criterion, print_incorrect_images, saving_dir, device):
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
    
def test_individual(model, grad_cam, testing_loader, criterion, print_incorrect_images, print_heat_map, saving_dir, device):
    model.eval()
    running_loss = 0.0
    num_correct = 0
    all_labels, all_predictions = [], []
    grad_cam_identifier = 0

    for i, batch in enumerate(testing_loader):
        data, labels = batch['data'].to(device), batch['label'].to(device)
        output = model(data).flatten()

        loss = criterion(output, labels)
        running_loss += loss.item()
        for index, prediction in enumerate(output.round()):
            prediction = prediction.cpu().item()
            all_predictions.append(prediction)
            if prediction == labels[index]:
                num_correct += 1
            elif print_incorrect_images:
                utilities.print_image(data[index], prediction, saving_dir, i)
            
            # Just looking at every 5 samples
            if print_heat_map and grad_cam_identifier % 10 == 0:
                utilities.create_heat_map(grad_cam, data[index], prediction, labels[index], saving_dir, grad_cam_identifier)
            grad_cam_identifier += 1

        all_labels.extend(labels.cpu())

    loss = running_loss/len(testing_loader.dataset)
    accuracy = num_correct/len(testing_loader.dataset)
    return loss, accuracy, all_labels, all_predictions

def test(model, grad_cam, batch_testing_loader, individual_testing_loader, device, criterion, saving_dir):
    model.to(device)
    
    individual_testing_loss, individual_testing_accuracy, individual_labels, individual_predictions = test_individual(model, grad_cam, individual_testing_loader, criterion, False, True, saving_dir, device)
    print("individual testing loss (MSE): " + str(individual_testing_loss) + " and individual testing accuracy: "+ str(individual_testing_accuracy))

    batch_testing_loss, batch_testing_accuracy, batch_labels, batch_predictions = test_batch(model, batch_testing_loader, criterion, False, saving_dir, device)
    print("batch testing loss (MSE): " + str(batch_testing_loss) + " and batch testing accuracy: "+ str(batch_testing_accuracy))
    
def get_data_loaders(data_dir, json_file_name):
    batch_testing_data, batch_testing_labels, individual_data, individual_labels = utilities.fetch_data(data_dir, json_file_name, False, False, False)
    batch_testing_data_set = utilities.image_data_set(batch_testing_data, batch_testing_labels)
    individual_testing_data_set = utilities.image_data_set(individual_data, individual_labels)
    batch_data_loader = DataLoader(dataset = batch_testing_data_set, batch_size = 1, shuffle = True)
    individual_data_loader = DataLoader(dataset = individual_testing_data_set, batch_size = 10, shuffle = True)
    return batch_data_loader, individual_data_loader

# Declaring Constants
batch_size = 10
cottonwood_eastface_json_file_name = "2023_Cottonwood_Eastface_5.30_7.10_key.json"
cottonwood_westface_json_file_name = "2023_Cottonwood_Westface_5.30_7.10_102RECNX_key.json"
ngilchrist_eastface_json_file_name = "2022_NGilchrist_Eastface_055_07.12_07.20_key.json"
idaho_json_file_name = "Idaho_loc_0099_key.json"
data_dir = "/nfs/stak/users/isonc/hpc-share/saved_data/testing_animal_count/"

resnet_50_saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/batch_count_ResNet50/"
resnet_152_saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/batch_count_ResNet152/"

resnet50_weights_path = resnet_50_saving_dir + "ResNet50.pt"
resnet152_weights_path = resnet_152_saving_dir + "ResNet152.pt"

resnet_50_grad_cam_path = resnet_50_saving_dir + "grad_cam/"
resnet_152_grad_cam_path = resnet_152_saving_dir + "grad_cam/"

print(torch.__version__)
print(torchvision.__version__)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
criterion = nn.MSELoss()

print("\nGetting Cottonwood Eastface data")
cottonwood_eastface_batch_testing_loader, cottonwood_eastface_individual_data_loader = get_data_loaders(data_dir, cottonwood_eastface_json_file_name)

print("\nGetting Cottonwood Westface data")
cottonwood_westface_batch_testing_loader, cottonwood_westface_individual_data_loader = get_data_loaders(data_dir, cottonwood_westface_json_file_name)

print("\nGetting NGilchrist Eastface data")
ngilchrist_eastface_batch_testing_loader, ngilchrist_eastface_individual_data_loader = get_data_loaders(data_dir, ngilchrist_eastface_json_file_name)

print("\nGetting Idaho data")
idaho_batch_testing_loader, idaho_individual_data_loader = get_data_loaders(data_dir, idaho_json_file_name)

# Declaring Models
# Have follow same steps used to create model during training
resnet50 = models.resnet50()
in_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(in_features, 1)

resnet152 = models.resnet152()
in_features = resnet152.fc.in_features
resnet152.fc = nn.Linear(in_features, 1)

#Layer 4 is just recommended by GradCam documentation for ResNet
resnet50_cam = FullGrad(model=resnet50, target_layers=[], use_cuda=torch.cuda.is_available())
resnet152_cam = FullGrad(model=resnet152, target_layers=[], use_cuda=torch.cuda.is_available())

#Loading trained model weights
resnet50.load_state_dict(torch.load(resnet50_weights_path))
resnet152.load_state_dict(torch.load(resnet152_weights_path))

if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    resnet50 = nn.DataParallel(resnet50)
    resnet152 = nn.DataParallel(resnet152)

# Testing
print("\nTesting ResNet50 on Cottonwood Eastface")
test(resnet50, resnet50_cam, cottonwood_eastface_batch_testing_loader, cottonwood_eastface_individual_data_loader, device, criterion, resnet_50_grad_cam_path + "cottonwood_eastface/")
print("\nTesting ResNet50 on Cottonwood Westface")
test(resnet50, resnet50_cam, cottonwood_westface_batch_testing_loader, cottonwood_westface_individual_data_loader, device, criterion, resnet_50_grad_cam_path + "cottonwood_westface/")
print("\nTesting ResNet50 on NGilchrist Eastface")
test(resnet50, resnet50_cam, ngilchrist_eastface_batch_testing_loader, ngilchrist_eastface_individual_data_loader, device, criterion, resnet_50_grad_cam_path + "ngilchrist_eastface/")
print("\nTesting ResNet50 on Idaho")
test(resnet50, resnet50_cam, idaho_batch_testing_loader, idaho_individual_data_loader, device, criterion, resnet_50_grad_cam_path + "idaho/")

print("\nTesting ResNet152 on Cottonwood Eastface")
test(resnet152, resnet152_cam, cottonwood_eastface_batch_testing_loader, cottonwood_eastface_individual_data_loader, device, criterion, resnet_152_grad_cam_path + "cottonwood_eastface/")
print("\nTesting ResNet152 on Cottonwood Westface")
test(resnet152, resnet152_cam, cottonwood_westface_batch_testing_loader, cottonwood_westface_individual_data_loader, device, criterion, resnet_152_grad_cam_path + "cottonwood_westface/")
print("\nTesting ResNet152 on NGilchrist Eastface")
test(resnet152, resnet152_cam, ngilchrist_eastface_batch_testing_loader, ngilchrist_eastface_individual_data_loader, device, criterion, resnet_152_grad_cam_path + "ngilchrist_eastface/")
print("\nTesting ResNet152 on Idaho")
test(resnet152, resnet152_cam, idaho_batch_testing_loader, idaho_individual_data_loader, device, criterion, resnet_152_grad_cam_path + "idaho/")



