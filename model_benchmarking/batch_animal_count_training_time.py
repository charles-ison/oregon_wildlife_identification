import os
import copy
import time
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
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from custom_data_sets.image_data_set import ImageDataSet
from torch.nn.parallel import DistributedDataParallel
    
    
def get_predictions(bounding_boxes):
    predictions = []
    for boxes in bounding_boxes:
        num_animals = 0
        for score_index, score in enumerate(boxes["scores"]):
            if score > 0.5 and boxes["labels"][score_index] == 1:
                num_animals += 1
        predictions.append(num_animals)
    return predictions
    
    
def get_num_equal_list_elements(labels, predictions):
    return sum(label == prediction for label, prediction in zip(labels, predictions))
    

def is_directory_supplemental(directory):
    if directory.name == "caltech":
        return True
    else:
        return False
    

def fetch_training_data(data_dir):
    json_file_name = "animal_count_key.json"
    training_data, training_labels, validation_data, validation_labels, batch_training_data, batch_training_labels, batch_validation_data, batch_validation_labels = [], [], [], [], [], [], [], []
    
    for directory in os.scandir(data_dir):
        if directory.is_dir():
            directory_path = directory.path + "/"
            print("\nFeting data from directory: ", directory_path)
            is_supplmental = is_directory_supplemental(directory)
            temp_train_data, temp_train_lab, temp_val_data, temp_valid_lab, temp_batch_train_data, temp_batch_train_lab, temp_batch_val_data, temp_batch_val_lab = utilities.fetch_data(directory_path, json_file_name, True, is_supplmental)
            print("Number of images found: ", len(temp_train_data) + len(temp_val_data))
            print("Number of batches found: ", len(temp_batch_train_data) + len(temp_batch_val_data))
            training_data.extend(temp_train_data)
            training_labels.extend(temp_train_lab)
            validation_data.extend(temp_val_data)
            validation_labels.extend(temp_valid_lab)
            batch_training_data.extend(temp_batch_train_data)
            batch_training_labels.extend(temp_batch_train_lab)
            batch_validation_data.extend(temp_batch_val_data)
            batch_validation_labels.extend(temp_batch_val_lab)
    
    training_data = training_data[0:int(len(training_data)/10)]
    training_labels = training_labels[0:int(len(training_labels)/10)]
    
    validation_data = validation_data[0:int(len(validation_data)/10)]
    validation_labels = validation_labels[0:int(len(validation_labels)/10)]
    
    batch_training_data = batch_training_data[0:int(len(batch_training_data)/10)]
    batch_training_labels = batch_training_labels[0:int(len(batch_training_labels)/10)]
    
    batch_validation_data = batch_validation_data[0:int(len(batch_validation_data)/10)]
    batch_validation_labels = batch_validation_labels[0:int(len(batch_validation_labels)/10)]
    
    print("\nNumber of training images: ", len(training_data))
    print("Number of validation images: ", len(validation_data))
    print("Number of training batches: ", len(batch_training_data))
    print("Number of validation batches: ", len(batch_validation_data))
    
    return training_data, training_labels, validation_data, validation_labels, batch_training_data, batch_training_labels, batch_validation_data, batch_validation_labels
    
    

def train(model, training_data_set, criterion, optimizer, rank, batch_size):
    model.train()
    running_loss = 0.0
    
    for index in range(0, len(training_data_set), batch_size):
        batch = training_data_set[index:index + batch_size]
        data, targets = utilities.get_info_from_batch(batch, rank)
        data = torch.stack(data)
        labels = utilities.get_labels_from_targets(targets)
        labels = torch.FloatTensor(labels).to(rank)
        
        optimizer.zero_grad()
        output = model(data).flatten()

        loss = criterion(output, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    loss = running_loss/len(training_data_set)
    return loss
    

def train_object_detection(model, training_data_set, optimizer, rank, batch_size):
    model.train()
    running_loss = 0.0
    for index in range(0, len(training_data_set), batch_size):
        batch = training_data_set[index:index + batch_size]
        data, targets = utilities.get_info_from_batch(batch, rank)
        
        optimizer.zero_grad()
        losses_dict = model(data, targets)
        
        sum_losses = sum(loss for loss in losses_dict.values())
        running_loss += sum_losses.item()
        sum_losses.backward()
        optimizer.step()

    loss = running_loss/len(training_data_set)
    return loss
    
    
def validation_object_detection(model, validation_data_set, mse_criterion, mae_criterion, rank, batch_size):
    running_mse = 0.0
    running_mae = 0.0
    num_correct = 0

    for index in range(0, len(validation_data_set), batch_size):
        batch = validation_data_set[index:index + batch_size]
        data, targets = utilities.get_info_from_batch(batch, rank)
        
        model.eval()
        bounding_boxes = model(data)
        
        predictions = get_predictions(bounding_boxes)
        labels = utilities.get_labels_from_targets(targets)
        
        running_mse += mse_criterion(torch.FloatTensor(predictions), torch.FloatTensor(labels)).item()
        running_mae += mae_criterion(torch.FloatTensor(predictions), torch.FloatTensor(labels)).item()
        num_correct += get_num_equal_list_elements(labels, predictions)

    mse = running_mse/len(validation_data_set)
    mae = running_mae/len(validation_data_set)
    accuracy = num_correct/len(validation_data_set)
    return mse, mae, accuracy


def validation(model, validation_data_set, mse_criterion, mae_criterion, rank, batch_size):
    model.eval()
    running_mse = 0.0
    running_mae = 0.0
    num_correct = 0

    for index in range(0, len(validation_data_set), batch_size):
        batch = validation_data_set[index:index + batch_size]
        data, targets = utilities.get_info_from_batch(batch, rank)
        data = torch.stack(data)
        labels = utilities.get_labels_from_targets(targets)
        labels = torch.FloatTensor(labels).to(rank)
        
        output = model(data).flatten()

        running_mse += mse_criterion(output, labels).item()
        running_mae += mae_criterion(output, labels).item()
        num_correct += (output.round() == labels).sum().item()

    mse = running_mse/len(validation_data_set)
    mae = running_mae/len(validation_data_set)
    accuracy = num_correct/len(validation_data_set)
    return mse, mae, accuracy
    
    
def get_thread_subset_of_data(rank, world_size, data, labels):
    start_index = int((rank/world_size) * len(data))
    end_index = int(((rank+1.0)/world_size) * len(data))
    return data[start_index:end_index], labels[start_index:end_index]


def train_and_validate(rank, world_size, num_epochs, model, model_name, training_data, training_labels, validation_data, validation_labels, saving_dir, batch_size, is_object_detection):
    
    training_data, training_labels = get_thread_subset_of_data(rank, world_size, training_data, training_labels)
    validation_data, validation_labels = get_thread_subset_of_data(rank, world_size, validation_data, validation_labels)
    
    training_data_set = ImageDataSet(training_data, training_labels)
    validation_data_set = ImageDataSet(validation_data, validation_labels)

    dist.init_process_group(world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    huber_loss = nn.HuberLoss()
    lowest_batch_val_mae = None
    saving_dir = saving_dir + "batch_count_" + model_name + "/"
    start_time = time.time()

    for epoch in range(num_epochs):
        training_data_set.shuffle()
        validation_data_set.shuffle()

        if is_object_detection:
            training_loss = train_object_detection(model, training_data_set, optimizer, rank, batch_size)
        else:
           training_loss = train(model, training_data_set, huber_loss, optimizer, rank, batch_size)
        
        print(time.time() - start_time)
        
    dist.destroy_process_group()


if __name__ == '__main__':
    # Declaring Constants
    num_epochs = 10
    batch_size = 64
    object_detection_batch_size = 10
    data_dir = "../saved_data/training_animal_count/"
    saving_dir = "../saved_models/"

    print(torch.__version__)
    print(torchvision.__version__)
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Configuring these based on: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    training_data, training_labels, validation_data, validation_labels, batch_training_data, batch_training_labels, batch_validation_data, batch_validation_labels = fetch_training_data(data_dir)
    training_data = torch.stack(training_data, dim=0).clone()
    validation_data = torch.stack(validation_data, dim=0).clone()

    for i in range(torch.cuda.device_count()):

        # Declaring Models
        resnet34 = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
        in_features = resnet34.fc.in_features
        resnet34.fc = nn.Linear(in_features, 1)

        resnet50 = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        in_features = resnet50.fc.in_features
        resnet50.fc = nn.Linear(in_features, 1)

        resnet152 = models.resnet152(weights = models.ResNet152_Weights.DEFAULT)
        in_features = resnet152.fc.in_features
        resnet152.fc = nn.Linear(in_features, 1)

        retina_net = models.detection.retinanet_resnet50_fpn_v2(weights=models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        
        world_size = i + 1
        print("\nUsing " + str(world_size) + " GPUs")

        # Training
        print("\nTraining and Validating ResNet34")
        mp.spawn(train_and_validate, args=(world_size, num_epochs, resnet34, "ResNet34", training_data, training_labels, validation_data, validation_labels, saving_dir, batch_size, False), nprocs=world_size, join=True)

        print("\nTraining and Validating ResNet50")
        mp.spawn(train_and_validate, args=(world_size, num_epochs, resnet50, "ResNet50", training_data, training_labels, validation_data, validation_labels, saving_dir, batch_size, False), nprocs=world_size, join=True)

        print("\nTraining and Validating ResNet152")
        mp.spawn(train_and_validate, args=(world_size, num_epochs, resnet152, "ResNet152", training_data, training_labels, validation_data, validation_labels, saving_dir, batch_size, False), nprocs=world_size, join=True)

        print("\nTraining and Validating RetinaNet")
        mp.spawn(train_and_validate, args=(world_size, num_epochs, retina_net, "RetinaNet", training_data, training_labels, validation_data, validation_labels, saving_dir, object_detection_batch_size, True), nprocs=world_size, join=True)




