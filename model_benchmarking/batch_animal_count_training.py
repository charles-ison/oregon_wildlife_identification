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
from custom_models.aggregating_cnn import AggregatingCNN
from custom_models.cnn_wrapper import CNNWrapper
from custom_data_sets.image_data_set import ImageDataSet
    

def set_device_for_list_of_dicts(some_list, device):
    for some_dict in some_list:
        some_dict["boxes"] = some_dict["boxes"].to(device)
        some_dict["labels"] = some_dict["labels"].to(device)
        

def get_info_from_batch(batch):
    data, targets = batch['data'], batch['label']
    utilities.set_device_for_list_of_tensors(data, device)
    set_device_for_list_of_dicts(targets, device)
    return data, targets
    
    
def get_predictions(bounding_boxes):
    predictions = []
    for boxes in bounding_boxes:
        num_animals = 0
        for score_index, score in enumerate(boxes["scores"]):
            if score > 0.5 and boxes["labels"][score_index] == 1:
                num_animals += 1
        predictions.append(num_animals)
    return predictions
    

def fetch_training_data(data_dir):
    json_file_name = "animal_count_key.json"
    training_data, training_labels, validation_data, validation_labels, batch_training_data, batch_training_labels, batch_validation_data, batch_validation_labels = [], [], [], [], [], [], [], []
    
    for directory in os.scandir(data_dir):
        if directory.is_dir():
            directory_path = directory.path + "/"
            print("\nFeting data from directory: ", directory_path)
            temp_train_data, temp_train_lab, temp_val_data, temp_valid_lab, temp_batch_train_data, temp_batch_train_lab, temp_batch_val_data, temp_batch_val_lab = utilities.fetch_data(directory_path, json_file_name, False, True, True)
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
    
    print("\nNumber of training images: ", len(training_data))
    print("Number of validation images: ", len(validation_data))
    print("Number of batches for training: ", len(batch_training_data))
    print("Number of batches for validation: ", len(batch_validation_data))
    utilities.plot_histogram(training_labels, data_dir)
    
    return training_data, training_labels, validation_data, validation_labels, batch_training_data, batch_training_labels, batch_validation_data, batch_validation_labels
    
    
def train_aggregating_cnn(model, training_data_set, criterion, optimizer, device, batch_size):
    model.train()
    running_loss = 0.0

    for index in range(0, len(training_data_set), batch_size):
        batch = training_data_set[index:index + batch_size]
    
        batch_labels = []
        for index, (data, targets) in enumerate(zip(batch["data"], batch["label"])):
            batch["data"][index] = data.to(device)
            labels = utilities.get_labels_from_targets(targets)
            labels = torch.FloatTensor(labels).to(device)
            label = torch.max(labels)
            batch_labels.append(label)
    
        data = batch["data"]
        labels = torch.stack(batch_labels)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    loss = running_loss/len(training_data_set)
    return loss
    

def train(model, training_data_set, criterion, optimizer, device, batch_size):
    model.train()
    running_loss = 0.0
    
    for index in range(0, len(training_data_set), batch_size):
        batch = training_data_set[index:index + batch_size]
        data, targets = get_info_from_batch(batch)
        data = torch.stack(data)
        labels = utilities.get_labels_from_targets(targets)
        labels = torch.FloatTensor(labels).to(device)
        
        optimizer.zero_grad()
        output = model(data).flatten()

        loss = criterion(output, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    loss = running_loss/len(training_data_set)
    return loss
    

def train_object_detection(model, training_data_set, optimizer, device, batch_size):
    model.train()
    running_loss = 0.0
    for index in range(0, len(training_data_set), batch_size):
        batch = training_data_set[index:index + batch_size]
        data, targets = get_info_from_batch(batch)
        
        optimizer.zero_grad()
        losses_dict = model(data, targets)
        
        sum_losses = sum(loss for loss in losses_dict.values())
        running_loss += sum_losses.item()
        sum_losses.backward()
        optimizer.step()

    loss = running_loss/len(training_data_set)
    return loss
    
    
def validation_object_detection(model, validation_data_set, mse_criterion, mae_criterion, device, batch_size):
    running_mse = 0.0
    running_mae = 0.0

    for index in range(0, len(validation_data_set), batch_size):
        batch = validation_data_set[index:index + batch_size]
        data, targets = get_info_from_batch(batch)
        
        model.eval()
        bounding_boxes = model(data)
        
        predictions = get_predictions(bounding_boxes)
        labels = utilities.get_labels_from_targets(targets)
        
        running_mse += mse_criterion(torch.FloatTensor(predictions), torch.FloatTensor(labels)).item()
        running_mae += mae_criterion(torch.FloatTensor(predictions), torch.FloatTensor(labels)).item()

    mse = running_mse/len(validation_data_set)
    mae = running_mae/len(validation_data_set)
    return mse, mae



def validation(model, validation_data_set, mse_criterion, mae_criterion, device, batch_size):
    model.eval()
    running_mse = 0.0
    running_mae = 0.0

    for index in range(0, len(validation_data_set), batch_size):
        batch = validation_data_set[index:index + batch_size]
        data, targets = get_info_from_batch(batch)
        data = torch.stack(data)
        labels = utilities.get_labels_from_targets(targets)
        labels = torch.FloatTensor(labels).to(device)
        
        output = model(data).flatten()

        running_mse += mse_criterion(output, labels).item()
        running_mae += mae_criterion(output, labels).item()

    mse = running_mse/len(validation_data_set)
    mae = running_mae/len(validation_data_set)
    return mse, mae
    
    
def batch_validation_aggregating_cnn(model, batch_validation_data_set, mse_criterion, mae_criterion, device):
    model.eval()
    running_mse = 0.0
    running_mae = 0.0
    all_labels, all_predictions = [], []
    
    for batch in batch_validation_data_set:
        data, targets = batch['data'].to(device), batch['label']
        data = torch.unsqueeze(data, dim=0)
        labels = utilities.get_labels_from_targets(targets)
        labels = torch.FloatTensor(labels).to(device)
        label = torch.max(labels)
        label = torch.unsqueeze(label, dim=0)
        
        output = model(data)
        
        running_mse += mse_criterion(output, label).item()
        running_mae += mae_criterion(output, label).item()
        
        all_labels.append(label.item())
        all_predictions.append(output.item())

    mse = running_mse/len(batch_validation_data_set)
    mae = running_mae/len(batch_validation_data_set)
    return mse, mae, all_labels, all_predictions


def batch_validation(model, batch_validation_data_set, mse_criterion, mae_criterion, device):
    model.eval()
    running_mse = 0.0
    running_mae = 0.0
    all_labels, all_predictions = [], []

    for batch in batch_validation_data_set:
        data, targets = batch['data'], batch['label']
        labels = utilities.get_labels_from_targets(targets)
        labels = torch.FloatTensor(labels).to(device)
        
        # This is to prevent cuda memory issues for large batches
        max_prediction = 0
        for image in data:
            image = torch.unsqueeze(image, dim=0).to(device)
            output = model(image).flatten()
            max_prediction = max(max_prediction, output.item())

        max_prediction = torch.tensor(max_prediction).to(device)
        max_label = torch.max(labels)
        
        running_mse += mse_criterion(max_prediction, max_label).item()
        running_mae += mae_criterion(max_prediction, max_label).item()

        all_predictions.append(max_prediction.item())
        all_labels.append(max_label.item())

    mse = running_mse/len(batch_validation_data_set)
    mae = running_mae/len(batch_validation_data_set)
    return mse, mae, all_labels, all_predictions
    
    
def batch_validation_object_detection(model, batch_validation_data_set, mse_criterion, mae_criterion, print_incorrect_images, saving_dir, device):
    model.eval()
    running_mse = 0.0
    running_mae = 0.0
    all_labels, all_predictions = [], []
    count = 0

    for batch in batch_validation_data_set:
        data, targets = batch['data'], batch['label']
        labels = utilities.get_labels_from_targets(targets)

        # This is to prevent cuda memory issues for large batches
        max_prediction = 0
        max_label = 0
        for index, image in enumerate(data):
            image = torch.unsqueeze(image, dim=0).to(device)
            bounding_boxes = model(image)
            prediction = get_predictions(bounding_boxes)[0]
            label = labels[index]
            
            if print_incorrect_images and label != prediction:
                utilities.print_image(torch.squeeze(image), prediction, saving_dir + "incorrect_images/", count, bounding_boxes)
            
            max_prediction = max(max_prediction, prediction)
            max_label = max(max_label, label)
            count += 1
            
        running_mse += mse_criterion(torch.FloatTensor([max_label]), torch.FloatTensor([max_prediction])).item()
        running_mae += mae_criterion(torch.FloatTensor([max_label]), torch.FloatTensor([max_prediction])).item()

        all_predictions.append(max_prediction)
        all_labels.append(max_label)

    mse = running_mse/len(batch_validation_data_set)
    mae = running_mae/len(batch_validation_data_set)
    return mse, mae, all_labels, all_predictions

def train_and_validate(num_epochs, model, model_name, training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, batch_size, is_object_detection, is_aggregating_cnn):
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    huber_loss = nn.HuberLoss()
    lowest_batch_val_mae = None
    saving_dir = saving_dir + "batch_count_" + model_name + "/"

    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch))
        training_data_set.shuffle()
        batch_training_data_set.shuffle()

        #TODO: Use OOP here
        if is_object_detection:
            training_loss = train_object_detection(model, training_data_set, optimizer, device, batch_size)
        elif is_aggregating_cnn:
            training_loss = train_aggregating_cnn(model, batch_training_data_set, huber_loss, optimizer, device, batch_size)
        else:
            training_loss = train(model, training_data_set, huber_loss, optimizer, device, batch_size)
        print("training loss: " + str(training_loss))

        if not is_aggregating_cnn:
            if is_object_detection:
                val_mse, val_mae = validation_object_detection(model, validation_data_set, mse, mae, device, batch_size)
            else:
                val_mse, val_mae = validation(model, validation_data_set, mse, mae, device, batch_size)
            print("validation MSE: " + str(val_mse) + " and MAE: " + str(val_mae))
        
        if is_object_detection:
            batch_val_mse, batch_val_mae, batch_labels, batch_predictions = batch_validation_object_detection(model, batch_validation_data_set, mse, mae, False, saving_dir, device)
        elif is_aggregating_cnn:
            batch_val_mse, batch_val_mae, batch_labels, batch_predictions = batch_validation_aggregating_cnn(model, batch_validation_data_set, mse, mae, device)
        else:
            batch_val_mse, batch_val_mae, batch_labels, batch_predictions = batch_validation(model, batch_validation_data_set, mse, mae, device)
        print("batch validation MSE: " + str(batch_val_mse) + " and MAE: " + str(batch_val_mae))

        if lowest_batch_val_mae is None or batch_val_mae < lowest_batch_val_mae:
            print("Lowest batch validation MAE achieved, saving weights")
            lowest_batch_val_mae = batch_val_mae
            if is_object_detection or is_aggregating_cnn:
                torch.save(model.state_dict(), saving_dir + model_name + ".pt")
            else:
                torch.save(model.module.state_dict(), saving_dir + model_name + ".pt")
            utilities.print_regression_analysis(batch_labels, batch_predictions, model_name + "_Batch_Validation", saving_dir)


# Declaring Constants
num_epochs = 10
batch_size = 50
object_detection_batch_size = 5
data_dir = "/nfs/stak/users/isonc/hpc-share/saved_data/training_animal_count/"
saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/oregon_wildlife_identification/"

print(torch.__version__)
print(torchvision.__version__)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

training_data, training_labels, validation_data, validation_labels, batch_training_data, batch_training_labels, batch_validation_data, batch_validation_labels = fetch_training_data(data_dir)
training_data_set = ImageDataSet(training_data, training_labels)
validation_data_set = ImageDataSet(validation_data, validation_labels)
batch_training_data_set = ImageDataSet(batch_validation_data, batch_validation_labels)
batch_validation_data_set = ImageDataSet(batch_validation_data, batch_validation_labels)

# Declaring Models
resnet34 = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
in_features = resnet34.fc.in_features
resnet34.fc = nn.Linear(in_features, 1)

#cnn = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
#cnn_wrapper = CNNWrapper(cnn)

resnet50 = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
in_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(in_features, 1)

resnet152 = models.resnet152(weights = models.ResNet152_Weights.DEFAULT)
in_features = resnet152.fc.in_features
resnet152.fc = nn.Linear(in_features, 1)

vit_l_16 = models.vit_l_16(weights = models.ViT_L_16_Weights.DEFAULT)
in_features = vit_l_16.heads[0].in_features
vit_l_16.heads[0] = nn.Linear(in_features, 1)

faster_rcnn = models.detection.fasterrcnn_resnet50_fpn_v2(weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

ssd = models.detection.ssd300_vgg16(weights=models.detection.SSD300_VGG16_Weights.DEFAULT)

retina_net = models.detection.retinanet_resnet50_fpn_v2(weights=models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)

if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    resnet34 = nn.DataParallel(resnet34)
    resnet50 = nn.DataParallel(resnet50)
    resnet152 = nn.DataParallel(resnet152)
    vit_l_16 = nn.DataParallel(vit_l_16)
    #cnn_wrapper = nn.DataParallel(cnn_wrapper)

# Training
print("\nTraining and Validating ResNet34")
train_and_validate(num_epochs, resnet34, "ResNet34", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, batch_size, False, False)

#max_batch_size = 100
#embedding_size = 512
#aggregating_cnn = AggregatingCNN(max_batch_size, embedding_size, resnet34)

#print("\nTraining and Validating Aggregating CNN")
#train_and_validate(num_epochs, aggregating_cnn, "AggregatingCNN", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, batch_size, False, True)

#print("\nTraining and Validating CNN Wrapper")
#train_and_validate(num_epochs, cnn_wrapper, "CNNWrapper", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, batch_size, False, False)

print("\nTraining and Validating ResNet50")
train_and_validate(num_epochs, resnet50, "ResNet50", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, batch_size, False, False)

print("\nTraining and Validating ResNet152")
train_and_validate(num_epochs, resnet152, "ResNet152", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, batch_size, False, False)

print("\nTraining and Validating Vision Transformer Large 16")
train_and_validate(num_epochs, vit_l_16, "ViTL16", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, batch_size, False, False)

print("\nTraining and Validating Faster R-CNN")
train_and_validate(num_epochs, faster_rcnn, "FasterR-CNN", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, object_detection_batch_size, True, False)

print("\nTraining and Validating SSD")
train_and_validate(num_epochs, ssd, "SSD", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, object_detection_batch_size, True, False)

print("\nTraining and Validating RetinaNet")
train_and_validate(num_epochs, retina_net, "RetinaNet", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, object_detection_batch_size, True, False)




