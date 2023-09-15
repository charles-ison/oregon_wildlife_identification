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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from custom_models.aggregating_cnn import AggregatingCNN


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
    

def set_device_for_list_of_dicts(some_list, device):
    for some_dict in some_list:
        some_dict["boxes"] = some_dict["boxes"].to(device)
        some_dict["labels"] = some_dict["labels"].to(device)
        

def get_info_from_batch(batch):
    data, targets = batch['data'], batch['label']
    utilities.set_device_for_list_of_tensors(data, device)
    set_device_for_list_of_dicts(targets, device)
    return data, targets
    
    
def get_labels(targets):
    labels = []
    for target in targets:
        label = target["labels"].size(dim=0)
        labels.append(label)
    return labels
    
    
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
    
    
def train_aggregating_cnn(model, training_data_set, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    num_correct = 0
    
    for batch in training_data_set:
        data, targets = batch['data'].to(device), batch['label']
        labels = get_labels(targets)
        labels = torch.FloatTensor(labels).to(device)
        label = torch.max(labels)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, label)
        running_loss += loss.item()
        num_correct += (output.round() == label).item()
        loss.backward()
        optimizer.step()

    loss = running_loss/len(training_data_set)
    accuracy = num_correct/len(training_data_set)
    return loss, accuracy
    

def train(model, training_data_set, criterion, optimizer, device, batch_size):
    model.train()
    running_loss = 0.0
    num_correct = 0
    
    for index in range(0, len(training_data_set), batch_size):
        batch = training_data_set[index:index + batch_size]
        data, targets = get_info_from_batch(batch)
        data = torch.stack(data)
        labels = get_labels(targets)
        labels = torch.FloatTensor(labels).to(device)
        
        optimizer.zero_grad()
        output = model(data).flatten()

        loss = criterion(output, labels)
        running_loss += loss.item()
        num_correct += (output.round() == labels).sum().item()
        loss.backward()
        optimizer.step()

    loss = running_loss/len(training_data_set)
    accuracy = num_correct/len(training_data_set)
    return loss, accuracy
    

def train_object_detection(model, training_data_set, optimizer, device, batch_size):
    running_loss = 0.0
    num_correct = 0
    for index in range(0, len(training_data_set), batch_size):
        batch = training_data_set[index:index + batch_size]
        data, targets = get_info_from_batch(batch)
        
        model.eval()
        bounding_boxes = model(data)
        
        labels = get_labels(targets)
        predictions = get_predictions(bounding_boxes)
        num_correct += get_num_equal_list_elements(labels, predictions)
        
        model.train()
        optimizer.zero_grad()
        losses_dict = model(data, targets)
        
        sum_losses = sum(loss for loss in losses_dict.values())
        running_loss += sum_losses.item()
        sum_losses.backward()
        optimizer.step()

    loss = running_loss/len(training_data_set)
    accuracy = num_correct/len(training_data_set)
    return loss, accuracy
    
    
def validation_object_detection(model, validation_data_set, device, batch_size):
    running_loss = 0.0
    num_correct = 0

    for index in range(0, len(validation_data_set), batch_size):
        batch = validation_data_set[index:index + batch_size]
        data, targets = get_info_from_batch(batch)
        
        model.train()
        losses_dict = model(data, targets)
        sum_losses = sum(loss for loss in losses_dict.values())
        running_loss += sum_losses.item()
        
        model.eval()
        bounding_boxes = model(data)
        
        labels = get_labels(targets)
        predictions = get_predictions(bounding_boxes)
        num_correct += get_num_equal_list_elements(labels, predictions)

    loss = running_loss/len(validation_data_set)
    accuracy = num_correct/len(validation_data_set)
    return loss, accuracy



def validation(model, validation_data_set, criterion, device, batch_size):
    model.eval()
    running_loss = 0.0
    num_correct = 0

    for index in range(0, len(validation_data_set), batch_size):
        batch = validation_data_set[index:index + batch_size]
        data, targets = get_info_from_batch(batch)
        data = torch.stack(data)
        labels = get_labels(targets)
        labels = torch.FloatTensor(labels).to(device)
        
        output = model(data).flatten()

        loss = criterion(output, labels)
        running_loss += loss.item()
        num_correct += (output.round() == labels).sum().item()

    loss = running_loss/len(validation_data_set)
    accuracy = num_correct/len(validation_data_set)
    return loss, accuracy
    
    
def batch_validation_aggregating_cnn(model, batch_validation_data_set, criterion, device):
    model.eval()
    running_loss = 0.0
    num_correct = 0
    all_labels, all_predictions = [], []
    
    for batch in batch_validation_data_set:
        data, targets = batch['data'].to(device), batch['label']
        labels = get_labels(targets)
        labels = torch.FloatTensor(labels).to(device)
        label = torch.max(labels)
        
        output = model(data)
        
        loss = criterion(output, label)
        running_loss += loss.item()
        num_correct += (output == label).item()
        
        all_labels.append(label.item())
        all_predictions.append(output.item())

    loss = running_loss/len(batch_validation_data_set)
    accuracy = num_correct/len(batch_validation_data_set)
    return loss, accuracy, all_labels, all_predictions


def batch_validation(model, batch_validation_data_set, criterion, print_incorrect_images, data_dir, device):
    model.eval()
    num_correct = 0
    running_loss = 0.0
    all_labels, all_predictions = [], []

    for batch in batch_validation_data_set:
        data, targets = batch['data'], batch['label']
        labels = get_labels(targets)
        labels = torch.FloatTensor(labels).to(device)
        
        # This is to prevent cuda memory issues for large batches
        max_prediction = 0
        for image in data:
            image = torch.unsqueeze(image, dim=0).to(device)
            output = model(image).flatten()
            max_prediction = max(max_prediction, output.round().item())

        max_prediction = torch.tensor(max_prediction).to(device)
        max_label = torch.max(labels)
        
        loss = criterion(max_prediction, max_label)
        running_loss += loss.item()

        if max_prediction == max_label:
            num_correct += 1

        all_predictions.append(max_prediction.item())
        all_labels.append(max_label.item())

    loss = running_loss/len(batch_validation_data_set)
    accuracy = num_correct/len(batch_validation_data_set)
    return loss, accuracy, all_labels, all_predictions
    
    
def batch_validation_object_detection(model, batch_validation_data_set, criterion, print_incorrect_images, saving_dir, device):
    model.eval()
    num_correct = 0
    running_loss = 0.0
    all_labels, all_predictions = [], []
    count = 0

    for batch in batch_validation_data_set:
        data, targets = batch['data'], batch['label']
        labels = get_labels(targets)

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
            
        running_loss += criterion(torch.FloatTensor([max_label]), torch.FloatTensor([max_prediction])).item()
        if max_prediction == max_label:
            num_correct += 1

        all_predictions.append(max_prediction)
        all_labels.append(max_label)

    loss = running_loss/len(batch_validation_data_set)
    accuracy = num_correct/len(batch_validation_data_set)
    return loss, accuracy, all_labels, all_predictions

def train_and_validate(num_epochs, model, model_name, training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, data_dir, saving_dir, batch_size, is_object_detection, is_aggregating_cnn):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    highest_batch_validation_accuracy = 0.0
    saving_dir = saving_dir + "batch_count_" + model_name + "/"

    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch))

        #TODO: Use OOP here
        if is_object_detection:
            training_loss, training_accuracy = train_object_detection(model, training_data_set, optimizer, device, batch_size)
        elif is_aggregating_cnn:
            training_loss, training_accuracy = train_aggregating_cnn(model, batch_training_data_set, criterion, optimizer, device)
        else:
            training_loss, training_accuracy = train(model, training_data_set, criterion, optimizer, device, batch_size)
        print("training loss: " + str(training_loss) + " and training accuracy: " + str(training_accuracy))

        if not is_aggregating_cnn:
            if is_object_detection:
                validation_loss, validation_accuracy = validation_object_detection(model, validation_data_set, device, batch_size)
            else:
                validation_loss, validation_accuracy = validation(model, validation_data_set, criterion, device, batch_size)
            print("validation loss: " + str(validation_loss) + " and validation accuracy: " + str(validation_accuracy))
        
        if is_object_detection:
            batch_validation_loss, batch_validation_accuracy, batch_labels, batch_predictions = batch_validation_object_detection(model, batch_validation_data_set, criterion, False, saving_dir, device)
        elif is_aggregating_cnn:
            batch_validation_loss, batch_validation_accuracy, batch_labels, batch_predictions = batch_validation_aggregating_cnn(model, batch_validation_data_set, criterion, device)
        else:
            batch_validation_loss, batch_validation_accuracy, batch_labels, batch_predictions = batch_validation(model, batch_validation_data_set, criterion, False, saving_dir, device)
        print("batch validation loss (MSE): " + str(batch_validation_loss) + " and batch validation accuracy: "+ str(batch_validation_accuracy))

        if highest_batch_validation_accuracy < batch_validation_accuracy:
            print("Highest batch validation accuracy achieved, saving weights")
            highest_batch_validation_accuracy = batch_validation_accuracy
            if is_object_detection or is_aggregating_cnn:
                torch.save(model.state_dict(), saving_dir + model_name + ".pt")
            else:
                torch.save(model.module.state_dict(), saving_dir + model_name + ".pt")
            print_validation_analysis(batch_labels, batch_predictions, model_name, data_dir, saving_dir)


# Declaring Constants
num_epochs = 3
batch_size = 5
json_file_name = "animal_count_key.json"
data_dir = "/nfs/stak/users/isonc/hpc-share/saved_data/2022_Cottonwood_Eastface_bounding_boxes/"
saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/oregon_wildlife_identification/"

print(torch.__version__)
print(torchvision.__version__)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

training_data, validation_data, training_labels, validation_labels, batch_training_data, batch_training_labels, batch_validation_data, batch_validation_labels = utilities.fetch_data(data_dir, json_file_name, False, True, True)
training_data_set = utilities.image_data_set(training_data, training_labels)
validation_data_set = utilities.image_data_set(validation_data, validation_labels)
batch_training_data_set = utilities.image_data_set(batch_validation_data, batch_validation_labels)
batch_validation_data_set = utilities.image_data_set(batch_validation_data, batch_validation_labels)

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

faster_rcnn = models.detection.fasterrcnn_resnet50_fpn_v2(weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

ssd = models.detection.ssd300_vgg16(weights=models.detection.SSD300_VGG16_Weights.DEFAULT)

retina_net = models.detection.retinanet_resnet50_fpn_v2(weights=models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)

max_batch_size = 100
aggregating_cnn = AggregatingCNN(max_batch_size)

if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    resnet50 = nn.DataParallel(resnet50)
    resnet152 = nn.DataParallel(resnet152)
    vit_l_16 = nn.DataParallel(vit_l_16)

# Training
print("\nTraining and Validating Aggregating CNN")
train_and_validate(num_epochs, aggregating_cnn, "AggregatingCNN", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, data_dir, saving_dir, batch_size, False, True)

print("\nTraining and Validating Faster R-CNN")
train_and_validate(num_epochs, faster_rcnn, "FasterR-CNN", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, data_dir, saving_dir, batch_size, True, False)

print("\nTraining and Validating SSD")
train_and_validate(num_epochs, ssd, "SSD", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, data_dir, saving_dir, batch_size, True, False)

print("\nTraining and Validating RetinaNet")
train_and_validate(num_epochs, retina_net, "RetinaNet", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, data_dir, saving_dir, batch_size, True, False)

print("\nTraining and Validating ResNet50")
train_and_validate(num_epochs, resnet50, "ResNet50", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, data_dir, saving_dir, batch_size, False, False)

print("\nTraining and Validating ResNet152")
train_and_validate(num_epochs, resnet152, "ResNet152", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, data_dir, saving_dir, batch_size, False, False)

print("\nTraining and Validating Vision Transformer Large 16")
train_and_validate(num_epochs, vit_l_16, "ViTL16", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, data_dir, saving_dir, batch_size, False, False)




