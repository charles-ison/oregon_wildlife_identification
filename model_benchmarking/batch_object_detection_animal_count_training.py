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
from torchvision.utils import draw_bounding_boxes


def print_testing_analysis(all_labels, all_predictions, title, saving_dir):
    plt.figure()
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
    

def set_device_for_list_of_dicts(some_list, device):
    for some_dict in some_list:
        some_dict["boxes"] = some_dict["boxes"].to(device)
        some_dict["labels"] = some_dict["labels"].to(device)
    

def set_device_for_list_of_tensors(some_list, device):
    for index, tensor in enumerate(some_list):
        some_list[index] = tensor.to(device)
        

def get_info_from_batch(batch):
    data, targets = batch['data'], batch['label']
    set_device_for_list_of_tensors(data, device)
    set_device_for_list_of_dicts(targets, device)
    return data, targets
    
    
def get_predictions_and_labels(bounding_boxes, targets):
    num_correct = 0
    labels, predictions = [], []
    for box_index, boxes in enumerate(bounding_boxes):
        num_animals = 0
        for score_index, score in enumerate(boxes["scores"]):
            if score > 0.5 and boxes["labels"][score_index] == 1:
                num_animals += 1
        label = targets[box_index]["labels"].size(dim=0)
        labels.append(label)
        predictions.append(num_animals)
        if num_animals == label:
            num_correct += 1
    return labels, predictions, num_correct


def train(model, training_data_set, batch_size, optimizer, device):
    running_loss = 0.0
    num_correct = 0
    for index in range(0, len(training_data_set), batch_size):
        batch = training_data_set[index:index + batch_size]
        data, targets = get_info_from_batch(batch)
        
        model.eval()
        bounding_boxes = model(data)
        _, _, batch_num_correct = get_predictions_and_labels(bounding_boxes, targets)
        num_correct += batch_num_correct
        
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


def test(model, testing_data_set, batch_size, print_incorrect_images, saving_dir, device):
    model.eval()
    running_loss = 0.0
    num_correct = 0
    labels, predictions = [], []

    for index in range(0, len(testing_data_set), batch_size):
        batch = testing_data_set[index:index + batch_size]
        data, targets = get_info_from_batch(batch)
        
        model.train()
        losses_dict = model(data, targets)
        sum_losses = sum(loss for loss in losses_dict.values())
        running_loss += sum_losses.item()
        
        model.eval()
        bounding_boxes = model(data)
        batch_labels, batch_predictions, batch_num_correct = get_predictions_and_labels(bounding_boxes, targets)
        
        num_correct += batch_num_correct
        labels.extend(batch_labels)
        predictions.extend(batch_predictions)

    loss = running_loss/len(testing_data_set)
    accuracy = num_correct/len(testing_data_set)
    return loss, accuracy, labels, predictions


def test_batch(model, batch_testing_data_set, print_incorrect_images, saving_dir, device):
    model.eval()
    num_correct = 0
    running_loss = 0.0
    all_labels, all_predictions = [], []
    mse = nn.MSELoss()
    count = 0

    for batch in batch_testing_data_set:
        data, targets = batch['data'], batch['label']

        # This is to prevent cuda memory issues for large batches
        max_prediction = 0
        max_label = 0
        for image in data:
            image = torch.unsqueeze(image, dim=0).to(device)
            bounding_boxes = model(image)
            labels, predictions, _ = get_predictions_and_labels(bounding_boxes, targets)
            
            if print_incorrect_images:
                utilities.print_image(torch.squeeze(image), predictions[0], saving_dir, count, bounding_boxes[0]["boxes"])
            
            max_prediction = max(max_prediction, predictions[0])
            max_label = max(max_label, labels[0])
            count += 1
            
        running_loss += mse(torch.FloatTensor([max_label]), torch.FloatTensor([max_prediction])).item()
        if max_prediction == max_label:
            num_correct += 1

        all_predictions.append(max_prediction)
        all_labels.append(max_label)

    loss = running_loss/len(batch_testing_data_set)
    accuracy = num_correct/len(batch_testing_data_set)
    return loss, accuracy, all_labels, all_predictions


def train_and_test(num_epochs, model, model_name, training_data_set, testing_data_set, batch_testing_data_set, batch_size, device, saving_dir):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    highest_batch_testing_accuracy = 0.0
    saving_dir = saving_dir + "batch_count_" + model_name + "/"

    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch))

        training_loss, training_accuracy = train(model, training_data_set, batch_size, optimizer, device)
        print("training loss: " + str(training_loss) + " and training accuracy: " + str(training_accuracy))

        testing_loss, testing_accuracy, _, _ = test(model, testing_data_set, batch_size, False, saving_dir, device)
        print("testing loss: " + str(testing_loss) + " and testing accuracy: " + str(testing_accuracy))

        should_print_images = (epoch == num_epochs - 1)
        batch_testing_loss, batch_testing_accuracy, batch_labels, batch_predictions = test_batch(model, batch_testing_data_set, should_print_images, saving_dir, device)
        print("batch testing loss (MSE): " + str(batch_testing_loss) + " and batch testing accuracy: "+ str(batch_testing_accuracy))

        if highest_batch_testing_accuracy < batch_testing_accuracy:
            print("Highest batch testing accuracy achieved, saving weights")
            highest_batch_testing_accuracy = batch_testing_accuracy
            torch.save(model.state_dict(), saving_dir + model_name + ".pt")
            print_testing_analysis(batch_labels, batch_predictions, model_name, saving_dir)


# Declaring Constants
num_epochs = 5
batch_size = 5
json_file_name = "animal_count_key.json"
data_dir = "/nfs/stak/users/isonc/hpc-share/saved_data/object_detection_testing/"
saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/"

print(torch.__version__)
print(torchvision.__version__)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

training_data, testing_data, training_labels, testing_labels, batch_testing_data, batch_testing_labels = utilities.fetch_data(data_dir, json_file_name, False, True, True)
training_data_set = utilities.image_data_set(training_data, training_labels)
testing_data_set = utilities.image_data_set(testing_data, testing_labels)
batch_testing_data_set = utilities.image_data_set(batch_testing_data, batch_testing_labels)

# Declaring Models
# TODO: YOLO and SSD
faster_rcnn = models.detection.fasterrcnn_resnet50_fpn_v2(weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
in_features = faster_rcnn.roi_heads.box_predictor.cls_score.in_features
faster_rcnn.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, 10)

# TODO: Need to use distributed data parallel for object detection models
# if torch.cuda.device_count() > 1:
    #print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    #faster_rcnn = nn.DataParallel(faster_rcnn)

# Training
print("\nTraining and Testing Faster R-CNN")
train_and_test(num_epochs, faster_rcnn, "FasterR-CNN", training_data_set, testing_data_set, batch_testing_data_set, batch_size, device, saving_dir)

