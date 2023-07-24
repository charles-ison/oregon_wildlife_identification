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
import matplotlib as matplotlib
import json
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from operator import itemgetter
from datetime import datetime
from pycocotools.coco import COCO

class image_data_set(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'data': self.data[index], 'label': self.labels[index]}

def get_image_tensor(file_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(file_path)
    return transform(image)

def remove_images_with_no_datetime(images):
    new_images = []
    for image in images:
        if "datetime" in image:
            new_images.append(image)
    return new_images

def get_sorted_images(images):
    images = remove_images_with_no_datetime(images)
    return sorted(images, key=lambda image: image["datetime"])

def flatten_list(data):
    return [image for batch in data for image in batch]

#TODO: Code smell here passing around these flags, should probably be refactored into separate classes
def get_data_sets(data_dir, json_file_name, is_classification, is_training):
    coco = COCO(data_dir + json_file_name)
    images = coco.loadImgs(coco.getImgIds())
    sort_images = get_sorted_images(images)

    batch_data, batch_labels, data, labels = [], [], [], []
    previous_time_stamp = None
    
    for image in sort_images:
        time_stamp = datetime.strptime(image["datetime"], '%Y:%m:%d %H:%M:%S')
        file_name = image["file_name"]
        file_path = data_dir + file_name
        
        annotation_id_list = coco.getAnnIds(imgIds=[image["id"]])
        annotation_list = coco.loadAnns(annotation_id_list)
        
        if len(annotation_list) != 0 and image["id"] == annotation_list[0]["image_id"] and os.path.isfile(file_path):
            label = annotation_list[0]["category_id"]
            
            if is_classification and label > 0:
                label = 1
                
            image_tensor = None
            try:
                image_tensor = get_image_tensor(file_path)
            except:
                print("Problematic image encountered, leaving out of training and testing")
                continue

            if previous_time_stamp == None or (time_stamp - previous_time_stamp).total_seconds() < 60:
                batch_data.append(image_tensor)
                batch_labels.append(label)
            else:
                data.append(torch.stack(batch_data))
                
                if is_classification:
                    labels.append(torch.LongTensor(batch_labels))
                else:
                    labels.append(torch.FloatTensor(batch_labels))

                batch_data, batch_labels = [], []
                batch_data.append(image_tensor)
                batch_labels.append(label)

            previous_time_stamp = time_stamp
            
    data.append(torch.stack(batch_data))
    
    if is_classification:
        labels.append(torch.LongTensor(batch_labels))
    else:
        labels.append(torch.FloatTensor(batch_labels))
    
    if is_training:
        batch_training_data, batch_testing_data, batch_training_labels, batch_testing_labels = train_test_split(data, labels, test_size = 0.20)
        training_data = flatten_list(batch_training_data)
        testing_data = flatten_list(batch_testing_data)
        training_labels = flatten_list(batch_training_labels)
        testing_labels = flatten_list(batch_testing_labels)

        print("\nNumber of training photos: ", len(training_data))
        print("Number of testing photos: ", len(testing_data))
        print("Number of batches for testing: ", len(batch_testing_data))

        return training_data, testing_data, training_labels, testing_labels, batch_testing_data, batch_testing_labels
    else:
        individual_data = flatten_list(data)
        individual_labels = flatten_list(labels)
        
        print("Number of batches for verification: ", len(data))
        print("Number of individual photos for verification: ", len(individual_data))
        return data, labels, individual_data, individual_labels

def print_image(image_tensor, prediction, data_dir, index):
    image_file_name = data_dir + str(prediction) + "_" + str(index) + ".png"
    plt.imshow(image_tensor[0].cpu(), cmap="gray")
    plt.title("Predicted: " + str(prediction))
    plt.imsave(image_file_name, image_tensor[0].cpu(), cmap="gray")

def create_heat_map(grad_cam, image, prediction, saving_dir):
    # Just processing one image at a time
    grayscale_cam = grad_cam(input_tensor=image.unsqueeze(dim=0))[0]
        
    # Transformations required for Grad Cam
    image = image.reshape([224, 224, 3]).cpu().numpy()
    image_with_heat_map = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    
    title = "grad_cam_for_prediction_" + str(prediction.item())
    image_file_name = saving_dir + title + ".png"
    plt.title(title)
    
    plt.imshow(image[:, :, 0], cmap="gray")
    plt.imsave(image_file_name, image[:, :, 0], cmap="gray")
