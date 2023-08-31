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


def get_image_tensor(file_path, is_training):
    image = Image.open(file_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    tensor = transform(image)
    
    if is_training:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15)
        ])
        tensor = transform(tensor)
        
    return tensor


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
    
        
def get_object_detection_label(annotation_list):
    bounding_boxes = torch.zeros(len(annotation_list), 4)
    labels = torch.zeros(len(annotation_list))
    for index, annotation in enumerate(annotation_list):
        if "bbox" in annotation:
            bounding_box = annotation["bbox"]
            bounding_box[1] = bounding_box[1] - bounding_box[3]
            bounding_box[2] = bounding_box[0] + bounding_box[2]
            bounding_box[3] = bounding_box[1] + bounding_box[3]
            bounding_boxes[index] = torch.FloatTensor(bounding_box).reshape(1, 4)
    
            label = annotation["category_id"]
            labels[index] = label
    return {"boxes":  bounding_boxes, "labels": labels.long()}
    
    
def get_label(annotation_list, image, is_classification, is_object_detection):
    if is_object_detection:
        return get_object_detection_label(annotation_list)
    elif image["id"] == annotation_list[0]["image_id"]:
        label = annotation_list[0]["category_id"]
        if is_classification and label > 1:
            return 1
        else:
            return label
    raise Exception("No label found for image.")
    
    
def append_batch_labels(labels, batch_labels, is_classification, is_object_detection):
    if is_classification:
        labels.append(torch.LongTensor(batch_labels))
    elif is_object_detection:
        labels.append(batch_labels)
    else:
        labels.append(torch.FloatTensor(batch_labels))
    

#TODO: Code smell here passing around these flags, should probably be refactored into separate classes
def fetch_data(data_dir, json_file_name, is_classification, is_object_detection, is_training):
    coco = COCO(data_dir + json_file_name)
    images = coco.loadImgs(coco.getImgIds())
    sorted_images = get_sorted_images(images)

    batch_data, batch_labels, data, labels = [], [], [], []
    previous_time_stamp = None
    
    for image in sorted_images:
        time_stamp = datetime.strptime(image["datetime"], '%Y:%m:%d %H:%M:%S')
        file_name = image["file_name"]
        file_path = data_dir + file_name
        
        annotation_id_list = coco.getAnnIds(imgIds=[image["id"]])
        annotation_list = coco.loadAnns(annotation_id_list)

        if os.path.isfile(file_path):
            image_tensor = None
            label = None
            try:
                image_tensor = get_image_tensor(file_path, is_training)
                label = get_label(annotation_list, image, is_classification, is_object_detection)
            except:
                print("Problematic image or label encountered, leaving out of training and testing")
                continue

            if previous_time_stamp == None or (time_stamp - previous_time_stamp).total_seconds() < 60:
                batch_data.append(image_tensor)
                batch_labels.append(label)
            else:
                data.append(torch.stack(batch_data))
                append_batch_labels(labels, batch_labels, is_classification, is_object_detection)

                batch_data, batch_labels = [], []
                batch_data.append(image_tensor)
                batch_labels.append(label)

            previous_time_stamp = time_stamp
            
    data.append(torch.stack(batch_data))
    append_batch_labels(labels, batch_labels, is_classification, is_object_detection)
    
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
    image_tensor = image_tensor.permute(1, 2, 0).cpu()
    normalize = plt.Normalize()
    image_tensor = normalize(image_tensor)
    
    title = str(prediction) + "_" + str(index)
    image_file_name = data_dir + title + ".png"
    plt.imshow(image_tensor)
    plt.title(title)
    plt.imsave(image_file_name, image_tensor)


def create_heat_map(grad_cam, image, prediction, label, saving_dir, identifier):
    
    title = "grad_cam_for_prediction_" + str(int(prediction)) + "_and_label_" + str(int(label)) + "_identifier_" + str(identifier)
    image_file_name = saving_dir + title + ".png"
    plt.title(title)
    
    # Just processing one image at a time
    grayscale_cam = grad_cam(input_tensor=image.unsqueeze(dim=0))[0]
        
    # Transformations required for Grad Cam
    image = image.permute(1, 2, 0).cpu().numpy()
    image_with_heat_map = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    
    normalize = plt.Normalize()
    normalized_image_with_heat_map = normalize(image_with_heat_map)
    
    plt.imshow(normalized_image_with_heat_map)
    plt.imsave(image_file_name, normalized_image_with_heat_map)
