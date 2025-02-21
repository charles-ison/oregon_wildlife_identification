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
import matplotlib.patches as patches
import json
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from operator import itemgetter
from datetime import datetime
from pycocotools.coco import COCO


def print_classification_analysis(labels, predictions, title, saving_dir):
    subplot = plt.subplot()

    cf_matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cf_matrix = np.flip(cf_matrix, axis=0)
    sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues')

    subplot.set_xlabel('Predictions')
    subplot.set_ylabel('Labels')
    title = title + "_Confusion_Matrix"
    subplot.set_title(title)
    subplot.xaxis.set_ticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    subplot.yaxis.set_ticklabels([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    plot_file_name = saving_dir + title + ".png"
    plt.savefig(plot_file_name)
    plt.show()
    plt.close()

    accuracy = accuracy_score(labels, predictions)
    print(title + " Accuracy: " + str(accuracy))

    precision, recall, f_score, support = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0.0)
    print(title + " Precision: " + str(precision))
    print(title + " Recall: " + str(recall))
    print(title + " F-Score: " + str(f_score))
    
    r2 = r2_score(labels, predictions)
    print(title + " R^2: " + str(r2))


def jitter(some_list):
    return some_list + 0.02 * np.random.randn(len(some_list))


def print_regression_analysis(labels, predictions, title, saving_dir):
    subplot = plt.subplot()
    
    subplot.scatter(predictions, jitter(labels), marker='o', s=15, alpha=0.1, c='blue')
    max_value = max(max(labels), max(predictions)) + 0.5
    subplot.plot([0, max_value], [0, max_value], color="red")
    
    subplot.set_xlabel('Predictions')
    subplot.set_ylabel('Labels')
    r2 = r2_score(labels, predictions)
    print("R^2: ", r2)
    r2_text = "R-Squared: " + str(r2)
    subplot.text(0, max_value, r2_text)
    title = title + "_Predicted_Vs_Actual"
    subplot.set_title(title)
    plot_file_name = saving_dir + title + ".png"
    plt.savefig(plot_file_name)
    plt.show()
    plt.close()
        

def set_device_for_list_of_tensors(some_list, device):
    for index, tensor in enumerate(some_list):
        some_list[index] = tensor.to(device)
    return some_list
    

def get_image_tensor(file_path, new_image_height_and_width):
    image = Image.open(file_path)
    transform = transforms.Compose([
        transforms.Resize((new_image_height_and_width, new_image_height_and_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

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
    

def rescale_bounding_box(bounding_box, old_image_width, old_image_height, new_image_height_and_width):
    bounding_box[0] = (bounding_box[0] * new_image_height_and_width) / old_image_width
    bounding_box[1] = (bounding_box[1] * new_image_height_and_width) / old_image_height
    bounding_box[2] = (bounding_box[2] * new_image_height_and_width) / old_image_width
    bounding_box[3] = (bounding_box[3] * new_image_height_and_width) / old_image_height
    return bounding_box
        
        
def get_label(annotation_list, image, new_image_height_and_width):
    annotation_list_len = 0
    for annotation in annotation_list:
        if annotation["category_id"] == 1:
            annotation_list_len += 1
            
    bounding_boxes = torch.zeros(annotation_list_len, 4)
    labels = torch.zeros(annotation_list_len).long()
    old_image_width = image["width"]
    old_image_height = image["height"]
    for index, annotation in enumerate(annotation_list):
        if "bbox" in annotation and annotation["category_id"] == 1:
            bounding_box = rescale_bounding_box(annotation["bbox"], old_image_width, old_image_height, new_image_height_and_width)
            bounding_box[1] = bounding_box[1] - bounding_box[3]
            bounding_box[2] = bounding_box[0] + bounding_box[2]
            bounding_box[3] = bounding_box[1] + bounding_box[3]
            bounding_boxes[index] = torch.FloatTensor(bounding_box).reshape(1, 4)
    
            label = annotation["category_id"]
            labels[index] = label
    return {"boxes":  bounding_boxes, "labels": labels}
    
    
def append_batch_labels(labels, batch_labels):
    labels.append(batch_labels)
        
def get_labels_from_targets(targets):
    labels = []
    for target in targets:
        label = target["labels"].size(dim=0)
        labels.append(label)
    return labels
    

def plot_histogram(training_labels, data_dir):
    labels = get_labels_from_targets(training_labels)
    max_label = max(labels)
    bins = np.arange(max_label)
    subplot = plt.subplot()
    subplot.set_xlim(0, max_label)
    subplot.hist(labels, bins = bins, align = "mid")
    subplot.set_xlabel('Labels')
    subplot.set_ylabel('Counts')
    
    title = "Training_Labels_Histogram"
    subplot.set_title(title)
    plot_file_name = data_dir + title + ".png"
    plt.savefig(plot_file_name)
    plt.show()
    plt.close()
    

#TODO: Code smell here passing around these flags, should probably be refactored into separate classes
def fetch_data(data_dir, json_file_name, is_training, is_supplemental):
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
                new_image_height_and_width = 224
                label = get_label(annotation_list, image, new_image_height_and_width)
                if is_training and label["labels"].size(dim=0) == 0 and is_supplemental:
                    continue
                image_tensor = get_image_tensor(file_path, new_image_height_and_width)
            except:
                print("Problematic image or label encountered, leaving out.")
                continue

            if previous_time_stamp == None or (time_stamp - previous_time_stamp).total_seconds() < 60:
                batch_data.append(image_tensor)
                batch_labels.append(label)
            else:
                data.append(torch.stack(batch_data))
                append_batch_labels(labels, batch_labels)

                batch_data, batch_labels = [], []
                batch_data.append(image_tensor)
                batch_labels.append(label)

            previous_time_stamp = time_stamp
        else:
            print("No file found for: ", file_path)
            continue
        
    data.append(torch.stack(batch_data))
    append_batch_labels(labels, batch_labels)
    
    if is_training:
        batch_training_data, batch_validation_data, batch_training_labels, batch_validation_labels = train_test_split(data, labels, test_size = 0.20)
        training_data = flatten_list(batch_training_data)
        validation_data = flatten_list(batch_validation_data)
        training_labels = flatten_list(batch_training_labels)
        validation_labels = flatten_list(batch_validation_labels)
        return training_data, training_labels, validation_data, validation_labels, batch_training_data, batch_training_labels, batch_validation_data, batch_validation_labels
    else:
        individual_data = flatten_list(data)
        individual_labels = flatten_list(labels)
        
        print("Number of batches for verification: ", len(data))
        print("Number of individual images for verification: ", len(individual_data))
        return data, labels, individual_data, individual_labels
    

def print_image(image_tensor, prediction, saving_dir, index, bounding_boxes = None):
    fig, ax = plt.subplots()
    image_tensor = image_tensor.permute(1, 2, 0).cpu()
    normalize = plt.Normalize()
    image_tensor = normalize(image_tensor)
    
    title = str(prediction) + "_" + str(index)
    plt.title(title)
    plt.imshow(image_tensor)
    
    if bounding_boxes is not None:
        boxes = bounding_boxes[0]["boxes"]
        # Can use scores here to only print bounding boxes > 0.5
        scores = bounding_boxes[0]["scores"]
        for box_index, box in enumerate(boxes):
            width = box[2] - box[0]
            height = box[3] - box[1]
            ax.add_patch(patches.Rectangle((box[0].item(), box[1].item()), width.item(), height.item(), linewidth=1, edgecolor='r', facecolor='none'))
    
    image_file_name = saving_dir + title + ".png"
    plt.savefig(image_file_name)
    plt.close(fig)


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
    
    
def set_device_for_list_of_dicts(some_list, device):
    for some_dict in some_list:
        some_dict["boxes"] = some_dict["boxes"].to(device)
        some_dict["labels"] = some_dict["labels"].to(device)
        

def get_info_from_batch(batch, device):
    data, targets = batch['data'], batch['label']
    set_device_for_list_of_tensors(data, device)
    set_device_for_list_of_dicts(targets, device)
    return data, targets
