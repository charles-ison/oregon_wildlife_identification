#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from google.colab import drive
from datetime import datetime
from collections import OrderedDict


# # Defining Functions

# In[31]:


def get_image_tensor(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform(image)

def get_image_dictionary(directory):
    image_dictionary = {}
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isdir(file_path):
            leaf_image_dictionary = get_image_dictionary(file_path)
            image_dictionary.update(leaf_image_dictionary)
        elif os.path.isfile(file_path):
            try:
                image = Image.open(file_path)
                datetime = image._getexif()[36867]
                image_tensor = get_image_tensor(image)
                image_dictionary[datetime] = image_tensor
            except:
                print("Truncated image encountered, leaving out of training and testing")
                continue

    image_dictionary = OrderedDict(sorted(image_dictionary.items()))
    return image_dictionary

def get_batched_images(dictionary):
    images, batch_images = [], []
    previous_time_stamp = None
    for key, value in dictionary.items():
        time_stamp = datetime.strptime(key, '%Y:%m:%d %H:%M:%S')
        if previous_time_stamp == None or (time_stamp - previous_time_stamp).total_seconds() < 60:
            batch_images.append(value)
        else:
            images.append(torch.stack(batch_images))
            batch_images = []
            batch_images.append(value)

        previous_time_stamp = time_stamp

    return images

def get_max_predictions(batched_images, model, device):
    max_predictions = []
    for image_batch in batched_images:
        # This is to prevent cuda memory issues for large batches
        max_prediction = 0
        for image in image_batch:
            image = torch.unsqueeze(image, dim=0).to(device)
            output = model(image).flatten()
            max_prediction = max(max_prediction, output.round().item())
        max_predictions.append(max_prediction)
    return max_predictions

def analyze(directory, model, device):
    image_dictionary = get_image_dictionary(directory)
    print("len(image_dictionary):", len(image_dictionary))

    batched_images = get_batched_images(image_dictionary)
    print("len(batched_images):", len(batched_images))

    max_predictions = get_max_predictions(batched_images, model, device)
    print("len(max_predictions):", len(max_predictions))

    predicted_total_num_animals = sum(max_predictions)
    print("predicted_total_num_animals:", predicted_total_num_animals)


# 
# # Declaring Constants

# In[19]:


cottonwood_directory = "Cottonwood_Eastface_6.06_6.13/"
ngilchrist_directory = "NGilchrist_Eastface_6.06_6.13/"
sgilchrist_directory = "SGilchrist_Eastface_6.06_6.13/"

print(torch.__version__)
print(torchvision.__version__)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


# # Loading Data

# In[22]:


# Use this to connect to Google Drive in Google Colab
drive.mount('/content/drive')

# Use this to unzip file in Google Colab
get_ipython().system('unzip -qq drive/MyDrive/SGilchrist_Eastface_6.06_6.13')
get_ipython().system('unzip -qq drive/MyDrive/Cottonwood_Eastface_6.06_6.13')
get_ipython().system('unzip -qq drive/MyDrive/NGilchrist_Eastface_6.06_6.13')


# # Declaring Models

# In[23]:


resnet152 = torch.load("batch_count_ResNet152.pt", map_location=device)

if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    resnet152 = nn.DataParallel(resnet152)


# # Orchestrating

# In[32]:


print("Analyzing Cottonwood")
analyze(cottonwood_directory, resnet152, device)

print("\nAnalyzing NGilchrist")
analyze(ngilchrist_directory, resnet152, device)

print("\nAnalyzing SGilchrist")
analyze(sgilchrist_directory, resnet152, device)


# In[ ]:




