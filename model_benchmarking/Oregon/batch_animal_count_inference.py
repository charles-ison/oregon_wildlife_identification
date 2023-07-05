import os
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime, timedelta
from collections import OrderedDict

import matplotlib.pyplot as plt
def print_image(image_tensor, prediction, index):
    image_file_name = str(prediction) + "_" + str(index) + ".png"
    plt.title("Predicted " + str(prediction) + " Animals Present")
    plt.imshow(image_tensor[0][0].cpu(), cmap="gray")
    plt.imsave(image_file_name, image_tensor[0][0].cpu(), cmap="gray")

def get_image_tensor(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform(image)
    
def handle_duplicate_timestamps(timestamp, image_dictionary):
    if timestamp in image_dictionary.keys():
        timestamp += timedelta(milliseconds=1)
        return handle_duplicate_timestamps(timestamp, image_dictionary)
    else:
        return timestamp
        
def get_timestamp(image, image_dictionary):
    timestamp = image._getexif()[36867]
    timestamp = datetime.strptime(timestamp, '%Y:%m:%d %H:%M:%S')
    return handle_duplicate_timestamps(timestamp, image_dictionary)

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
                timestamp = get_timestamp(image, image_dictionary)
                image_dictionary[timestamp] = get_image_tensor(image)
            except:
                print("Problematic image encountered, leaving out of training and testing")
                continue

    image_dictionary = OrderedDict(sorted(image_dictionary.items()))
    return image_dictionary

def get_batched_images(dictionary):
    images, batch_images = [], []
    previous_time_stamp = None
    for key, value in dictionary.items():
        if previous_time_stamp == None or (key - previous_time_stamp).total_seconds() < 60:
            batch_images.append(value)
        else:
            images.append(torch.stack(batch_images))
            batch_images = []
            batch_images.append(value)

        previous_time_stamp = key

    images.append(torch.stack(batch_images))
    return images

def get_max_predictions(batched_images, model, device):
    max_predictions = []
    count = 0
    for image_batch in batched_images:
        # This is to prevent cuda memory issues for large batches
        max_prediction = 0
        for image in image_batch:
            image = torch.unsqueeze(image, dim=0).to(device)
            output = model(image).flatten()
            
            #print_image(image, output.round().item(), count)
            count += 1
            
            max_prediction = max(max_prediction, output.round().item())
        max_predictions.append(max_prediction)
    return max_predictions

def analyze(directory, model, device):
    image_dictionary = get_image_dictionary(directory)
    print("len(image_dictionary):", len(image_dictionary))

    batched_images = get_batched_images(image_dictionary)
    print("len(batched_images):", len(batched_images))

    max_predictions = get_max_predictions(batched_images, model, device)
    predicted_total_num_animals = sum(max_predictions)
    print("predicted_total_num_animals:", predicted_total_num_animals)

# Declaring Constants
cottonwood_directory = "/nfs/stak/users/isonc/hpc-share/saved_data/Cottonwood_Eastface_6.06_6.13/"
ngilchrist_directory = "/nfs/stak/users/isonc/hpc-share/saved_data/NGilchrist_Eastface_6.06_6.13/"
sgilchrist_directory = "/nfs/stak/users/isonc/hpc-share/saved_data/SGilchrist_Eastface_6.06_6.13/"
MP152_ODOT003_eastface_directory = "/nfs/stak/users/isonc/hpc-share/saved_data/MP152_ODOT003_EASTFACE/"

model_weights_path = "/nfs/stak/users/isonc/hpc-share/saved_models/batch_count_ResNet50/ResNet50.pt"

print(torch.__version__)
print(torchvision.__version__)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Declaring Models
# Have follow same steps used to create model during training
model = models.resnet50()
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)

#Loading trained model weights
model.load_state_dict(torch.load(model_weights_path))

if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    model = nn.DataParallel(model)
    
model.eval()
model.to(device)

# Orchestrating
print("\nAnalyzing Cottonwood")
#analyze(cottonwood_directory, model, device)

print("\nAnalyzing NGilchrist")
analyze(ngilchrist_directory, model, device)

print("\nAnalyzing SGilchrist")
analyze(sgilchrist_directory, model, device)

print("\nAnalyzing MP152_ODOT003_EASTFACE")
analyze(MP152_ODOT003_eastface_directory, model, device)
