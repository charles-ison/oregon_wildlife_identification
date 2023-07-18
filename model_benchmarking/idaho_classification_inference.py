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
import urllib.request
import shutil
import json
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class image_data_set(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        return {'data': self.data[index], 'label': self.labels[index]}

def download_zip(downloaded_data_dir, zip_name, blob_name):    
    if not os.path.isfile(downloaded_data_dir + zip_name):
        print("Downloading zip: " + zip_name)
        zip_file_name = zip_name + ".zip"
        zip_to_download = blob_name + zip_file_name
        download_zip_command = "azcopy cp '%s' '%s'" % (zip_to_download, downloaded_data_dir)
        os.system(download_zip_command)
        shutil.unpack_archive(downloaded_data_dir + zip_file_name, downloaded_data_dir)
        os.remove(downloaded_data_dir + zip_file_name)
    else:
        print("Required zip already downloaded")
        
def get_image_tensor(file_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(file_path)
    return transform(image)
    
def get_data_sets(images, annotations, unzipped_data_dir, categories_to_label_dict): 
    num_wildlife_present_images = 0

    data, labels, batch_data, batch_labels = [], [], [], []
    for index, image in enumerate(images):
        file_name = image["file_name"]
        file_path = unzipped_data_dir + file_name
        category_id = annotations[index]["category_id"]
        label = categories_to_label_dict[category_id]
        
        if image["frame_num"] == 0 and index != 0:
            data.append(batch_data)
            labels.append(batch_labels)
            batch_data, batch_labels = [], []
            
        if image["id"] == annotations[index]["image_id"] and os.path.isfile(file_path):
            try:
                image_tensor = get_image_tensor(file_path)
                batch_data.append(image_tensor)
                batch_labels.append(label)
                
                if label == 1:
                    num_wildlife_present_images += 1
            except:
                print("Truncated image encountered, leaving out of training and testing")
    
    data.append(batch_data)
    labels.append(batch_labels)
    print("num_wildlife_present_images: ", num_wildlife_present_images)
    
    return data, labels

def print_image(image_tensor, prediction, saving_dir, index):
    if(prediction == 1):
        prediction_string = "Wildlife_Present"
    else:
        prediction_string = "No_Wildlife_Present"

    image_file_name = saving_dir + prediction_string + "_" + str(index) + ".png"
    
    #Alternative normalized RGB visualization: plt.imshow(image_tensor.cpu().permute(1, 2, 0).numpy())
    plt.imshow(image_tensor[0].cpu(), cmap="gray")
    plt.title("Incorrectly Predicted " + prediction_string) 
    plt.show()
    plt.imsave(image_file_name, image_tensor[0].cpu(), cmap="gray")

def print_testing_analysis(all_labels, all_predictions, title, saving_dir):
    subplot = plt.subplot()

    cf_matrix = confusion_matrix(all_labels, all_predictions, labels=[1, 0])
    sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues')

    subplot.set_xlabel('Predictions')
    subplot.set_ylabel('Labels')
    subplot.set_title(title + ' Testing Confusion Matrix')
    subplot.xaxis.set_ticklabels(['Wildlife Present', 'No Wildlife Present'])
    subplot.yaxis.set_ticklabels(['Wildlife Present', 'No Wildlife Present'])
    
    plot_file_name = saving_dir + title + "_Confusion_Matrix.png"
    plt.savefig(plot_file_name)
    plt.show()

    accuracy = accuracy_score(all_labels, all_predictions)
    print(title + " Accuracy: " + str(accuracy))

    precision, recall, f_score, support = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    print(title + " Precision: " + str(precision))
    print(title + " Recall: " + str(recall))
    print(title + " F-Score: " + str(f_score))
    
def test(model, loader, criterion, print_incorrect_images, device):
    model.eval()
    num_correct = 0
    all_labels, all_predictions = [], []

    for batch in loader:
        data, labels = batch['data'][0], batch['label'][0]

        # This is to prevent cuda memory issues for large batches
        batch_prediction = 0
        for image in data:
            image = torch.unsqueeze(image, dim=0).to(device)
            output = model(image.to(device))
            print("output: ", output)
            _, prediction = torch.max(output.data, 1)
            print("prediction: ", prediction)
            if prediction == 1:
                batch_prediction = 1
                
        print("batch_prediction: ", batch_prediction)
        batch_label = 0
        for label in labels:
            print("label: ", label)
            if label == 1:
                batch_label = 1
        
        print("batch_label: ", batch_label)
        if batch_prediction == batch_label:
            num_correct += 1

        all_predictions.append(batch_prediction)
        all_labels.append(batch_label)

    accuracy = num_correct/len(loader.dataset)
    return accuracy, all_labels, all_predictions
    
def flatten_list(data):
    return [image for batch in data for image in batch]

# Declaring Constants
num_epochs = 5
num_classes = 2
batch_size = 10
json_file_name = "idaho-camera-traps.json"
zip_prefix = "idaho-camera-traps-images.part_"
downloaded_data_dir = "/nfs/hpc/share/isonc/downloaded_data/"
saving_dir = "/nfs/stak/users/isonc/hpc-share/oregon_wildlife_identification/model_benchmarking/run_logs"
resnet50_weights_path = "/nfs/stak/users/isonc/hpc-share/saved_models/batch_classification_ResNet50/ResNet50.pt"
resnet152_weights_path = "/nfs/stak/users/isonc/hpc-share/saved_models/batch_classification_ResNet152/ResNet152.pt"
blob_name = "https://lilablobssc.blob.core.windows.net/idaho-camera-traps/"
unzipped_data_dir = downloaded_data_dir + "public/"

# Mapping canines, big cats, bears, and ungulates to wildlife present and all other categories to no wildlife present
# This is mostly arbitrary and could be reworked, we just need to draw the line somewhere
categories_to_label_dict = {
    0:0, 1:0, 2:0, 3:1, 4:0, 5:1, 6:1, 7:0, 8:0, 9:1, 
    10:1, 11:0, 12:1, 13:1, 14:0, 15:0, 16:1, 17:0, 18:1, 19:0,
    20:1, 21:0, 22:1, 23:0, 24:1, 25:0, 26:0, 27:0, 28:0, 29:0,
    30:0, 31:0, 32:0, 33:0, 34:0, 35:0, 36:0, 37:0, 38:1, 39:1,
    40:1, 41:0, 42:0, 43:0, 44:0, 45:1, 46:0, 47:0, 48:1, 49:0,
    50:0, 51:0, 52:0, 53:0, 54:0, 55:0, 56:0, 57:0, 58:0, 59:0,
    60:0, 61:0,
}

print(torch.__version__)
print(torchvision.__version__)
print("torch.cuda.is_available(): ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
criterion = nn.CrossEntropyLoss()

# Declaring Models
resnet50 = models.resnet50()
in_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(in_features, num_classes)
resnet50.load_state_dict(torch.load(resnet50_weights_path))
resnet50.to(device)

resnet152 = models.resnet152()
in_features = resnet152.fc.in_features
resnet152.fc = nn.Linear(in_features, num_classes)
resnet152.load_state_dict(torch.load(resnet152_weights_path))
resnet152.to(device)

if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: ", torch.cuda.device_count())
    resnet50 = nn.DataParallel(resnet50)
    resnet152 = nn.DataParallel(resnet152)

# Orchestration
download_zip(downloaded_data_dir, json_file_name, blob_name)
json_file = open(downloaded_data_dir + json_file_name)
coco_key = json.load(json_file)
images_json = coco_key["images"]
annotations = coco_key["annotations"]
print("len(images_json): ", len(images_json))

all_resnet50_predictions, all_resnet50_labels = [], []
all_resnet152_predictions, all_resnet152_labels = [], []
step_size = 1000

highest_resnet50_testing_accuracy = 0
highest_resnet152_testing_accuracy = 0
    
#TODO
for i in range(1):
    zip_name = zip_prefix + str(i)
        
    #TODO  
    #download_zip(downloaded_data_dir, zip_name, blob_name)
        
    # Inner loop needed to avoid memory issues
    for index in range(0, len(images_json), step_size):
        print("\nindex: ", index)
        data, labels = get_data_sets(images_json[index:index+step_size], annotations[index:index+step_size], unzipped_data_dir, categories_to_label_dict)
            
        if len(data) == 0:
            continue
            
        data_set = image_data_set(data, labels)
        loader = DataLoader(dataset = data_set, batch_size = batch_size, shuffle = True)
        
        testing_accuracy, labels, predictions = test(resnet50, loader, criterion, False, device)
        print("ResNet50 testing accuracy: ", testing_accuracy)
        all_resnet50_labels.extend(labels)
        all_resnet50_predictions.extend(predictions)
        
        testing_accuracy, labels, predictions = test(resnet152, loader, criterion, False, device)
        print("ResNet152 testing accuracy: ", testing_accuracy)
        all_resnet152_labels.extend(labels)
        all_resnet152_predictions.extend(predictions)
    
    #TODO   
    #shutil.rmtree(unzipped_data_dir)  
    
print_testing_analysis(all_resnet50_labels, all_resnet50_predictions, "Resnet50_Overall", saving_dir)   
print_testing_analysis(all_resnet152_labels, all_resnet152_predictions, "Resnet152_Overall", saving_dir)

json_file.close()
os.remove(downloaded_data_dir + json_file_name)


