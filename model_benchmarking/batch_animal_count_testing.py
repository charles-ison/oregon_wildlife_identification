import os
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import utilities
from PIL import Image
from torch.utils.data import DataLoader
from pytorch_grad_cam import FullGrad
from custom_models.aggregating_cnn import AggregatingCNN
from custom_data_sets.image_data_set import ImageDataSet
from custom_models.cnn_wrapper import CNNWrapper


def test_batch(model, model_name, batch_testing_loader, criterion, print_incorrect_images, saving_dir, device):
    model.eval()
    num_correct = 0
    running_loss = 0.0
    all_labels, all_predictions = [], []

    for batch in batch_testing_loader:
        data, labels = torch.squeeze(batch['data'], dim=0).to(device), batch['label'].to(device)

        # This is to prevent cuda memory issues for large batches
        max_prediction = 0
        for image in data:
            image = torch.unsqueeze(image, dim=0)
            output = model(image).flatten()
            max_prediction = max(max_prediction, output.round().item())

        max_prediction = torch.tensor(max_prediction).to(device)
        max_label = torch.max(labels)
        
        loss = criterion(max_prediction, max_label)
        running_loss += loss.item()

        if max_prediction == max_label:
            num_correct += 1
            
        all_labels.append(max_label.item())
        all_predictions.append(max_prediction.item())

    loss = running_loss/len(batch_testing_loader.dataset)
    accuracy = num_correct/len(batch_testing_loader.dataset)
    utilities.print_analysis(all_labels, all_predictions, model_name + "_Testing", saving_dir)
    return loss, accuracy
    
    
def test_individual(model, grad_cam, testing_loader, criterion, print_incorrect_images, print_heat_map, saving_dir, device):
    model.eval()
    running_loss = 0.0
    num_correct = 0
    grad_cam_identifier = 0

    for i, batch in enumerate(testing_loader):
        data, labels = batch['data'].to(device), batch['label'].to(device)
        output = model(data).flatten()

        loss = criterion(output, labels)
        running_loss += loss.item()
        for index, prediction in enumerate(output.round()):
            prediction = prediction.cpu().item()
            if prediction == labels[index]:
                num_correct += 1
            elif print_incorrect_images:
                utilities.print_image(data[index], prediction, saving_dir, i)
            
            # Just looking at every 5 samples
            if print_heat_map and grad_cam_identifier % 10 == 0 and grad_cam != None:
                utilities.create_heat_map(grad_cam, data[index], prediction, labels[index], saving_dir, grad_cam_identifier)
            grad_cam_identifier += 1

    loss = running_loss/len(testing_loader.dataset)
    accuracy = num_correct/len(testing_loader.dataset)
    return loss, accuracy


def test(model, model_name, grad_cam, batch_loader, individual_loader, device, criterion, saving_dir):
    model.to(device)
    
    individual_loss, individual_accuracy = test_individual(model, grad_cam, individual_loader, criterion, False, False, saving_dir, device)
    print("individual testing loss (MSE): " + str(individual_loss) + " and individual testing accuracy: "+ str(individual_accuracy))

    batch_loss, batch_accuracy = test_batch(model, model_name, batch_loader, criterion, False, saving_dir, device)
    print("batch testing loss (MSE): " + str(batch_loss) + " and batch testing accuracy: "+ str(batch_accuracy))
    
    
def get_predictions(bounding_boxes):
    num_correct = 0
    labels, predictions = [], []
    for box_index, boxes in enumerate(bounding_boxes):
        num_animals = 0
        for score_index, score in enumerate(boxes["scores"]):
            if score > 0.5 and boxes["labels"][score_index] == 1:
                num_animals += 1
        predictions.append(num_animals)
    return predictions


def test_individual_object_detection(model, individual_data_set, batch_size, criterion, saving_dir, device):
    model.eval()
    running_loss = 0.0
    num_correct = 0

    for index in range(0, len(individual_data_set), batch_size):
        batch = individual_data_set[index:index + batch_size]
        data, labels = batch['data'], batch['label']
        utilities.set_device_for_list_of_tensors(data, device)
        
        bounding_boxes = model(data)
        predictions = get_predictions(bounding_boxes)
        labels_tensor = torch.FloatTensor(labels)
        predictions_tensor = torch.FloatTensor(predictions)
        running_loss += criterion(labels_tensor, predictions_tensor).item()
        num_correct += (labels_tensor == predictions_tensor).sum().item()

    loss = running_loss/len(individual_data_set)
    accuracy = num_correct/len(individual_data_set)
    return loss, accuracy
    

def test_batch_object_detection(model, model_name, batch_data_set, criterion, saving_dir, device):
    model.eval()
    num_correct = 0
    running_loss = 0.0
    all_labels, all_predictions = [], []

    for batch in batch_data_set:
        data, labels = batch['data'], batch['label']

        # This is to prevent cuda memory issues for large batches
        max_prediction = 0
        max_label = 0
        for index, image in enumerate(data):
            image = torch.unsqueeze(image, dim=0).to(device)
            bounding_boxes = model(image)
            prediction = get_predictions(bounding_boxes)[0]
            label = labels[index]
            
            max_prediction = max(max_prediction, prediction)
            max_label = max(max_label, label)
            
        running_loss += criterion(torch.FloatTensor([max_label]), torch.FloatTensor([max_prediction])).item()
        if max_prediction == max_label:
            num_correct += 1
            
        all_labels.append(max_label)
        all_predictions.append(max_predictions)

    loss = running_loss/len(batch_data_set)
    accuracy = num_correct/len(batch_data_set)
    utilities.print_analysis(all_labels, all_predictions, model_name + "_Testing", saving_dir)
    return loss, accuracy
    
    
def test_aggregating_cnn(model, model_name, batch_loader, device, criterion, saving_dir):
    model.to(device)
    model.eval()
    running_loss = 0.0
    num_correct = 0
    all_labels, all_predictions = [], []
    
    for batch in batch_loader:
        data, labels = torch.squeeze(batch['data'], dim=0).to(device), batch['label'].to(device)
        label = torch.max(labels)
        
        output = model(data)

        loss = criterion(output, label)
        running_loss += loss.item()
        num_correct += (output == label).item()
        
        all_labels.append(label.item())
        all_predictions.append(output.item())

    loss = running_loss/len(batch_loader.dataset)
    accuracy = num_correct/len(batch_loader.dataset)
    utilities.print_analysis(all_labels, all_predictions, model_name + "_Testing", saving_dir)
    print("batch testing loss (MSE): " + str(loss) + " and batch testing accuracy: "+ str(accuracy))

    
def test_object_detection(model, model_name, batch_data_set, individual_data_set, batch_size, device, criterion, saving_dir):
    model.to(device)
    
    individual_loss, individual_accuracy = test_individual_object_detection(model, individual_data_set, batch_size, criterion, saving_dir, device)
    print("individual testing loss (MSE): " + str(individual_loss) + " and individual testing accuracy: "+ str(individual_accuracy))

    batch_loss, batch_accuracy = test_batch_object_detection(model, model_name, batch_data_set, criterion, saving_dir, device)
    print("batch testing loss (MSE): " + str(batch_loss) + " and batch testing accuracy: "+ str(batch_accuracy))

    
def get_data(batch_size, data_dir, json_file_name):
    batch_testing_data, batch_testing_labels, individual_data, individual_labels = utilities.fetch_data(data_dir, json_file_name, False, False, False)
    batch_data_set = ImageDataSet(batch_testing_data, batch_testing_labels)
    individual_data_set = ImageDataSet(individual_data, individual_labels)
    batch_data_loader = DataLoader(dataset = batch_data_set, batch_size = 1, shuffle = True)
    individual_data_loader = DataLoader(dataset = individual_data_set, batch_size = batch_size, shuffle = True)
    return batch_data_loader, individual_data_loader, batch_data_set, individual_data_set


# Declaring Constants
batch_size = 5
cottonwood_eastface_json_file_name = "2023_Cottonwood_Eastface_5.30_7.10_key.json"
cottonwood_westface_json_file_name = "2023_Cottonwood_Westface_5.30_7.10_102RECNX_key.json"
ngilchrist_eastface_json_file_name = "2022_NGilchrist_Eastface_055_07.12_07.20_key.json"
idaho_json_file_name = "Idaho_loc_0099_key.json"
data_dir = "/nfs/stak/users/isonc/hpc-share/saved_data/testing_animal_count/"

resnet_34_saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/oregon_wildlife_identification/batch_count_ResNet34/"
resnet_50_saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/oregon_wildlife_identification/batch_count_ResNet50/"
resnet_152_saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/oregon_wildlife_identification/batch_count_ResNet152/"
faster_rcnn_saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/oregon_wildlife_identification/batch_count_FasterR-CNN/"
ssd_saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/oregon_wildlife_identification/batch_count_SSD/"
retina_net_saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/oregon_wildlife_identification/batch_count_RetinaNet/"
aggregating_cnn_saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/oregon_wildlife_identification/batch_count_AggregatingCNN/"
vit_l_16_saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/oregon_wildlife_identification/batch_count_ViTL16/"
cnn_wrapper_saving_dir = "/nfs/stak/users/isonc/hpc-share/saved_models/oregon_wildlife_identification/batch_count_CNNWrapper/" 

resnet34_weights_path = resnet_34_saving_dir + "ResNet34.pt"
resnet50_weights_path = resnet_50_saving_dir + "ResNet50.pt"
resnet152_weights_path = resnet_152_saving_dir + "ResNet152.pt"
faster_rcnn_weights_path = faster_rcnn_saving_dir + "FasterR-CNN.pt"
ssd_weights_path = ssd_saving_dir + "SSD.pt"
retina_net_weights_path = retina_net_saving_dir + "RetinaNet.pt"
aggregating_cnn_weights_path = aggregating_cnn_saving_dir + "AggregatingCNN.pt"
vit_l_16_weights_path = vit_l_16_saving_dir + "ViTL16.pt"
cnn_wrapper_weights_path = aggregating_cnn_saving_dir + "CNNWrapper.pt"

resnet_50_grad_cam_path = resnet_50_saving_dir + "grad_cam/"
resnet_152_grad_cam_path = resnet_152_saving_dir + "grad_cam/"

print(torch.__version__)
print(torchvision.__version__)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
criterion = nn.MSELoss()

print("\nGetting Cottonwood Eastface data")
cottonwood_ef_batch_loader, cottonwood_ef_individual_loader, cottonwood_ef_batch_data_set, cottonwood_ef_individual_data_set = get_data(batch_size, data_dir, cottonwood_eastface_json_file_name)

print("\nGetting Cottonwood Westface data")
cottonwood_wf_batch_loader, cottonwood_wf_individual_loader, cottonwood_wf_batch_data_set, cottonwood_wf_individual_data_set = get_data(batch_size, data_dir, cottonwood_westface_json_file_name)

print("\nGetting NGilchrist Eastface data")
ngilchrist_ef_batch_loader, ngilchrist_ef_individual_loader, ngilchrist_ef_batch_data_set, ngilchrist_ef_individual_data_set = get_data(batch_size, data_dir, ngilchrist_eastface_json_file_name)

print("\nGetting Idaho data")
idaho_batch_loader, idaho_individual_loader, idaho_batch_data_set, idaho_individual_data_set = get_data(batch_size, data_dir, idaho_json_file_name)

# Declaring Models
# Have to follow same steps used to create model during training
resnet34 = models.resnet34()
in_features = resnet34.fc.in_features
resnet34.fc = nn.Linear(in_features, 1)

resnet50 = models.resnet50()
in_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(in_features, 1)

resnet152 = models.resnet152()
in_features = resnet152.fc.in_features
resnet152.fc = nn.Linear(in_features, 1)

vit_l_16 = models.vit_l_16()
in_features = vit_l_16.heads[0].in_features
vit_l_16.heads[0] = nn.Linear(in_features, 1)

faster_rcnn = models.detection.fasterrcnn_resnet50_fpn_v2()
ssd = models.detection.ssd300_vgg16()
retina_net = models.detection.retinanet_resnet50_fpn_v2()

max_batch_size = 100
embedding_size = 512
cnn = nn.DataParallel(models.resnet34())
aggregating_cnn = AggregatingCNN(max_batch_size, embedding_size, cnn)

cnn = models.resnet34()
cnn_wrapper = CNNWrapper(cnn)

#Layer 4 is just recommended by GradCam documentation for ResNet
resnet34_cam = FullGrad(model=resnet34, target_layers=[], use_cuda=torch.cuda.is_available())
resnet50_cam = FullGrad(model=resnet50, target_layers=[], use_cuda=torch.cuda.is_available())
resnet152_cam = FullGrad(model=resnet152, target_layers=[], use_cuda=torch.cuda.is_available())

#Loading trained model weights
resnet34.load_state_dict(torch.load(resnet34_weights_path))
resnet50.load_state_dict(torch.load(resnet50_weights_path))
resnet152.load_state_dict(torch.load(resnet152_weights_path))
faster_rcnn.load_state_dict(torch.load(faster_rcnn_weights_path))
ssd.load_state_dict(torch.load(ssd_weights_path))
retina_net.load_state_dict(torch.load(retina_net_weights_path))
aggregating_cnn.load_state_dict(torch.load(aggregating_cnn_weights_path))
vit_l_16.load_state_dict(torch.load(vit_l_16_weights_path))

# Object Detection models cannot be used with data parallel
if torch.cuda.device_count() > 1:
    print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    resnet34 = nn.DataParallel(resnet34)
    resnet50 = nn.DataParallel(resnet50)
    resnet152 = nn.DataParallel(resnet152)
    cnn_wrapper = nn.DataParallel(cnn_wrapper)

# Testing
model_name = "ResNet34"
print("\nTesting ResNet34 on Cottonwood Eastface")
test(resnet34, model_name + "_Cottonwood_EF", resnet34_cam, cottonwood_ef_batch_loader, cottonwood_ef_individual_loader, device, criterion, resnet_34_saving_dir)
print("\nTesting ResNet34 on Cottonwood Westface")
test(resnet34, model_name + "_Cottonwood_WF", resnet34_cam, cottonwood_wf_batch_loader, cottonwood_wf_individual_loader, device, criterion, resnet_34_saving_dir)
print("\nTesting ResNet34 on NGilchrist Eastface")
test(resnet34, model_name + "_NGilchrist_EF", resnet34_cam, ngilchrist_ef_batch_loader, ngilchrist_ef_individual_loader, device, criterion, resnet_34_saving_dir)
print("\nTesting ResNet34 on Idaho")
test(resnet34, model_name + "_Idaho", resnet34_cam, idaho_batch_loader, idaho_individual_loader, device, criterion, resnet_34_saving_dir)

model_name = "ResNet50"
print("\nTesting ResNet50 on Cottonwood Eastface")
test(resnet50, model_name + "_Cottonwood_EF", resnet50_cam, cottonwood_ef_batch_loader, cottonwood_ef_individual_loader, device, criterion, resnet_50_saving_dir)
print("\nTesting ResNet50 on Cottonwood Westface")
test(resnet50, model_name + "_Cottonwood_WF", resnet50_cam, cottonwood_wf_batch_loader, cottonwood_wf_individual_loader, device, criterion, resnet_50_saving_dir)
print("\nTesting ResNet50 on NGilchrist Eastface")
test(resnet50, model_name + "_NGilchrist_EF", resnet50_cam, ngilchrist_ef_batch_loader, ngilchrist_ef_individual_loader, device, criterion, resnet_50_saving_dir)
print("\nTesting ResNet50 on Idaho")
test(resnet50, model_name + "_Idaho", resnet50_cam, idaho_batch_loader, idaho_individual_loader, device, criterion, resnet_50_saving_dir)

model_name = "ResNet152"
print("\nTesting ResNet152 on Cottonwood Eastface")
test(resnet152, model_name + "_Cottonwood_EF", resnet152_cam, cottonwood_ef_batch_loader, cottonwood_ef_individual_loader, device, criterion, resnet_152_saving_dir)
print("\nTesting ResNet152 on Cottonwood Westface")
test(resnet152, model_name + "_Cottonwood_WF", resnet152_cam, cottonwood_wf_batch_loader, cottonwood_wf_individual_loader, device, criterion, resnet_152_saving_dir)
print("\nTesting ResNet152 on NGilchrist Eastface")
test(resnet152, model_name + "_NGilchrist_EF", resnet152_cam, ngilchrist_ef_batch_loader, ngilchrist_ef_individual_loader, device, criterion, resnet_152_saving_dir)
print("\nTesting ResNet152 on Idaho")
test(resnet152, model_name + "_Idaho", resnet152_cam, idaho_batch_loader, idaho_individual_loader, device, criterion, resnet_152_saving_dir)

model_name = "CNNWrapper"
print("\nTesting CNNWrapper on Cottonwood Eastface")
test(cnn_wrapper, model_name + "_Cottonwood_EF", None, cottonwood_ef_batch_loader, cottonwood_ef_individual_loader, device, criterion, cnn_wrapper_saving_dir)
print("\nTesting CNNWrapper on Cottonwood Westface")
test(cnn_wrapper, model_name + "_Cottonwood_WF", None, cottonwood_wf_batch_loader, cottonwood_wf_individual_loader, device, criterion, cnn_wrapper_saving_dir)
print("\nTesting CNNWrapper on NGilchrist Eastface")
test(cnn_wrapper, model_name + "_NGilchrist_EF", None, ngilchrist_ef_batch_loader, ngilchrist_ef_individual_loader, device, criterion, cnn_wrapper_saving_dir)
print("\nTesting CNNWrapper on Idaho")
test(cnn_wrapper, model_name + "_Idaho", None, idaho_batch_loader, idaho_individual_loader, device, criterion, cnn_wrapper_saving_dir)

model_name = "AggregatingCNN"
print("\nTesting Aggregating CNN on Cottonwood Eastface")
test_aggregating_cnn(aggregating_cnn, model_name + "_Cottonwood_EF", cottonwood_ef_batch_loader, device, criterion, aggregating_cnn_saving_dir)
print("\nTesting Aggregating CNN on Cottonwood Westface")
test_aggregating_cnn(aggregating_cnn, model_name + "_Cottonwood_WF", cottonwood_wf_batch_loader, device, criterion, aggregating_cnn_saving_dir)
print("\nTesting Aggregating CNN on NGilchrist Eastface")
test_aggregating_cnn(aggregating_cnn, model_name + "_NGilchrist_EF", ngilchrist_ef_batch_loader, device, criterion, aggregating_cnn_saving_dir)
print("\nTesting Aggregating CNN on Idaho")
test_aggregating_cnn(aggregating_cnn, model_name + "_Idaho", idaho_batch_loader, device, criterion, aggregating_cnn_saving_dir)

model_name = "ViTL16"
print("\nTesting Vision Transformer Large 16 on Cottonwood Eastface")
test(vit_l_16, model_name + "_Cottonwood_EF", None, cottonwood_ef_batch_loader, cottonwood_ef_individual_loader, device, criterion, vit_l_16_saving_dir)
print("\nTesting Vision Transformer Large 16 on Cottonwood Westface")
test(vit_l_16, model_name + "_Cottonwood_WF", None, cottonwood_wf_batch_loader, cottonwood_wf_individual_loader, device, criterion, vit_l_16_saving_dir)
print("\nTesting Vision Transformer Large 16 on NGilchrist Eastface")
test(vit_l_16, model_name + "_NGilchrist_EF", None, ngilchrist_ef_batch_loader, ngilchrist_ef_individual_loader, device, criterion, vit_l_16_saving_dir)
print("\nTesting Vision Transformer Large 16 on Idaho")
test(vit_l_16, model_name + "_Idaho", None, idaho_batch_loader, idaho_individual_loader, device, criterion, vit_l_16_saving_dir)

model_name = "FasterR-CNN"
print("\nTesting Faster R-CNN on Cottonwood Eastface")
test_object_detection(faster_rcnn, model_name + "_Cottonwood_EF", cottonwood_ef_batch_data_set, cottonwood_ef_individual_data_set, batch_size, device, criterion, faster_rcnn_saving_dir)
print("\nTesting Faster R-CNN on Cottonwood Westface")
test_object_detection(faster_rcnn,  model_name + "_Cottonwood_WF", cottonwood_wf_batch_data_set, cottonwood_wf_individual_data_set, batch_size, device, criterion, faster_rcnn_saving_dir)
print("\nTesting Faster R-CNN on NGilchrist Eastface")
test_object_detection(faster_rcnn, model_name + "_NGilchrist_EF", ngilchrist_ef_batch_data_set, ngilchrist_ef_individual_data_set, batch_size, device, criterion, faster_rcnn_saving_dir)
print("\nTesting Faster R-CNN on Idaho")
test_object_detection(faster_rcnn,  model_name + "_Idaho", idaho_batch_data_set, idaho_individual_data_set, batch_size, device, criterion, faster_rcnn_saving_dir)

model_name = "SSD"
print("\nTesting SSD on Cottonwood Eastface")
test_object_detection(ssd, model_name + "_Cottonwood_EF", cottonwood_ef_batch_data_set, cottonwood_ef_individual_data_set, batch_size, device, criterion, ssd_saving_dir)
print("\nTesting SSD on Cottonwood Westface")
test_object_detection(ssd, model_name + "_Cottonwood_WF", cottonwood_wf_batch_data_set, cottonwood_wf_individual_data_set, batch_size, device, criterion, ssd_saving_dir)
print("\nTesting SSD on NGilchrist Eastface")
test_object_detection(ssd, model_name + "_NGilchrist_EF", ngilchrist_ef_batch_data_set, ngilchrist_ef_individual_data_set, batch_size, device, criterion, ssd_saving_dir)
print("\nTesting SSD on Idaho")
test_object_detection(ssd, model_name + "_Idaho", idaho_batch_data_set, idaho_individual_data_set, batch_size, device, criterion, ssd_saving_dir)

model_name = "RetinaNet"
print("\nTesting RetinaNet on Cottonwood Eastface")
test_object_detection(retina_net, model_name + "_Cottonwood_EF", cottonwood_ef_batch_data_set, cottonwood_ef_individual_data_set, batch_size, device, criterion, retina_net_saving_dir)
print("\nTesting RetinaNet on Cottonwood Westface")
test_object_detection(retina_net, model_name + "_Cottonwood_WF", cottonwood_wf_batch_data_set, cottonwood_wf_individual_data_set, batch_size, device, criterion, retina_net_saving_dir)
print("\nTesting RetinaNet on NGilchrist Eastface")
test_object_detection(retina_net, model_name + "_NGilchrist_EF", ngilchrist_ef_batch_data_set, ngilchrist_ef_individual_data_set, batch_size, device, criterion, retina_net_saving_dir)
print("\nTesting RetinaNet on Idaho")
test_object_detection(retina_net, model_name + "_Idaho", idaho_batch_data_set, idaho_individual_data_set, batch_size, device, criterion, retina_net_saving_dir)


