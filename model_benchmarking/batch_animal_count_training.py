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
from custom_data_sets.image_data_set import ImageDataSet


#new
import itertools
import random
from numpy.random import randint
from numpy.random import rand
import pickle
from copy import deepcopy
import timeit
    
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
    

def is_directory_supplemental(directory):
    if directory.name == "caltech":
        return True
    else:
        return False
    

def fetch_training_data(data_dir):
    json_file_name = "animal_count_key.json"
    training_data, training_labels, validation_data, validation_labels, batch_training_data, batch_training_labels, batch_validation_data, batch_validation_labels = [], [], [], [], [], [], [], []
    
    for directory in os.scandir(data_dir):
        if directory.is_dir():
            directory_path = directory.path + "/"
            print("\nFeting data from directory: ", directory_path)
            is_supplmental = is_directory_supplemental(directory)
            temp_train_data, temp_train_lab, temp_val_data, temp_valid_lab, temp_batch_train_data, temp_batch_train_lab, temp_batch_val_data, temp_batch_val_lab = utilities.fetch_data(directory_path, json_file_name, True, is_supplmental)
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
    #utilities.plot_histogram(training_labels, data_dir)
    
    return training_data, training_labels, validation_data, validation_labels, batch_training_data, batch_training_labels, batch_validation_data, batch_validation_labels
    
    

def train(model, training_data_set, criterion, optimizer, device, batch_size):
    model.train()
    running_loss = 0.0
    
    for index in range(0, len(training_data_set), batch_size):
        batch = training_data_set[index:index + batch_size]
        data, targets = utilities.get_info_from_batch(batch, device)
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
        data, targets = utilities.get_info_from_batch(batch, device)
        
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
    num_correct = 0

    for index in range(0, len(validation_data_set), batch_size):
        batch = validation_data_set[index:index + batch_size]
        data, targets = utilities.get_info_from_batch(batch, device)
        
        model.eval()
        bounding_boxes = model(data)
        
        predictions = get_predictions(bounding_boxes)
        labels = utilities.get_labels_from_targets(targets)
        
        running_mse += mse_criterion(torch.FloatTensor(predictions), torch.FloatTensor(labels)).item()
        running_mae += mae_criterion(torch.FloatTensor(predictions), torch.FloatTensor(labels)).item()
        num_correct += get_num_equal_list_elements(labels, predictions)

    mse = running_mse/len(validation_data_set)
    mae = running_mae/len(validation_data_set)
    accuracy = num_correct/len(validation_data_set)
    return mse, mae, accuracy


def validation(model, validation_data_set, mse_criterion, mae_criterion, device, batch_size):
    model.eval()
    running_mse = 0.0
    running_mae = 0.0
    num_correct = 0

    for index in range(0, len(validation_data_set), batch_size):
        batch = validation_data_set[index:index + batch_size]
        data, targets = utilities.get_info_from_batch(batch, device)
        data = torch.stack(data)
        labels = utilities.get_labels_from_targets(targets)
        labels = torch.FloatTensor(labels).to(device)
        
        output = model(data).flatten()

        running_mse += mse_criterion(output, labels).item()
        running_mae += mae_criterion(output, labels).item()
        num_correct += (output.round() == labels).sum().item()

    mse = running_mse/len(validation_data_set)
    mae = running_mae/len(validation_data_set)
    accuracy = num_correct/len(validation_data_set)
    return mse, mae, accuracy


def batch_validation(model, batch_validation_data_set, mse_criterion, mae_criterion, device):
    model.eval()
    running_mse = 0.0
    running_mae = 0.0
    num_correct = 0
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
        if max_prediction.round().item() == max_label:
            num_correct += 1

        all_predictions.append(max_prediction.item())
        all_labels.append(max_label.item())

    mse = running_mse/len(batch_validation_data_set)
    mae = running_mae/len(batch_validation_data_set)
    accuracy = num_correct/len(batch_validation_data_set)
    return mse, mae, accuracy, all_labels, all_predictions
    
    
def batch_validation_object_detection(model, batch_validation_data_set, mse_criterion, mae_criterion, print_incorrect_images, saving_dir, device):
    model.eval()
    running_mse = 0.0
    running_mae = 0.0
    num_correct = 0
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
        if max_prediction == max_label:
            num_correct += 1

        all_predictions.append(max_prediction)
        all_labels.append(max_label)

    mse = running_mse/len(batch_validation_data_set)
    mae = running_mae/len(batch_validation_data_set)
    accuracy = num_correct/len(batch_validation_data_set)
    return mse, mae, accuracy, all_labels, all_predictions

def train_and_validate(train_size, num_epochs, model, model_name, optim, optim_lr, training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, batch_testing_data_set, individual_testing_data_set, device, saving_dir, batch_size, is_object_detection, lowest_batch_mae):
    model.to(device)
    
    optimizer = optim(model.parameters(), lr=optim_lr)
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    huber_loss = nn.HuberLoss()
    #lowest_batch_val_mae = None
    saving_dir = saving_dir + "batch_count_" + model_name + "/"
    #print("optimizer: " + str(optim) + ", lr: " + str(optim_lr))
    
    for epoch in range(num_epochs):
        training_data_set.shuffle()
        validation_data_set.shuffle()
        batch_training_data_set.shuffle()


        #TODO: Use OOP here
        if is_object_detection:
            training_loss = train_object_detection(model, training_data_set, optimizer, device, batch_size)
        else:
            training_loss = train(model, training_data_set, huber_loss, optimizer, device, batch_size)
        #print("training loss: " + str(training_loss))

        if epoch == 0 or (epoch+1) % 5 == 0:
            if is_object_detection:
                val_mse, val_mae, val_acc = validation_object_detection(model, validation_data_set, mse, mae, device, batch_size)
            else:
                val_mse, val_mae, val_acc = validation(model, validation_data_set, mse, mae, device, batch_size)
            #print("validation MSE: " + str(val_mse) + ", MAE: " + str(val_mae) + " and ACC: " + str(val_acc))
            
            if is_object_detection:
                batch_val_mse, batch_val_mae, batch_val_acc, batch_labels, batch_predictions = batch_validation_object_detection(model, batch_validation_data_set, mse, mae, False, saving_dir, device)
            else:
                batch_val_mse, batch_val_mae, batch_val_acc, batch_labels, batch_predictions = batch_validation(model, batch_validation_data_set, mse, mae, device)
            
            #print("batch validation MSE: " + str(batch_val_mse) + ", MAE: " + str(batch_val_mae) + " and ACC: " + str(batch_val_acc))
        
            r_squared = utilities.regression_analysis(batch_labels, batch_predictions)
            if is_object_detection:
                ind_test_mse, ind_test_mae, ind_test_acc, batch_test_mse, batch_test_mae, batch_test_acc = test_object_detection(model, model_name, batch_testing_data_set, individual_testing_data_set, 5, device, mse, mae)
            else:
                ind_test_mse, ind_test_mae, ind_test_acc, batch_test_mse, batch_test_mae, batch_test_acc = test(model, model_name, None, batch_testing_data_set, individual_testing_data_set, 5, device, mse, mae)

            #if lowest_batch_mae is None or batch_test_mae < lowest_batch_mae:
            #    lowest_batch_mae = batch_test_mae
            print(str(train_size)+","+str(epoch+1)+","+str(optim)+","+str(optim_lr)+","+str(batch_size)+","+str(training_loss)+","+str(val_mse)+","+str(val_mae)+","+str(val_acc)+","+str(batch_val_mse)+","+str(batch_val_mae)+","+str(batch_val_acc)+","+str(r_squared)+","+str(ind_test_mse)+','+str(ind_test_mae)+','+str(ind_test_acc)+','+str(batch_test_mse)+','+str(batch_test_mae)+','+str(batch_test_acc))
            
    return batch_test_mae
        
# Testing Functions
def test_batch(model, model_name, data_set, mse_criterion, mae_criterion, device):
    model.eval()
    running_mse = 0.0
    running_mae = 0.0
    num_correct = 0
    all_labels, all_predictions = [], []

    for batch in data_set:
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
        if max_prediction.round().item() == max_label:
            num_correct += 1
            
        all_labels.append(max_label.item())
        all_predictions.append(max_prediction.item())

    mse = running_mse/len(data_set)
    mae = running_mae/len(data_set)
    acc = num_correct/len(data_set)
    return mse, mae, acc
    
def test_individual(model, grad_cam, data_set, mse_criterion, mae_criterion, batch_size, device):
    model.eval()
    running_mse = 0.0
    running_mae = 0.0
    num_correct = 0
    grad_cam_identifier = 0

    for index in range(0, len(data_set), batch_size):
        batch = data_set[index:(index + batch_size)]
        data, targets = utilities.get_info_from_batch(batch, device)
        data = torch.stack(data)
        labels = utilities.get_labels_from_targets(targets)
        labels = torch.FloatTensor(labels).to(device)
        
        output = model(data).flatten()

        running_mse += mse_criterion(output, labels).item()
        running_mae += mae_criterion(output, labels).item()
        num_correct += (output.round() == labels).sum().item()

    mse = running_mse/len(data_set)
    mae = running_mae/len(data_set)
    acc = num_correct/len(data_set)
    return mse, mae, acc

def test(model, model_name, grad_cam, batch_data_set, individual_data_set, batch_size, device, mse_criterion, mae_criterion):
    model.to(device)
    ind_mse, ind_mae, ind_acc = test_individual(model, grad_cam, individual_data_set, mse_criterion, mae_criterion, batch_size, device)
    batch_mse, batch_mae, batch_acc = test_batch(model, model_name, batch_data_set, mse_criterion, mae_criterion, device)
    return ind_mse, ind_mae, ind_acc, batch_mse, batch_mae, batch_acc

def test_individual_object_detection(model, individual_data_set, batch_size, mse_criterion, mae_criterion, device):
    model.eval()
    running_mse = 0.0
    running_mae = 0.0
    num_correct = 0

    for index in range(0, len(individual_data_set), batch_size):
        batch = individual_data_set[index:index + batch_size]
        data, targets = utilities.get_info_from_batch(batch, device)
        labels = utilities.get_labels_from_targets(targets)
        
        bounding_boxes = model(data)
        predictions = get_predictions(bounding_boxes)
        labels_tensor = torch.FloatTensor(labels)
        predictions_tensor = torch.FloatTensor(predictions)
        
        running_mse += mse_criterion(labels_tensor, predictions_tensor).item()
        running_mae += mae_criterion(labels_tensor, predictions_tensor).item()
        num_correct += (labels_tensor == predictions_tensor).sum().item()

    mse = running_mse/len(individual_data_set)
    mae = running_mae/len(individual_data_set)
    acc = num_correct/len(individual_data_set)
    return mse, mae, acc
    

def test_batch_object_detection(model, model_name, batch_data_set, mse_criterion, mae_criterion, device):
    model.eval()
    running_mse = 0.0
    running_mae = 0.0
    num_correct = 0
    all_labels, all_predictions = [], []

    for batch in batch_data_set:
        data, targets = utilities.get_info_from_batch(batch, device)
        labels = utilities.get_labels_from_targets(targets)

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
        
        running_mse += mse_criterion(torch.FloatTensor([max_label]), torch.FloatTensor([max_prediction])).item()
        running_mae += mae_criterion(torch.FloatTensor([max_label]), torch.FloatTensor([max_prediction])).item()
        if max_prediction == max_label:
            num_correct += 1
            
        all_labels.append(max_label)
        all_predictions.append(max_prediction)

    mse = running_mse/len(batch_data_set)
    mae = running_mae/len(batch_data_set)
    acc = num_correct/len(batch_data_set)
    return mse, mae, acc


def test_object_detection(model, model_name, batch_data_set, individual_data_set, batch_size, device, mse_criterion, mae_criterion):
    model.to(device)
    ind_mse, ind_mae, ind_acc = test_individual_object_detection(model, individual_data_set, batch_size, mse_criterion, mae_criterion, device)
    batch_mse, batch_mae, batch_acc = test_batch_object_detection(model, model_name, batch_data_set, mse_criterion, mae_criterion, device)
    return ind_mse, ind_mae, ind_acc, batch_mse, batch_mae, batch_acc
    
def get_testing_data(batch_size, data_dir, json_file_name_list, name_list):
    batch_testing_data, batch_testing_labels, individual_data, individual_labels = [], [], [], []
    for i in range(0, len(json_file_name_list)):
        print("\nGetting"+name_list[i]+"data")
        temp_batch_testing_data, temp_batch_testing_labels, temp_individual_data, temp_individual_labels = utilities.fetch_data(data_dir, json_file_name_list[i], False, False)
        batch_testing_data.extend(temp_batch_testing_data)
        batch_testing_labels.extend(temp_batch_testing_labels)
        individual_data.extend(temp_individual_data)
        individual_labels.extend(temp_individual_labels)
    batch_testing_data_set = ImageDataSet(batch_testing_data, batch_testing_labels)
    individual_testing_data_set = ImageDataSet(individual_data, individual_labels)
    return batch_testing_data_set, individual_testing_data_set

'''
def train_and_test(model_param, model, model_name, training_data, training_labels, validation_data_set, batch_training_data_set, batch_validation_data_set, batch_testing_data_set, device, saving_dir, best_result):
    
    curr_model = pickle.loads(pickle.dumps(model))
    training_data_size = int(len(training_data)*model_param[0])
    indices = random.sample(range(len(training_data)), training_data_size)
    training_data_subset = [training_data[i] for i in indices]
    training_label_subset = [training_labels[i] for i in indices]
    training_data_set = ImageDataSet(training_data_subset, training_label_subset)
    
    if torch.cuda.device_count() > 1:
        curr_model = nn.DataParallel(curr_model)
        result = train_and_validate(training_data_size, 15, curr_model, model_name, optim.Adam, model_param[1], training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, batch_testing_data_set, individual_testing_data_set, device, saving_dir, model_param[2], False, best_result)
    
    return result

def crossover(parents, r_cross):
    # crossover two parents to create two childrens
    p1 = deepcopy(parents[0])
    p2 = deepcopy(parents[1])
    # check for recombination
    if rand() < r_cross:
        pt = randint(1, len(p1)-1)
        # crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    else:
        c1, c2 = p1, p2
    return (c1, c2)

def mutation(param_comb, param_grid, r_mut):
    # mutate the hyperparameter combination
    temp_param = list(param_comb)
    for i in range(len(temp_param)):
        if rand() < r_mut:
            possible_params = list(param_grid.items())[i][1].copy()
            possible_params.remove(temp_param[i])
            temp_param[i] = random.choice(possible_params)
    return tuple(temp_param)
    
def genetic_algorithm(param_grid, model, model_name, n_iter, n_model, r_cross, r_mut, training_data, training_labels, validation_data_set, batch_training_data_set, batch_validation_data_set, batch_testing_data_set, device, saving_dir):
    # initial population of random hyperparameter combinations
    init_pop = list(itertools.product(param_grid['train_ratio'], param_grid['lr'], param_grid['batch_size']))
    curr_pop = random.sample(init_pop, n_model)
    best_result = None
    for gen in range(n_iter-1):
        # obtain the current top three performance models
        curr_pop_result = [(i, curr_pop[i], train_and_test(curr_pop[i], model, model_name, training_data, training_labels, validation_data_set, batch_training_data_set, batch_validation_data_set, batch_testing_data_set, device, saving_dir, best_result)) for i in range(n_model)]
        curr_pop_result.sort(key=lambda x: x[2])
        # tracking the best combination
        best_individuals = [curr_pop_result[i][1] for i in range(5)]
        if best_result is None or curr_pop_result[0][2] <= best_result:
            best_result = curr_pop_result[0][2]
            
        children = []
        while len(children) < n_model:
            parents = random.sample(best_individuals, 2)
            c1, c2 = crossover(parents, r_cross)
            mut_c1 = mutation(c1, param_grid, r_mut)
            children.append(mut_c1)
            mut_c2 = mutation(c2, param_grid, r_mut)
            children.append(mut_c2)
        curr_pop = children
'''
# -----------------------------------
start_time = timeit.default_timer()
# Declaring Constants
# num_epochs = 5
# batch_size = 100
object_detection_batch_size = 10
data_dir = "../saved_data/training_animal_count/"
saving_dir = "../saved_models/"
test_file_name_list = ['Cottonwood Eastface', 'NGilchrist Eastface', 'Fence Ends', 'Idaho']
cottonwood_eastface_json_file_name = "2023_Cottonwood_Eastface_5.30_7.10_key.json"
ngilchrist_eastface_json_file_name = "2022_NGilchrist_Eastface_055_07.12_07.20_key.json"
fence_ends_json_file_name = "2023_fence_ends_HERS0024_MP178_EAST_key.json"
idaho_json_file_name = "Idaho_loc_0099_key.json"
json_file_name_list = [cottonwood_eastface_json_file_name, ngilchrist_eastface_json_file_name, fence_ends_json_file_name, idaho_json_file_name]
testing_data_dir = "../saved_data/testing_animal_count/"

print(torch.__version__)
print(torchvision.__version__)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
param_grid = {
    'train_ratio': [0.2, 0.4, 0.6, 0.8, 1],
    #'num_epochs':[1, 5, 10, 15],
    #'dropout_rate': [0, 0.1, 0.2],
    #'optimizer': [optim.SGD, optim.Adam],
    'lr': [0.000001, 0.00001, 0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64, 80]
    #'momentum': [0, 0.9, 0.95, 0.99]
    #'decay_rate': [0, 0.01, 0.1],
}

#training_data, training_labels, validation_data, validation_labels, batch_training_data, batch_training_labels, batch_validation_data, batch_validation_labels = fetch_training_data(data_dir)
#training_data_set = ImageDataSet(training_data, training_labels)
#validation_data_set = ImageDataSet(validation_data, validation_labels)
#batch_training_data_set = ImageDataSet(batch_validation_data, batch_validation_labels)
#batch_validation_data_set = ImageDataSet(batch_validation_data, batch_validation_labels)

# Declaring Models
#resnet34 = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
#in_features = resnet34.fc.in_features
#resnet34.fc = nn.Linear(in_features, 1)


#resnet50 = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
#in_features = resnet50.fc.in_features
#resnet50.fc = nn.Linear(in_features, 1)

#resnet152 = models.resnet152(weights = models.ResNet152_Weights.DEFAULT)
#in_features = resnet152.fc.in_features
#resnet152.fc = nn.Linear(in_features, 1)

#vit_l_16 = models.vit_l_16(weights = models.ViT_L_16_Weights.DEFAULT)
#in_features = vit_l_16.heads[0].in_features
#vit_l_16.heads[0] = nn.Linear(in_features, 1)

#faster_rcnn = models.detection.fasterrcnn_resnet50_fpn_v2(weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

#ssd = models.detection.ssd300_vgg16(weights=models.detection.SSD300_VGG16_Weights.DEFAULT)

#retina_net = models.detection.retinanet_resnet50_fpn_v2(weights=models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)

#if torch.cuda.device_count() > 1:
    #print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
    #resnet34 = nn.DataParallel(resnet34)
    #resnet50 = nn.DataParallel(resnet50)
    #resnet152 = nn.DataParallel(resnet152)
    #vit_l_16 = nn.DataParallel(vit_l_16)
    #cnn_wrapper = nn.DataParallel(cnn_wrapper)

# Training

#print("\nTraining and Validating ResNet34")
#train_and_validate(num_epochs, resnet34, "ResNet34", optim, lr, training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, batch_size, False)

#print("\nTraining and Validating ResNet50")
#train_and_validate(num_epochs, resnet50, "ResNet50", optim, lr, training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, batch_size, False)

#print("\nTraining and Validating ResNet152")
#train_and_validate(num_epochs, resnet152, "ResNet152", optim, lr, training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, batch_size, False)
        
#print("\nTraining and Validating Vision Transformer Large 16")
#train_and_validate(num_epochs, vit_l_16, "ViTL16", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, batch_size, False, False)

#print("\nTraining and Validating Faster R-CNN")
#train_and_validate(num_epochs, faster_rcnn, "FasterR-CNN", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, object_detection_batch_size, True, False)
        
#print("\nTraining and Validating SSD")
#train_and_validate(num_epochs, ssd, "SSD", training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, object_detection_batch_size, True, False)
        
#print("\nTraining and Validating RetinaNet")
#train_and_validate(num_epochs, retina_net, "RetinaNet", optim, lr, training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, object_detection_batch_size, True)

# def train_and_validate(train_size, num_epochs, model, model_name, dropout_rate, optim, optim_lr, decay_rate, training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, device, saving_dir, batch_size, is_object_detection):

print('Load training datasets\n')
training_data, training_labels, validation_data, validation_labels, batch_training_data, batch_training_labels, batch_validation_data, batch_validation_labels = fetch_training_data(data_dir)
validation_data_set = ImageDataSet(validation_data, validation_labels)
batch_training_data_set = ImageDataSet(batch_validation_data, batch_validation_labels)
batch_validation_data_set = ImageDataSet(batch_validation_data, batch_validation_labels)

print('Load testing datasets\n')
batch_testing_data_set, individual_testing_data_set = get_testing_data(5, testing_data_dir, json_file_name_list, test_file_name_list)

'''
# Grid-Search Training for ResNet 18
print("\nHyperparameter for ResNet18")
num_epochs = 15
best_result = None
print("training size,epoch number,optimizer,learning rate,batch size,training loss,validation MSE,validation MAE,validation ACC,batch validation MSE,batch validation MAE,batch validation ACC,r2,individual testing MSE,individual testing MAE,individual testing ACC,batch testing MSE,batching testing MAE,batching testing ACC")
# optim_param_comb = list(itertools.product(param_grid['optimizer'], param_grid['lr'], param_grid['decay_rate'], param_grid['batch_size']))
optim_param_comb = list(itertools.product(param_grid['lr'], param_grid['batch_size']))
for train_ratio in param_grid["train_ratio"]:
    training_data_size = int(len(training_data)*train_ratio)
    indices = random.sample(range(len(training_data)), training_data_size)
    training_data_subset = [training_data[i] for i in indices]
    training_labels_subset = [training_labels[i] for i in indices]
    training_data_set = ImageDataSet(training_data_subset, training_labels_subset)
    #for dropout_rate in param_grid["dropout_rate"]:
    for optim_param in optim_param_comb:
        resnet18 = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        in_features = resnet18.fc.in_features
        resnet18.fc = nn.Linear(in_features, 1)
        optim_lr, batch_size = optim_param
        if torch.cuda.device_count() > 1:
            resnet18 = nn.DataParallel(resnet18)
        result = train_and_validate(training_data_size, num_epochs, resnet18, "ResNet18", optim.Adam, optim_lr, training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, batch_testing_data_set, individual_testing_data_set, device, saving_dir, batch_size, False, best_result)
        if best_result is None or result < best_result:
            best_result = result

# Grid-Search Training for ResNet 34
print("\nHyperparameter for ResNet34")
num_epochs = 15
best_result = None
print("training size,epoch number,optimizer,learning rate,batch size,training loss,validation MSE,validation MAE,validation ACC,batch validation MSE,batch validation MAE,batch validation ACC,r2,individual testing MSE,individual testing MAE,individual testing ACC,batch testing MSE,batching testing MAE,batching testing ACC")
# optim_param_comb = list(itertools.product(param_grid['optimizer'], param_grid['lr'], param_grid['decay_rate'], param_grid['batch_size']))
optim_param_comb = list(itertools.product(param_grid['lr'], param_grid['batch_size']))
for train_ratio in param_grid["train_ratio"]:
    training_data_size = int(len(training_data)*train_ratio)
    indices = random.sample(range(len(training_data)), training_data_size)
    training_data_subset = [training_data[i] for i in indices]
    training_labels_subset = [training_labels[i] for i in indices]
    training_data_set = ImageDataSet(training_data_subset, training_labels_subset)
    #for dropout_rate in param_grid["dropout_rate"]:
    for optim_param in optim_param_comb:
        resnet34 = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
        in_features = resnet34.fc.in_features
        resnet34.fc = nn.Linear(in_features, 1)
        optim_lr, batch_size = optim_param
        if torch.cuda.device_count() > 1:
            resnet34 = nn.DataParallel(resnet34)
        result = train_and_validate(training_data_size, num_epochs, resnet34, "ResNet34", optim.Adam, optim_lr, training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, batch_testing_data_set, individual_testing_data_set, device, saving_dir, batch_size, False, best_result)
        if best_result is None or result < best_result:
            best_result = result

# Grid-Search Training for ResNet 50

print("\nHyperparameter for ResNet50")
num_epochs = 15
best_result = None
print("training size,epoch number,optimizer,learning rate,batch size,training loss,validation MSE,validation MAE,validation ACC,batch validation MSE,batch validation MAE,batch validation ACC,r2,individual testing MSE,individual testing MAE,individual testing ACC,batch testing MSE,batching testing MAE,batching testing ACC")
optim_param_comb = list(itertools.product(param_grid['lr'], param_grid['batch_size']))
for train_ratio in param_grid["train_ratio"]:
    training_data_size = int(len(training_data)*train_ratio)
    indices = random.sample(range(len(training_data)), training_data_size)
    training_data_subset = [training_data[i] for i in indices]
    training_labels_subset = [training_labels[i] for i in indices]
    training_data_set = ImageDataSet(training_data_subset, training_labels_subset)
    #for dropout_rate in param_grid["dropout_rate"]:
    for optim_param in optim_param_comb:
        resnet50 = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        in_features = resnet50.fc.in_features
        resnet50.fc = nn.Linear(in_features, 1)
        optim_lr, batch_size = optim_param
        if torch.cuda.device_count() > 1:
            resnet50 = nn.DataParallel(resnet50)
        result = train_and_validate(training_data_size, num_epochs, resnet50, "ResNet50", optim.Adam, optim_lr, training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, batch_testing_data_set, individual_testing_data_set, device, saving_dir, batch_size, False, best_result)
        if best_result is None or result < best_result:
            best_result = result

# Grid-Search Training for ResNet 152

print("\nHyperparameter for ResNet152")
num_epochs = 15
best_result = None
print("training size,epoch number,optimizer,learning rate,batch size,training loss,validation MSE,validation MAE,validation ACC,batch validation MSE,batch validation MAE,batch validation ACC,r2,individual testing MSE,individual testing MAE,individual testing ACC,batch testing MSE,batching testing MAE,batching testing ACC")
optim_param_comb = list(itertools.product(param_grid['lr'], param_grid['batch_size']))
for train_ratio in param_grid["train_ratio"]:
    training_data_size = int(len(training_data)*train_ratio)
    indices = random.sample(range(len(training_data)), training_data_size)
    training_data_subset = [training_data[i] for i in indices]
    training_labels_subset = [training_labels[i] for i in indices]
    training_data_set = ImageDataSet(training_data_subset, training_labels_subset)
    #for dropout_rate in param_grid["dropout_rate"]:
    for optim_param in optim_param_comb:
        resnet152 = models.resnet152(weights = models.ResNet152_Weights.DEFAULT)
        in_features = resnet152.fc.in_features
        resnet152.fc = nn.Linear(in_features, 1)
        optim_lr, batch_size = optim_param
        if torch.cuda.device_count() > 1:
            resnet152 = nn.DataParallel(resnet152)
        result = train_and_validate(training_data_size, num_epochs, resnet152, "ResNet152", optim.Adam, optim_lr, training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, batch_testing_data_set, individual_testing_data_set, device, saving_dir, batch_size, False, best_result)
        if best_result is None or result < best_result:
            best_result = result
'''

# Grid-Search Training for ResNet 101

print("\nHyperparameter for ResNet101")
num_epochs = 15
best_result = None
print("training size,epoch number,optimizer,learning rate,batch size,training loss,validation MSE,validation MAE,validation ACC,batch validation MSE,batch validation MAE,batch validation ACC,r2,individual testing MSE,individual testing MAE,individual testing ACC,batch testing MSE,batching testing MAE,batching testing ACC")
optim_param_comb = list(itertools.product(param_grid['lr'], param_grid['batch_size']))
for train_ratio in param_grid["train_ratio"]:
    training_data_size = int(len(training_data)*train_ratio)
    indices = random.sample(range(len(training_data)), training_data_size)
    training_data_subset = [training_data[i] for i in indices]
    training_labels_subset = [training_labels[i] for i in indices]
    training_data_set = ImageDataSet(training_data_subset, training_labels_subset)
    #for dropout_rate in param_grid["dropout_rate"]:
    for optim_param in optim_param_comb:
        resnet101 = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)
        in_features = resnet101.fc.in_features
        resnet101.fc = nn.Linear(in_features, 1)
        optim_lr, batch_size = optim_param
        if torch.cuda.device_count() > 1:
            resnet101 = nn.DataParallel(resnet101)
        result = train_and_validate(training_data_size, num_epochs, resnet101, "ResNet101", optim.Adam, optim_lr, training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, batch_testing_data_set, individual_testing_data_set, device, saving_dir, batch_size, False, best_result)
        if best_result is None or result < best_result:
            best_result = result

     
'''
print("\nHyperparameter for RetinaNet")
num_epochs = 15
best_result = None
print("training size,epoch number,optimizer,learning rate,batch size,training loss,validation MSE,validation MAE,validation ACC,batch validation MSE,batch validation MAE,batch validation ACC,r2,individual testing MSE,individual testing MAE,individual testing ACC,batch testing MSE,batch testing MAE,batch testing ACC")
optim_param_comb = list(itertools.product(param_grid['lr'], param_grid['batch_size']))
for train_ratio in param_grid["train_ratio"]:
    training_data_size = int(len(training_data)*train_ratio)
    indices = random.sample(range(len(training_data)), training_data_size)
    training_data_subset = [training_data[i] for i in indices]
    training_labels_subset = [training_labels[i] for i in indices]
    training_data_set = ImageDataSet(training_data_subset, training_labels_subset)
    #for dropout_rate in param_grid["dropout_rate"]:
    for optim_param in optim_param_comb:
        retina_net = models.detection.retinanet_resnet50_fpn_v2(weights=models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        optim_lr, batch_size = optim_param
        result = train_and_validate(training_data_size, num_epochs, retina_net, "RetinaNet", optim.Adam, optim_lr, training_data_set, validation_data_set, batch_training_data_set, batch_validation_data_set, batch_testing_data_set, individual_testing_data_set, device, saving_dir, batch_size, True, best_result)
        if best_result is None or result < best_result:
            best_result = result

# Genetic Algorithm for Hyperparameter Tuning

# number of models to be evaluated in each iteration
n_model = 15
# number of iterations
n_iter = 4
# crossover rate
r_cross = 0.9
# mutation (depends on the number of hyperparameters)
r_mut = 0.3

# Resnet34
resnet34 = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
in_features = resnet34.fc.in_features
resnet34.fc = nn.Linear(in_features, 1)
genetic_algorithm(param_grid, resnet34, 'ResNet34', n_iter, n_model, r_cross, r_mut, training_data, training_labels, validation_data_set, batch_training_data_set, batch_validation_data_set, batch_testing_data_set, device, saving_dir)
'''

elapsed = timeit.default_timer() - start_time
print("{0:.2f} sec".format(elapsed))