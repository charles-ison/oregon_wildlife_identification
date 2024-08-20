import os
import torch
import json
import random
import shutil
import sys
import numpy as np
import pandas as pd
import torch.nn as nn
import scipy.optimize
from pycocotools.coco import COCO
from pingouin import intraclass_corr
from pybboxes import BoundingBox
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score

# Adding model_benchmarking/ to the system path
sys.path.append("..")
import utilities

def sample_data(data_dir, if_training, target_dir, json_file_name=None, testing_folder_names=None):
    total_sampled_imgs = 0
    if if_training:
        json_file_name = "animal_count_key.json"
        for directory in os.scandir(data_dir):
            if directory.is_dir():
                directory_path = directory.path + "/"
                # Skip caltech because it is supplemental and not labeled by us
                if "caltech" in directory_path:
                    continue
                else:
                    print("\nSample data from directory: ", directory_path)
                    num_sampled_imgs = sample_data_from_single_folder(directory_path, json_file_name, target_dir, True)
                    total_sampled_imgs += num_sampled_imgs
    else:
        for i in range(len(json_file_name)):
            print("\nSample data from directory: ", data_dir + testing_folder_names[i])
            num_sampled_imgs = sample_data_from_single_folder(data_dir, json_file_name[i], target_dir, False, testing_folder_names[i])
            total_sampled_imgs += num_sampled_imgs
            
    return total_sampled_imgs

def sample_data_from_single_folder(data_dir, json_file_name, target_dir, is_training, folder_name=None):
    coco = COCO(data_dir + json_file_name)
    images = coco.loadImgs(coco.getImgIds())
    sorted_images = utilities.get_sorted_images(images)
        
    data_ids, data_paths, file_names, labels = [], [], [], []
        
    for image in sorted_images:
        file_name = image["file_name"]
        file_path = data_dir + file_name
            
        annotation_id_list = coco.getAnnIds(imgIds=[image["id"]])
        annotation_list = coco.loadAnns(annotation_id_list)
            
        if os.path.isfile(file_path):
            data_ids.append(image["id"])
            data_paths.append(file_path)
            file_names.append(file_name)
            labels.append(annotation_list)
        else:
            print("No file found for: ", file_path)
        
    print("Number of images found:", len(data_paths))

    sampled_indices = sorted(random.sample(range(0, len(data_paths)), int(len(data_paths)*0.05)))

    sampled_ids = [data_ids[i] for i in sampled_indices]
    sampled_images = [image for image in sorted_images if image['id'] in sampled_ids]
    sampled_data_paths = [data_paths[i] for i in sampled_indices]
    sampled_labels = [label for i in sampled_indices for label in labels[i]]
    sampled_filenames = [file_names[i] for i in sampled_indices]
        
    if is_training:
        target_folder = target_dir + data_dir.split('/')[-2]
    else:
        target_folder = target_dir + folder_name
        #target_folder = target_dir + file_name.split('/')[0]
        #folder_name = file_name.split('/')[0]
    os.makedirs(target_folder, exist_ok=True)
    
    try:
        with os.scandir(target_folder) as files:
            for file in files:
                if file.is_file():
                    os.unlink(file.path)
            print("All files deleted successfully.")
    except OSError:
        print("No files exist.")
        
    # Write sampled label to the JSON file
    file_name = "animal_count_key.json"
    file_path = os.path.join(target_folder, file_name)
            
    # Copy sampled images into the new folder
    for i in range(len(sampled_data_paths)):
        if is_training:
            file_name = sampled_filenames[i].replace("/", "_")
        else:
            file_name = sampled_filenames[i].replace(folder_name+"/", "").replace("/", "_")
        sampled_images[i]['file_name'] = file_name
        target_path = target_folder+"/"+file_name
        shutil.copy(sampled_data_paths[i], target_path)
        
    new_coco = {
        "licenses": coco.dataset.get('licenses', []),
        "info": coco.dataset.get('info', {}),
        "categories": coco.dataset.get('categories', []),
        "images": sorted(sampled_images, key=lambda x: x["id"]),
        "annotations": sorted(sampled_labels, key=lambda x: x["id"]),
    }
                    
    with open(file_path, 'w') as json_file:
        json.dump(new_coco, json_file, indent=4)
    
    print("Number of images sampled: ", len(new_coco['images']), ' Number of labels sampled: ', len(new_coco["annotations"]))
    
    return len(new_coco['images'])

def iou_for_single_label(bboxA, bboxB):
    coco_bboxA = BoundingBox.from_coco(*bboxA)
    coco_bboxB = BoundingBox.from_coco(*bboxB)
    
    return coco_bboxA.iou(coco_bboxB)
    
def iou_for_multilabels_on_one_image(bboxes_gt_ls, bboxes_pred_ls):
    n_gt = len(bboxes_gt_ls)
    n_pred = len(bboxes_pred_ls)
    
    iou_matrix = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            iou_matrix[i, j] = iou_for_single_label(bboxes_gt_ls[i], bboxes_pred_ls[j])
            
    # Hungarian matching to find the best matched pairs of old and new labels yielding largest iou
    idxs_gt, idxs_pred = scipy.optimize.linear_sum_assignment(-iou_matrix)
    
    matched_iou = []
    
    for i, j in zip(idxs_gt, idxs_pred):
        matched_iou.append(iou_matrix[i, j])
    
    return np.mean(matched_iou)
    
def inter_over_union(x1, x2):
    total_iou = 0
    n_annotated_imgs = 0
    
    for i in range(len(x1)):
        if len(x1[i]) == 0 and len(x2[i]) == 0:
            continue
        else:
            n_annotated_imgs += 1
            curr_iou = 0
            if len(x1[i]) == 0 or len(x2[i]) == 0:
                continue
            elif len(x1[i]) == 1 and len(x2[i]) == 1:
                curr_iou = iou_for_single_label(x1[i][0], x2[i][0])
            else:
                curr_iou = iou_for_multilabels_on_one_image(x1[i], x2[i])
            if curr_iou > 0:
                total_iou += curr_iou
                
    if n_annotated_imgs > 0:
        return total_iou / n_annotated_imgs
    else:
        return "N/A"

def icc2(x1, x2):
    # Two-way random effects, absolute agreement, single rater/measurement
    data = pd.DataFrame({
        'Images': list(range(len(x1))) + list(range(len(x2))),
        'Labeler': [1] * len(x1) + [2] * len(x2), 
        'Count': x1 + x2
    })
    icc_df = intraclass_corr(data=data, targets='Images', raters='Labeler', ratings='Count')
    icc = icc_df[icc_df["Type"] == "ICC2"]["ICC"].values[0]
    
    return icc

def load_label_and_count_from_coco(coco_data, images):
    labels, count = [], []
    
    for image in images:
        annotation_id_list = coco_data.getAnnIds(imgIds=[image["id"]])
        annotation_list = coco_data.loadAnns(annotation_id_list)
        if len(annotation_list) == 0:
            labels.append(annotation_list)
        else:
            annotations = []
            for annotation in annotation_list:
                annotations.append(annotation["bbox"])
            labels.append(annotations)
        count.append(len(annotation_list))
        
    return labels, count
    
def quality_check_for_folder(directory_path):
    old_json_file = directory_path + "animal_count_key.json"
    new_json_file = directory_path + "animal_count_key_new.json"

    old_coco = COCO(old_json_file)
    new_coco = COCO(new_json_file)
    old_images = old_coco.loadImgs(old_coco.getImgIds())
    old_images = sorted(old_images, key=lambda x:x["file_name"])
    new_images = new_coco.loadImgs(new_coco.getImgIds())
    new_images = sorted(new_images, key=lambda x:x["file_name"])
    if len(old_images) == len(new_images):
        num_images = len(old_images)
        print("Total number of sampled images for quality check: ", num_images)
    else:
        print("Relabeled images cannot match sampled images.")
    
    old_labels, old_count = load_label_and_count_from_coco(old_coco, old_images)
    new_labels, new_count = load_label_and_count_from_coco(new_coco, new_images)
    
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    num_equal_labels = sum(len(old_label) == len(new_label) for old_label, new_label in zip(old_labels, new_labels))
    
    mse = mse_criterion(torch.FloatTensor(old_count), torch.FloatTensor(new_count)).item()
    mae = mae_criterion(torch.FloatTensor(old_count), torch.FloatTensor(new_count)).item()
    accuracy = num_equal_labels / len(old_count)
    f1 = f1_score(old_count, new_count, average = "weighted")
    if old_count == new_count:
        cohen_kappa = 1.0
        icc = 1.0
    else:
        cohen_kappa = cohen_kappa_score(old_count, new_count, weights = "linear")
        icc = icc2(old_count, new_count)
    iou = inter_over_union(old_labels, new_labels)
    
    return num_images, mae, mse, accuracy, f1, cohen_kappa, icc, iou

def quality_check_for_all_samples(target_dir):
    img_lengths, all_mae, all_mse, all_accuracy, all_f1, all_cohen, all_icc, all_iou = [], [], [], [], [], [], [], []
    for directory in os.scandir(target_dir):
        if directory.is_dir():
            directory_path = directory.path + "/"
            print("\nCheck labels from directory: ", directory_path)
            num_images, mae, mse, accuracy, f1, cohen_kappa, icc, iou = quality_check_for_folder(directory_path)
            print("Quality check results:")
            print("MAE:", mae, "    MSE:", mse, "   Accuracy:", accuracy, "    F1-score:", f1)
            print("Weighted Cohen's Kappa:", cohen_kappa)
            print("Intraclass correlation coefficient: ", icc)
            print("IoU:", iou)
            img_lengths.append(num_images)
            all_mae.append(mae)
            all_mse.append(mse)
            all_accuracy.append(accuracy)
            all_f1.append(f1)
            all_cohen.append(cohen_kappa)
            all_icc.append(icc)
            all_iou.append(iou)
    
    avg_mae, avg_mse, avg_accuracy, avg_f1, avg_cohen, avg_icc, avg_iou, weight, weight_i = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(len(img_lengths)):
        avg_mae += all_mae[i]*img_lengths[i]
        avg_mse += all_mse[i]*img_lengths[i]
        avg_accuracy += all_accuracy[i]*img_lengths[i]
        avg_f1 += all_f1[i]*img_lengths[i]
        avg_cohen += all_cohen[i]*img_lengths[i]
        avg_icc += all_icc[i]*img_lengths[i]
        weight += img_lengths[i]
        if type(all_iou[i]) != str:
            avg_iou += all_iou[i]*img_lengths[i]
            weight_i += img_lengths[i]
        if i == len(img_lengths)-1:
            avg_mae /= weight
            avg_mse /= weight
            avg_accuracy /= weight
            avg_f1 /= weight
            avg_cohen /= weight
            avg_icc /= weight
            avg_iou /= weight_i
    
    return sum(img_lengths), avg_mae, avg_mse, avg_accuracy, avg_f1, avg_cohen, avg_icc, avg_iou
    
target_dir = "../../label_quality_check/"
training_data_dir = "../../saved_data/training_animal_count/"
testing_data_dir = "../../saved_data/testing_animal_count/"
json_file_names = ["2022_NGilchrist_Eastface_055_07.12_07.20_key.json", "2023_Cottonwood_Eastface_5.30_7.10_key.json", "2023_fence_ends_HERS0024_MP178_EAST_key.json", "Idaho_loc_0099_key.json"]
testing_folder_names = ["2022_NGilchrist_Eastface_055_07.12_07.20", "2023_Cottonwood_Eastface_5.30_7.10", "2023_fence_ends_HERS0024_MP178_EAST", "Idaho"]

# Step 1: Sample data from training and testing folders

#sampled_training_imgs = sample_data(training_data_dir, True, target_dir)
#sampled_testing_imgs = sample_data(testing_data_dir, False, target_dir, json_file_names, testing_folder_names)
#print("\n --------------------------------------------------------------------------------- ")
#print("Total images sampled: ", sampled_training_imgs + sampled_testing_imgs)

# Step 2: Compute the MAE, MSE, Accuracy, and IoU

num_images, mae, mse, accuracy, f1, cohen_kappa, icc, iou = quality_check_for_all_samples(target_dir)
print("\n --------------------------------------------------------------------------------- ")
print("Total number of images sampled for the label quality check:  ", num_images)
print("Overall quality check results (weighted average) for all sampled training and testing data folders: ")
print("MAE:", mae, "    MSE:", mse, "   Accuracy:", accuracy, "    F1-score:", f1)
print("Weighted Cohen's Kappa:", cohen_kappa)
print("Intraclass correlation coefficient: ", icc)
print("IoU:", iou)
