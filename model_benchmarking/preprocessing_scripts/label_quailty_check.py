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
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

# Adding model_benchmarking/utilities.py to the system path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
import utilities

def sample_data(data_dir, if_training, target_dir, json_file_name=None):
    if if_training:
        json_file_name = "animal_count_key.json"
        for directory in os.scandir(data_dir):
            if directory.is_dir():
                directory_path = directory.path + "/"
                print("\nSample data from directory: ", directory_path)
                temp_data, temp_lab = sample_data_from_folder(directory_path, json_file_name, target_dir, True)
                print("Number of images sampled: ", len(temp_data), ' Number of labels sampled: ', len(temp_lab))
    else:
        sample_data_from_folder(data_dir, json_file_name, target_dir, False)

def sample_data_from_folder(data_dir, json_file_name, target_dir, if_training):
    if if_training:
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
                continue
        
        print("Number of images found:", len(data_paths))
        
        if "caltech" in data_dir:
            sampled_indices = sorted(random.sample(range(0, len(data_paths)), 300))
        else:
            sampled_indices = sorted(random.sample(range(0, len(data_paths)), int(len(data_paths)*0.05)))
        # use percent 5%
        sampled_ids = [data_ids[i] for i in sampled_indices]
        sampled_images = [image for image in sorted_images if image['id'] in sampled_ids]
        sampled_data_paths = [data_paths[i] for i in sampled_indices]
        sampled_labels = [label for i in sampled_indices for label in labels[i]]
        sampled_filenames = [file_names[i] for i in sampled_indices]
        
        target_folder = target_dir + data_dir.split('/')[-2]
        os.makedirs(target_folder, exist_ok=True)
    
        try:
            with os.scandir(target_folder) as files:
                for file in files:
                    if file.is_file():
                        os.unlink(file.path)
                print("All files deleted successfully.")
        except OSError:
            print("No files exist.")
        
        # write sampled label to the JSON file
        file_name = data_dir.split('/')[-2]+'_sampled_animal_count_key.json'
        file_path = os.path.join(target_folder, file_name)
            
        # copy sampled images into the new folder
        for i in range(len(sampled_data_paths)):
            file_name = sampled_filenames[i].replace("/", "_")
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
        
        if "caltech" in data_dir:
            new_coco["categories"] = [{
                "id": 1,
                "name": "deer"
            }]
            new_annotations = []
            for label in new_coco["annotations"]:
                if label["category_id"] == 1:
                    label["segmentation"] = []
                    new_annotations.append(label)
            new_coco["annotations"] = new_annotations
                    
        with open(file_path, 'w') as json_file:
            json.dump(new_coco, json_file, indent=4)
        
        return new_coco['images'], new_coco["annotations"]
    
    else:
        for i_json_file_name in json_file_name:
            coco = COCO(data_dir + i_json_file_name)
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
                    continue
            
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
                continue

            sampled_indices = sorted(random.sample(range(0, len(sorted_images)), int(len(sorted_images)*0.05)))
            sampled_ids = [data_ids[i] for i in sampled_indices]
            sampled_images = [image for image in sorted_images if image['id'] in sampled_ids]
            sampled_data_paths = [data_paths[i] for i in sampled_indices]
            sampled_labels = [label for i in sampled_indices for label in labels[i]]
            sampled_filenames = [file_names[i] for i in sampled_indices]
            
            folder_name = file_name.split('/')[0]
            print("\nSample data from directory: ", data_dir + folder_name)
            target_folder = target_dir + file_name.split('/')[0]
            os.makedirs(target_folder, exist_ok=True)
        
            try:
                with os.scandir(target_folder) as files:
                    for file in files:
                        if file.is_file():
                            os.unlink(file.path)
                    print("All files deleted successfully.")
            except OSError:
                print("No files exist.")
            
            # write sampled label to the JSON file
            file_name = folder_name+'_sampled_animal_count_key.json'
            file_path = os.path.join(target_folder, file_name)
                
            # copy sampled images into the new folder
            for i in range(len(sampled_images)):
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
            
            print("Number of images sampled: ", len(new_coco["images"]), ' Number of labels sampled: ', len(new_coco["annotations"]))
                
        return None
        
def iou_for_bboxes(bboxA, bboxB):

    xA_tl, yA_tl, wA, hA = bboxA
    xB_tl, yB_tl, wB, hB = bboxB
    
    xA_br = xA_tl + wA
    yA_br = yA_tl - hA
    xB_br = xB_tl + wB
    yB_br = yB_tl - hB
    
    # intersection area
    xAB_tl = max(xA_tl, xB_tl)
    yAB_tl = min(yA_tl, yB_tl)
    xAB_br = min(xA_br, xB_br)
    yAB_br = max(yA_br, yB_br)
    w_AB = xAB_br - xAB_tl
    h_AB = yAB_tl - yAB_br
    inter_area = w_AB * h_AB
    if inter_area <= 0:
        return -1
    
    union_area = wA * hA + wB * hB - inter_area
    
    iou = inter_area / float(union_area)
    if iou > 1:
        return -1
    
    return iou
    
# Method adapted from: https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4 
def iou_for_multilabels_on_one_image(bboxes_gt_ls, bboxes_pred_ls, iou_limit = 0.5):
    n_gt = len(bboxes_gt_ls)
    n_pred = len(bboxes_pred_ls)
    min_iou = 0.0
    
    iou_matrix = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            iou_matrix[i, j] = iou_for_bboxes(bboxes_gt_ls[i], bboxes_pred_ls[j])
            
    # Hungarian matching to find the best matched pairs of old and new labels yielding largest iou
    idxs_gt, idxs_pred = scipy.optimize.linear_sum_assignment(-iou_matrix)
    
    matched_iou = []
    
    for i, j in zip(idxs_gt, idxs_pred):
        matched_iou.append(iou_matrix[i, j])
    
    return np.mean(matched_iou)

def critical_success_index(y_gt, y_pred):
    mcm = multilabel_confusion_matrix(y_gt, y_pred)
    tp = mcm[:, 1, 1].sum()
    fn = mcm[:, 1, 0].sum()
    fp = mcm[:, 0, 1].sum()
    return tp / (tp + fn + fp)
    
def intraclass_corr(x1, x2):
    # Two-way random effects, absolute agreement, single rater/measurement
    data = pd.DataFrame({'Rater1': x1, 
                         'Rater2': x2})
    n, k = data.shape

    ms_r = np.var(data.mean(axis=1), ddof=1) * k
    ms_c = np.var(data.mean(axis=0), ddof=1) * n
    ms_e = 0
    for i in range(n):
      for j in range(k):
        squared_error = (data.iloc[i, j] - np.mean(data.iloc[i, :]) - np.mean(data.iloc[:, j]) + np.mean(data.values.flatten()))**2 
        ms_e += squared_error
    ms_e /= n*(k-1)

    icc = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
    
    return icc
    
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

    old_labels, old_count, new_labels, new_count = [], [], [], []
    for old_image in old_images:
        annotation_id_list = old_coco.getAnnIds(imgIds=[old_image["id"]])
        annotation_list = old_coco.loadAnns(annotation_id_list)
        if len(annotation_list) == 0:
            old_labels.append(annotation_list)
        else:
            annotations = []
            for annotation in annotation_list:
                annotations.append(annotation["bbox"])
            old_labels.append(annotations)
        old_count.append(len(annotation_list))
                
    for new_image in new_images:
        annotation_id_list = new_coco.getAnnIds(imgIds=[new_image["id"]])
        annotation_list = new_coco.loadAnns(annotation_id_list)
        if len(annotation_list) == 0:
            new_labels.append(annotation_list)
        else:
            annotations = []
            for annotation in annotation_list:
                annotations.append(annotation["bbox"])
            new_labels.append(annotations)
        new_count.append(len(annotation_list))
    
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    num_equal_labels = sum(len(old_label) == len(new_label) for old_label, new_label in zip(old_labels, new_labels))
    
    mse = mse_criterion(torch.FloatTensor(old_count), torch.FloatTensor(new_count)).item()
    mae = mae_criterion(torch.FloatTensor(old_count), torch.FloatTensor(new_count)).item()
    accuracy = num_equal_labels / len(old_count)
    f1 = f1_score(old_count, new_count, average = "weighted")
    #matthews_coef = matthews_corrcoef(old_count, new_count)
    #csi = critical_success_index(old_count, new_count)
    if np.count_nonzero(np.array(old_count)) == 0 and np.count_nonzero(np.array(new_count)) == 0:
        cohen_kappa = 1.0
        icc = 1.0
    else:
        cohen_kappa = cohen_kappa_score(old_count, new_count, weights = "linear")
        icc = intraclass_corr(old_count, new_count)
    
    total_iou = 0
    n_labels = 0
    for i in range(len(old_labels)):
        if len(old_labels[i]) == 0 and len(new_labels[i]) == 0:
            continue
        elif len(old_labels[i]) == 0 or len(new_labels[i]) == 0:
            total_iou += 0
            n_labels += 1
        elif len(old_labels[i]) == 1 and len(new_labels[i]) == 1:
            curr_iou = iou_for_bboxes(old_labels[i][0], new_labels[i][0])
            if curr_iou > 0:
                total_iou += curr_iou
            else:
                total_iou += 0
            n_labels += 1
        else:
            old_label = sorted(old_labels[i], key=lambda x: x[0])
            new_label = sorted(new_labels[i], key=lambda x: x[0])
            curr_iou = iou_for_multilabels_on_one_image(old_label, new_label)
            if curr_iou < 0:
                total_iou += 0
            else:
                total_iou += curr_iou
            n_labels += 1
                
    if n_labels > 0:
        iou = total_iou / n_labels
    else:
        iou = "N/A"
    
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
            #print("Matthews Correlation Coefficient:", matthews_coef)
            print("Weighted Cohen's Kappa:", cohen_kappa)
            print("Intraclass correlation coefficient: ", icc)
            #print("Threat Score:", csi)
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

#sample_data(training_data_dir, True, target_dir)
#sample_data(testing_data_dir, False, target_dir, json_file_names)

# Step 2: Compute the MAE, MSE, Accuracy, and IoU

num_images, mae, mse, accuracy, f1, cohen_kappa, icc, iou = quality_check_for_all_samples(target_dir)
print("\n --------------------------------------------------------------------------------- ")
print("Total number of images sampled for the label quality check:  ", num_images)
print("Overall quality check results (weighted average) for all sampled training and testing data folders: ")
print("MAE:", mae, "    MSE:", mse, "   Accuracy:", accuracy, "    F1-score:", f1)
#print("Matthews Correlation Coefficient:", matthews_coef)
print("Weighted Cohen's Kappa:", cohen_kappa)
print("Intraclass correlation coefficient: ", icc)
#print("Threat Score:", csi)
print("IoU:", iou)


        
