import argparse

import scipy as sp
from pprint import pprint

import torch
import utils as ut
import tqdm
import os
import pickle as pkl
from torchvision import datasets, transforms
from model import Classifier, CD_Simu, Cycle_Cons, Encoder, Decoder
from unet_model import Cycle_Cons_3_Improvements, Convnet_SkipConnection
from torch import  nn, optim
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from nilearn import image
import nibabel as nib

parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=512,    help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=7500, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=50, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=93,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=15,     help="Flag for training")
parser.add_argument('--Type',     type=str, default='ADNI_CLS',     help="Flag for training")
''' Type: Run_cls / ADNI_CLS'''
parser.add_argument('--iter_restart', type=int, default=1400, help="Save model every n iterations")
parser.add_argument('--BATCH_SIZE',     type=int, default=64,     help="Flag for training")
parser.add_argument('--iter_load',     type=int, default=1,     help="Flag for loading version of model")
parser.add_argument('--Siamese',     type=str, default='SiameseNetAEReg', help="SiameseNetAE\SiameseNet\SiameseNetW\SiamgeseNetAEReg")
args = parser.parse_args()

cyc_con_layout = [
    ('model={:s}',  'Cyc_con'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
cyc_con_model_name = '_'.join([t.format(v) for (t, v) in cyc_con_layout])

ctrans_layout = [
    ('model={:s}',  'ctrans'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
ctrans_model_name = '_'.join([t.format(v) for (t, v) in ctrans_layout])

dtrans_layout = [
    ('model={:s}',  'dtrans'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
dtrans_model_name = '_'.join([t.format(v) for (t, v) in dtrans_layout])

Classifier_layout = [
    ('model={:s}',  'Classifier'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
Classifier_model_name = '_'.join([t.format(v) for (t, v) in Classifier_layout])

cls_list= [0,1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ut.set_seed(2020)


pathall_saveimg = "/ADNI/ADNI_Longitudinal_all/save_image/"
IPMI_save = "/ADNI/IPMI/checkpoints/"
ADNI_path_group_data = "/data/adni/img_64_longitudinal/"

ADNI_data_name_list_txt = "adni_list.txt"

IPMI_data_pickle_save_path = "/ADNI/IPMI/data_pickle_saved/"
FLAG_Save_data_pickle = False
DEBUG_VERBOSE = False

label_file_path = IPMI_data_pickle_save_path + 'ADNI1AND2.csv'
if FLAG_Save_data_pickle == True:
    ut.get_Id_attr_pkl_from_label_file(Dataset, label_file_path, IPMI_data_pickle_save_path) 
else:
    f = open(IPMI_data_pickle_save_path + "Id_attr.pkl","rb")
    Id_attr = pkl.load(f)
    f.close() 
    print('Load Id_attr.pkl success!')  


print(Id_attr)

def extract_metadata_list_from_Idx_attr(Id_attr, label='Group', label_list=['AD', 'Normal']):
    metadata_list = [[] for i in range(len(label_list))]
    label2num = ut.get_label2num(num_cls=3)
    for subject_idx, data_dict in Id_attr.items():
        l = data_dict[label]
        if l in label_list:
            l_num = label2num[l]
            age = data_dict['Age']
            sex = data_dict['Sex']
            metadata_list[l_num].append([subject_idx, l, l_num, age, sex])
    return metadata_list

metadata_list = extract_metadata_list_from_Idx_attr(Id_attr)
print(metadata_list)
print(len(metadata_list[0]), len(metadata_list[1]))

def sort_metadata_list(metadata_list, sort_idx=3):
    new_metadata_list = []
    for i in range(len(metadata_list)):
        sort_array = np.asarray([metadata_list[i][j][sort_idx] for j in range(len(metadata_list[i]))])
        arg_idx = np.argsort(sort_array)
        new_metadata = [metadata_list[i][idx] for idx in arg_idx]
        new_metadata_list.append(new_metadata)
    
    return new_metadata_list

new_metadata_list = sort_metadata_list(metadata_list)
print(new_metadata_list)

def find_comparable_pair_wrt_age(metadata_1, metadata_2):
    pair_idx = []
    cnt_min_idx = 0
    for i in range(len(metadata_1)):
        age_1 = metadata_1[i]
        min_gap = 1000
        min_idx = cnt_min_idx
        min_age = -1
        FLAG_smaller = True
        for j in range(cnt_min_idx, len(metadata_2)):
            age_2 = metadata_2[j]
            gap = abs(age_1 - age_2)
            
            if gap < min_gap:
                min_gap = gap
                min_idx = j
                min_age = age_2
            if age_2 > age_1:
                break
        pair_idx.append([i, min_idx, age_1, min_age, round(min_gap, 1)])
        cur_min_idx = min_idx 

    return pair_idx 

metadatas = []
for i in range(2):
    m = [new_metadata_list[i][j][3] for j in range(len(new_metadata_list[i]))]
    metadatas.append(m)

print(metadatas)

pair_idx = find_comparable_pair_wrt_age(metadatas[0], metadatas[1])
# print(pair_idx)
# print(pair_idx[55])
# sleep(1000)
def subject_idx_list_given_pair_metadata(idx_file_path, pair_idx, metadata_list):
    test_data_num = len(pair_idx)
    file_idx = np.genfromtxt(idx_file_path, dtype='str')
    print(file_idx)
    print(file_idx.shape)
    file_idx_prefix = np.asarray([file_idx[i][:10] for i in range(len(file_idx))], dtype='str')
    print(file_idx_prefix)
    test_data_list = np.zeros((test_data_num, 2), dtype='<U100')
    for i in range(test_data_num):
        subject_idx_0 = metadata_list[0][pair_idx[i][0]][0]
        subject_idx_1 = metadata_list[1][pair_idx[i][1]][0]
        # print(subject_idx_0)
        if subject_idx_0 in file_idx_prefix:
            first_index_0 = np.where(file_idx_prefix == subject_idx_0)[0][0]
        else:
            continue
        if subject_idx_1 in file_idx_prefix:
            first_index_1 = np.where(file_idx_prefix == subject_idx_1)[0][0]
        else:
            continue
        test_data_list[i][0] = file_idx[first_index_0]
        test_data_list[i][1] = file_idx[first_index_1]
    return test_data_list

test_data_list = subject_idx_list_given_pair_metadata(ADNI_data_name_list_txt, pair_idx, new_metadata_list)
# print(test_data_list)
# print(new_metadata_list[1][pair_idx[10][1]][0])
# print(test_data_list[10][1])
# sleep(1000)

def get_test_dataset_from_pair_idx_and_metadata_list(test_data_list, data_path, normalize=True, normalized_method='z_score', verbose=True):
    print("data_path=", data_path)   
    test_data_num = len(test_data_list)
    print("the num of image is:", test_data_num)
    patch_x = patch_y = patch_z = 64
    min_x = min_y = min_z = 0
    dataset = np.zeros((test_data_num, 2, patch_x, patch_y, patch_z), dtype=np.float32)
    FLAG_picked = np.zeros(test_data_num, dtype=bool)
    cnt = 0
    mean_list = [[], []]
    std_list = [[], []]
    for i in range(test_data_num):
        for j in range(2):
            subject_idx = test_data_list[i][j]
            filename_full = data_path + subject_idx
            img = nib.load(filename_full)
            img_data = img.get_fdata()
            dataset[i, j, :, :, :] = img_data[min_x:min_x+patch_x, min_y:min_y+patch_y, min_z:min_z+patch_z] 
            data = dataset[i, j, :, :, :]
            if normalize == True:
                if normalized_method == 'z_score':
                    data[data < 1e-6] = 0
                    mean, std = np.mean(data), np.std(data)
                    dataset[i, j, :, :, :] = (data - mean) / std
                    mean_list[j].append(mean)
                    std_list[j].append(std)
                    print(np.mean(dataset[i, j, :, :, :]), np.std(dataset[i, j, :, :, :]))
                    if i % 50 == 0 and verbose == True:
                        print(data, dataset[i, j, :, :, :], mean, std)
            if i == 0:
                print(filename_full)
            if verbose == True:
                print('No.', i, subject_idx)
    return dataset, mean_list, std_list 

test_dataset, mean_list, std_list = get_test_dataset_from_pair_idx_and_metadata_list(test_data_list, ADNI_path_group_data)
print(test_dataset.shape)
print(mean_list)
print(std_list)
FLAG_save_data_fold_pickle = True
if FLAG_save_data_fold_pickle == True:
    f = open(IPMI_data_pickle_save_path + "mean_std_of_test_dataset.pkl", "wb")
    pkl.dump([mean_list, std_list], f)
    f.close()
    print('Save mean std of test dataset success!') # TODO: modify the log information    

sleep(10000)
information_related_to_test_dataset = [test_dataset, test_data_list, pair_idx, new_metadata_list]

FLAG_save_data_fold_pickle = True
if FLAG_save_data_fold_pickle == True:
    f = open(IPMI_data_pickle_save_path + "information_related_to_test_dataset.pkl", "wb")
    pkl.dump(information_related_to_test_dataset, f)
    f.close()
    print('Save information wrt test dataset success! Len of Dataset is:', information_related_to_test_dataset[0].shape) # TODO: modify the log information    
