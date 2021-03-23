# CopyRight: Zixuan(Zucks) Liu
import argparse
import tqdm
import os
import pickle as pkl
import scipy as sp
import torch
from pprint import pprint

import utils as ut
from torchvision import datasets, transforms
from model import Classifier, CD_Simu, Cycle_Cons, Encoder, Decoder
from torch import  nn, optim
from torch import nn
from torch.nn import functional as F
import numpy as np


cls_list= [0, 1] # 0 - E+HE, 1 - NC
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ut.set_seed(2020)
pathall_saveimg = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_all/save_image/"
IPMI_save = "/scratch/users/zucks626/ADNI/IPMI/checkpoints/"
# ADNI_path_group_data = "/home/groups/kpohl/data/adni/img_64_longitudinal/"
LAB_path_group_data = "/home/groups/kpohl/t1_data/lab_data/img_64_longitudinal/"

# ADNI_data_name_list_txt = "/home/users/zucks626/miccai/adni_list.txt"
LAB_data_name_list_txt = "/home/users/zucks626/miccai/lab_list.txt"

IPMI_data_pickle_save_path = "/scratch/users/zucks626/ADNI/IPMI/data_pickle_saved/"
FLAG_Save_data_pickle = False
DEBUG_VERBOSE = True

# There should be a very first step to generate a Id_attr.pkl for metadata.
# First step: Generate Dataset, a Dict with name subjects clustered, item the image tensor.
FLAG_Save_data_pickle = False
if FLAG_Save_data_pickle == True:
    Dataset = ut.get_Longitudinal_dataset_from_raw_image(LAB_path_group_data)
    f = open(IPMI_data_pickle_save_path + "LAB_Dataset.pkl","wb")
    pkl.dump(Dataset, f)
    f.close()
    print('Save Dataset success! Len of Dataset is:', len(Dataset)) 
else:
    f = open(IPMI_data_pickle_save_path + "LAB_Dataset.pkl","rb")
    Dataset = pkl.load(f)
    f.close() 
    print('Load Dataset success! Len of Dataset is:', len(Dataset))  
    # if DEBUG_VERBOSE == True:
    #     for name, data in Dataset.items():
    #         print(name)


FLAG_Save_data_pickle = False
RETURN_TYPE = 'np'
# label_file_path = IPMI_data_pickle_save_path + 'ADNI1AND2.csv'
label_file_path = IPMI_data_pickle_save_path + 'demographics_lab.csv'
if FLAG_Save_data_pickle == True:
    ut.get_Id_attr_lab_pkl_from_label_file(Dataset, label_file_path, IPMI_data_pickle_save_path) 
else:
    f = open(IPMI_data_pickle_save_path + "Id_attr_lab.pkl", "rb")
    Id_attr = pkl.load(f)
    f.close() 
    print('Load Id_attr_lab.pkl success!') 
    if DEBUG_VERBOSE == True:
        cnt_C = cnt_E = cnt_HE = cnt_UA = 0
        for name, data in Id_attr.items():
            # print(name, data)
            if data == 'C':
                cnt_C += 1
            elif data == 'UA':
                cnt_UA += 1
            elif data == 'E':
                cnt_E += 1
            elif data == 'HE':
                cnt_HE += 1   
        print('C:', cnt_C, 'UA:', cnt_UA, 'E:', cnt_E, 'HE:', cnt_HE, 'E+HE:', cnt_E + cnt_HE)             


FLAG_save_data_fold_pickle = False
if FLAG_save_data_fold_pickle == True:
    dataset, labels, subject_id, num_image_wrt_subject, label_wrt_subject = ut.get_dataset_from_idx_file_and_label_from_Id_attr(LAB_data_name_list_txt, LAB_path_group_data, Id_attr, data_source='lab', mask_dataset=False, verbose=True)
    # if DEBUG_VERBOSE == True:
    #     print(labels)
    #     print(subject_id)
    #     print(num_image_wrt_subject)
    #     print(label_wrt_subject)
    selected_label_list, cls_stats = ut.select_data_given_label_list(labels, label_list=cls_list)
    selected_label_wrt_subject_list, cls_stats_1 = ut.select_data_given_label_list(label_wrt_subject, label_list=cls_list)

    if DEBUG_VERBOSE == True:
        print(selected_label_list, cls_stats)
        print(selected_label_wrt_subject_list, cls_stats_1)

    NUM_FOLD = 5
    fold_list = ut.get_fold_list_from_dataset_and_label(num_image_wrt_subject, label_wrt_subject, cls_stats, label_list=cls_list, num_fold=NUM_FOLD)
    if DEBUG_VERBOSE == True:
        print(list(fold_list))
        print(len(list(fold_list)))

    np.savetxt(IPMI_data_pickle_save_path + 'LAB_CE+HE_fold_list_wrt_subject.txt', fold_list, fmt='%1.1f')
    fold_list_4_image = ut.get_fold_list_4_image_from_wrt_subject(subject_id, fold_list)
    np.savetxt(IPMI_data_pickle_save_path + 'LAB_CE+HE_fold_list_4_image.txt', fold_list_4_image, fmt='%1.1f')
    
    data_fold, label_fold = ut.get_data_fold_given_fold_list_4_image(dataset, labels, fold_list_4_image, label_list=cls_list, fold_num=NUM_FOLD, return_type=RETURN_TYPE)
    if DEBUG_VERBOSE == True:
        for i in range(2):
            for j in range(5):
                print(len(label_fold[i][j]))

FLAG_save_data_fold_pickle = False
if FLAG_save_data_fold_pickle == True:
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_orig_data_fold.pkl", "wb")
    pkl.dump(data_fold, f)
    f.close()
    print('Save Orig Data fold success! Class of Dataset is:', len(data_fold), 'Fold:', len(data_fold[0])) # TODO: modify the log information    
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_orig_label_fold.pkl", "wb")
    pkl.dump(label_fold, f)
    f.close()
    print('Save Orig Label fold success! Class of Label is:', len(label_fold), 'Fold:', len(label_fold[0])) # TODO: modify the log information         
else:
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_orig_data_fold.pkl", "rb")
    data_fold = pkl.load(f)
    f.close()  
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_orig_label_fold.pkl", "rb")
    label_fold = pkl.load(f)
    f.close()  
    if DEBUG_VERBOSE == True:
        for i in range(2):
            for j in range(5):
                print(len(label_fold[i][j]))    
    print('Load Orig data_fold and label_fold success!')

NUM_BALANCE_DATA_NUM = 160
augment_data_fold, augment_label_fold = ut.augment_data_fold(data_fold, label_fold, num_after_augment=NUM_BALANCE_DATA_NUM, return_type=RETURN_TYPE)
print('Generate Augmented data fold and label success!')

FLAG_save_data_fold_pickle = True
if FLAG_save_data_fold_pickle == True:  
    for i in range(2):  
        f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_augment_data_fold_" + str(i) + ".pkl", "wb")
        pkl.dump(augment_data_fold[i], f)
        f.close()
        print('Save Augmented Data fold success! Len of Dataset is:', len(augment_data_fold)) # TODO: modify the log information    
        f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_augment_label_fold_" + str(i) + ".pkl", "wb")
        pkl.dump(augment_label_fold[i], f)
        f.close()
        print('Save Augmented Label fold success! Len of Label is:', len(augment_label_fold))    # TODO: modify the log information         

FLAG_save_data_fold_pickle = True
if FLAG_save_data_fold_pickle == True:  
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_augment_data_fold_0.pkl", "rb")
    augment_data_fold_0 = pkl.load(f)
    f.close()  
    for i in range(5):
        augment_data_fold_0[i] = augment_data_fold_0[i].astype(np.float32)
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_augment_data_fold_0_fp32.pkl", "wb")
    pkl.dump(augment_data_fold_0, f)
    f.close()
    print('Save Augmented data fold 0 in fp32 success!')    # TODO: modify the log information         

    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_augment_data_fold_1.pkl", "rb")
    augment_data_fold_1 = pkl.load(f)
    f.close()  
    for i in range(5):
        augment_data_fold_1[i] = augment_data_fold_1[i].astype(np.float32)
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_augment_data_fold_1_fp32.pkl", "wb")
    pkl.dump(augment_data_fold_1, f)
    f.close()
    print('Save Augmented data fold 1 in fp32 success!')    # TODO: modify the log information

FLAG_save_data_fold_pickle = True
if FLAG_save_data_fold_pickle == True:  
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_orig_data_fold.pkl", "rb")
    data_fold = pkl.load(f)
    f.close()  
    for i in range(2):
        for j in range(5):
            data_fold[i][j] = data_fold[i][j].astype(np.float32)
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_orig_data_fold_fp32.pkl", "wb")
    pkl.dump(data_fold, f)
    f.close()
    print('Save Original data fold in fp32 success!')    # TODO: modify the log information         

