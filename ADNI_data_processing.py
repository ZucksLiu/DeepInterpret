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


cls_list= [0, 1] # 0 - AD, 1 - NC, 2 - MCI
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ut.set_seed(2020)
IPMI_save = "/ADNI/IPMI/checkpoints/"
ADNI_path_group_data = "/data/adni/img_64_longitudinal/"

ADNI_data_name_list_txt = "adni_list.txt"

IPMI_data_pickle_save_path = "/ADNI/IPMI/data_pickle_saved/"
FLAG_Save_data_pickle = False
DEBUG_VERBOSE = False

# if FLAG_Save_data_pickle == True:
#     Dataset = ut.get_Longitudinal_dataset_from_raw_image(ADNI_path_group_data)
#     f = open(IPMI_data_pickle_save_path + "ADNI_Dataset.pkl","wb")
#     pkl.dump(Dataset, f)
#     f.close()
#     print('Save Dataset success! Len of Dataset is:', len(Dataset)) 
# else:
#     f = open(IPMI_data_pickle_save_path + "ADNI_Dataset.pkl","rb")
#     Dataset = pkl.load(f)
#     f.close() 
#     print('Load Dataset success! Len of Dataset is:', len(Dataset))  
RETURN_TYPE = 'np'
label_file_path = IPMI_data_pickle_save_path + 'ADNI1AND2.csv'
if FLAG_Save_data_pickle == True:
    ut.get_Id_attr_pkl_from_label_file(Dataset, label_file_path, IPMI_data_pickle_save_path) 
else:
    f = open(IPMI_data_pickle_save_path + "Id_attr.pkl", "rb")
    Id_attr = pkl.load(f)
    f.close() 
    print('Load Id_attr.pkl success!')  
FLAG_save_data_fold_pickle = True
if FLAG_save_data_fold_pickle == True:
    dataset, labels, subject_id, num_image_wrt_subject, label_wrt_subject = ut.get_dataset_from_idx_file_and_label_from_Id_attr(ADNI_data_name_list_txt, ADNI_path_group_data, Id_attr)
    selected_label_list, cls_stats = ut.select_data_given_label_list(labels, label_list=cls_list)
    selected_label_wrt_subject_list, cls_stats_1 = ut.select_data_given_label_list(label_wrt_subject, label_list=cls_list)

    if DEBUG_VERBOSE == True:
        print(selected_label_list, cls_stats)
        print(selected_label_wrt_subject_list, cls_stats_1)

    NUM_FOLD = 5
    fold_list = ut.get_fold_list_from_dataset_and_label(num_image_wrt_subject, label_wrt_subject, cls_stats, label_list=cls_list, num_fold=NUM_FOLD)
    if DEBUG_VERBOSE == True:
        print(list(fold_list))

    np.savetxt(IPMI_data_pickle_save_path + 'ADNI_ADNC_fold_list_wrt_subject.txt', fold_list, fmt='%1.1f')
    fold_list_4_image = ut.get_fold_list_4_image_from_wrt_subject(subject_id, fold_list)
    np.savetxt(IPMI_data_pickle_save_path + 'ADNI_ADNC_fold_list_4_image.txt', fold_list_4_image, fmt='%1.1f')

    
    data_fold, label_fold = ut.get_data_fold_given_fold_list_4_image(dataset, labels, fold_list_4_image, label_list=cls_list, fold_num=NUM_FOLD, return_type=RETURN_TYPE)

FLAG_save_data_fold_pickle = True
if FLAG_save_data_fold_pickle == True:
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_orig_data_fold.pkl", "wb")
    pkl.dump(data_fold, f)
    f.close()
    print('Save Orig Data fold success! Len of Dataset is:', len(data_fold)) # TODO: modify the log information    
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_orig_label_fold.pkl", "wb")
    pkl.dump(label_fold, f)
    f.close()
    print('Save Orig Label fold success! Len of Label is:', len(label_fold))    # TODO: modify the log information         
# else:
#     f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_orig_data_fold.pkl", "rb")
#     data_fold = pkl.load(f)
#     f.close()  
#     f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_orig_label_fold.pkl", "rb")
#     label_fold = pkl.load(f)
#     f.close()  
#     print('Load Orig data_fold and label_fold success!')
NUM_BALANCE_DATA_NUM = 1024
augment_data_fold, augment_label_fold = ut.augment_data_fold(data_fold, label_fold, num_after_augment=NUM_BALANCE_DATA_NUM, return_type=RETURN_TYPE)
print('Generate Augmented data fold and label success!')
FLAG_save_data_fold_pickle = True
if FLAG_save_data_fold_pickle == True:  
    for i in range(2):  
        f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_augment_data_fold_" + str(i) + ".pkl", "wb")
        pkl.dump(augment_data_fold[i], f)
        f.close()
        print('Save Augmented Data fold success! Len of Dataset is:', len(augment_data_fold)) # TODO: modify the log information    
        f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_augment_label_fold_" + str(i) + ".pkl", "wb")
        pkl.dump(augment_label_fold[i], f)
        f.close()
        print('Save Augmented Label fold success! Len of Label is:', len(augment_label_fold))    # TODO: modify the log information         

FLAG_save_data_fold_pickle = True
if FLAG_save_data_fold_pickle == True:  
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_augment_data_fold_0.pkl", "rb")
    augment_data_fold_0 = pkl.load(f)
    f.close()  
    for i in range(5):
        augment_data_fold_0[i] = augment_data_fold_0[i].astype(np.float32)
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_augment_data_fold_0_fp32.pkl", "wb")
    pkl.dump(augment_data_fold_0, f)
    f.close()
    print('Save Augmented data fold 0 in fp32 success!')    # TODO: modify the log information         

    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_augment_data_fold_1.pkl", "rb")
    augment_data_fold_1 = pkl.load(f)
    f.close()  
    for i in range(5):
        augment_data_fold_1[i] = augment_data_fold_1[i].astype(np.float32)
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_augment_data_fold_1_fp32.pkl", "wb")
    pkl.dump(augment_data_fold_1, f)
    f.close()
    print('Save Augmented data fold 1 in fp32 success!')    # TODO: modify the log information

FLAG_save_data_fold_pickle = True
if FLAG_save_data_fold_pickle == True:  
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_orig_data_fold.pkl", "rb")
    data_fold = pkl.load(f)
    f.close()  
    for i in range(2):
        for j in range(5):
            data_fold[i][j] = data_fold[i][j].astype(np.float32)
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_orig_data_fold_fp32.pkl", "wb")
    pkl.dump(data_fold, f)
    f.close()
    print('Save Original data fold in fp32 success!')    # TODO: modify the log information         

