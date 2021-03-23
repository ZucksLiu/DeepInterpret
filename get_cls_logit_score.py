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
from unet_model import Cycle_Cons_3_Improvements, Convnet_SkipConnection
from torch import  nn, optim
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score


parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=512,    help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=7500, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=50, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=96,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=15,     help="Flag for training")
parser.add_argument('--Type',     type=str, default='ADNI_CLS',     help="Flag for training")
''' Type: Run_cls / ADNI_CLS'''
parser.add_argument('--iter_restart', type=int, default=1400, help="Save model every n iterations")
parser.add_argument('--BATCH_SIZE',     type=int, default=48,     help="Flag for training")
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


pathall_saveimg = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_all/save_image/"
IPMI_save = "/scratch/users/zucks626/ADNI/IPMI/checkpoints/"
ADNI_path_group_data = "/home/groups/kpohl/data/adni/img_64_longitudinal/"

ADNI_data_name_list_txt = "/home/users/zucks626/miccai/adni_list.txt"

IPMI_data_pickle_save_path = "/scratch/users/zucks626/ADNI/IPMI/data_pickle_saved/"
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

if FLAG_Save_data_pickle == True:
    pass
else:
    f = open(IPMI_data_pickle_save_path + "information_related_to_test_dataset.pkl","rb")
    information_related_to_test_dataset = pkl.load(f)
    f.close() 
    print('Load test visualization dataset success!')  
test_dataset = information_related_to_test_dataset[0]
print('test visualization dataset shape is:', test_dataset.shape) # shape: (BS, 2, 64, 64, 64)
test_visualization_loader = ut.split_test_dataset(test_dataset, BATCH_SIZE=10)

if Type == 'Run_cls' and args.train > 0:
    ''' Extract data from pickle file'''
    # f = open(pathall_saveimg + "augment_pair_cls_AD.pkl", "rb")
    # pair = pkl.load(f)
    # f.close()
    # f = open(pathall_saveimg + "augment_d_cls_AD.pkl", "rb")
    f = open(pathall_saveimg + "dataset_pair_realone_adni.pkl", "rb")
    dataset = pkl.load(f)
    f.close()    
    print(dataset.shape)
    sleep(1000)
    f = open(pathall_saveimg + "augment_label_cls_AD.pkl", "rb")
    label = pkl.load(f)
    f.close()  
    id_idx, cal_idx = ut.get_idx_label(label)
    pair_new, label_new = ut.get_pair_idx_label_new(id_idx, pair, cls_list)
    print(pair_new)
    print(label_new)
    print(len(pair_new))
    train_loader_list, train_label_loader_list = ut.split_dataset_folds_new_true_subject(dataset, label_new, pair_new, folds=5, BATCH_SIZE = args.BATCH_SIZE, shuffle=True, seed=2020)
elif Type == 'ADNI_CLS':
    FLAG_TEST_FOLD = 4
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_orig_data_fold_fp32.pkl", "rb")
    data_fold = pkl.load(f)
    f.close()  
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_orig_label_fold.pkl", "rb")
    label_fold = pkl.load(f)
    f.close()  
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_augment_data_fold_0_fp32.pkl", "rb")
    augment_data_fold_0 = pkl.load(f)
    f.close()  
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_augment_label_fold_0.pkl", "rb")
    augment_label_fold_0 = pkl.load(f)
    f.close()  
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_augment_data_fold_1_fp32.pkl", "rb")
    augment_data_fold_1 = pkl.load(f)
    f.close()  
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_augment_label_fold_1.pkl", "rb")
    augment_label_fold_1 = pkl.load(f)
    f.close() 
    print('Load data fold and label success!') # TODO: rich log information
    test_data_fold, test_label_fold = ut.merge_data_fold_wrt_label(data_fold, label_fold)
    test_loader_list, test_label_loader_list = ut.split_data_fold(test_data_fold, test_label_fold, num_fold=5, BATCH_SIZE=args.BATCH_SIZE)
    print('Get test data loader success!')
    # augment_data_fold = [augment_data_fold_0, augment_data_fold_1]
    # augment_label_fold = [augment_label_fold_0, augment_label_fold_1]
    # data_fold, label_fold = ut.merge_data_fold_with_augmented_data(data_fold, label_fold, augment_data_fold, augment_label_fold)
    # for i in range(2):
    #     for j in range(5):
    #         print(data_fold[i][j].shape, label_fold[i][j].shape)
    data_fold, label_fold = ut.merge_data_fold_wrt_label(data_fold, label_fold)
    flipped = False
    if flipped == True:
        data_fold, label_fold = ut.flipped_data_fold(data_fold, label_fold)
    # for i in range(5):
    #     print(data_fold[i].shape, label_fold[i].shape)
    train_loader_list, train_label_loader_list = ut.split_data_fold(data_fold, label_fold, num_fold=5, BATCH_SIZE=args.BATCH_SIZE)
    # sleep(10000)
