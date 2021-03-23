import argparse
import tqdm
import os
import pickle as pkl
import scipy as sp
import torch
from pprint import pprint

import utils as ut
from torchvision import datasets, transforms
from model import Classifier, CD_Simu, Cycle_Cons, Encoder, Decoder, Discriminator

from unet_model_cond_warp import Cycle_Cons_3_Improvements, Convnet_SkipConnection
from torch import  nn, optim
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

import nibabel as nib


parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=512,    help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=7500, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=50, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=64,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=15,     help="Flag for training")
parser.add_argument('--Type',     type=str, default='ADNI_CLS',     help="Flag for training")
''' Type: Run_cls / ADNI_CLS'''
parser.add_argument('--iter_restart', type=int, default=1400, help="Save model every n iterations")
parser.add_argument('--BATCH_SIZE',     type=int, default=40,     help="Flag for training")
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

Discriminator_layout = [
    ('model={:s}',  'Discriminator'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
Discriminator_model_name = '_'.join([t.format(v) for (t, v) in Discriminator_layout])


cls_list= [0,1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ut.set_seed(2020)


pathall_saveimg = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_all/save_image/"
IPMI_save = "/scratch/users/zucks626/ADNI/IPMI/checkpoints/"
ADNI_path_group_data = "/home/groups/kpohl/data/adni/img_64_longitudinal/"

ADNI_data_name_list_txt = "/home/users/zucks626/miccai/adni_list.txt"

IPMI_data_pickle_save_path = "/scratch/users/zucks626/ADNI/IPMI/data_pickle_saved/"
fold_list_4_image_path = IPMI_data_pickle_save_path + 'ADNI_ADNC_fold_list_4_image.txt'
FLAG_Save_data_pickle = False
DEBUG_VERBOSE = True
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

label_file_path = IPMI_data_pickle_save_path + 'ADNI1AND2.csv'
if FLAG_Save_data_pickle == True:
    ut.get_Id_attr_pkl_from_label_file(Dataset, label_file_path, IPMI_data_pickle_save_path) 
else:
    f = open(IPMI_data_pickle_save_path + "Id_attr.pkl","rb")
    Id_attr = pkl.load(f)
    f.close() 
    print('Load Id_attr.pkl success!')  

# if FLAG_Save_data_pickle == True:
#     pass
# else:
#     f = open(IPMI_data_pickle_save_path + "information_related_to_test_dataset.pkl","rb")
#     information_related_to_test_dataset = pkl.load(f)
#     f.close() 
#     print('Load test visualization dataset success!')  
# test_dataset = information_related_to_test_dataset[0]
# num_of_test_pair = test_dataset.shape[0]
# print('test visualization dataset shape is:', test_dataset.shape) # shape: (BS, 2, 64, 64, 64)
# test_visualization_loader = ut.split_test_dataset(test_dataset, BATCH_SIZE=10)
# dataset, labels, subject_id, num_image_wrt_subject, label_wrt_subject = ut.get_dataset_from_idx_file_and_label_from_Id_attr(ADNI_data_name_list_txt, ADNI_path_group_data, Id_attr)
# selected_label_list, cls_stats = ut.select_data_given_label_list(labels, label_list=[0, 1])
# selected_label_wrt_subject_list, cls_stats_1 = ut.select_data_given_label_list(label_wrt_subject, label_list=[0, 1])
# if DEBUG_VERBOSE == True:
#     print(selected_label_list, cls_stats)
#     print(selected_label_wrt_subject_list, cls_stats_1)
# fold_list = ut.get_fold_list_from_dataset_and_label(num_image_wrt_subject, label_wrt_subject, cls_stats, label_list=[0, 1])
# if DEBUG_VERBOSE == True:
#     print(list(fold_list))
# np.savetxt(IPMI_data_pickle_save_path + 'ADNI_ADNC_fold_list_wrt_subject.txt', fold_list, fmt='%1.1f')
# fold_list_4_image = ut.get_fold_list_4_image_from_wrt_subject(subject_id, fold_list)
# np.savetxt(IPMI_data_pickle_save_path + 'ADNI_ADNC_fold_list_4_image.txt', fold_list_4_image, fmt='%1.1f')
# sleep(10000)
Type = args.Type    
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
    _, labels, _, _, _ = ut.get_dataset_from_idx_file_and_label_from_Id_attr(ADNI_data_name_list_txt, ADNI_path_group_data, Id_attr, normalize=False, mask_dataset=True)
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_orig_data_fold_fp32.pkl", "rb")
    data_fold = pkl.load(f)
    f.close()  
    f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_orig_label_fold.pkl", "rb")
    label_fold = pkl.load(f)
    f.close()  
    file_idx = np.genfromtxt(ADNI_data_name_list_txt, dtype='str') 
    fold_list_4_image = np.genfromtxt(fold_list_4_image_path, dtype='float') 
    # f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_augment_data_fold_0_fp32.pkl", "rb")
    # augment_data_fold_0 = pkl.load(f)
    # f.close()  
    # f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_augment_label_fold_0.pkl", "rb")
    # augment_label_fold_0 = pkl.load(f)
    # f.close()  
    # f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_augment_data_fold_1_fp32.pkl", "rb")
    # augment_data_fold_1 = pkl.load(f)
    # f.close()  
    # f = open(IPMI_data_pickle_save_path + "ADNI_ADNC_augment_label_fold_1.pkl", "rb")
    # augment_label_fold_1 = pkl.load(f)
    # f.close() 
    print('Load data fold and label success!') # TODO: rich log information
    filename_of_data_fold = ut.get_filename_of_data_fold_from_fold_list_4_image(ADNI_data_name_list_txt, fold_list_4_image, labels, num_cls=2, fold_num=5)
    print(filename_of_data_fold)
    if DEBUG_VERBOSE == True:
        for i in range(2):
            for j in range(5):
                print(len(filename_of_data_fold[i][j]))
        # sleep(10000)

def save_data_to_nii(folder_dir, file_name, data, fold, save_type='both'):
    if save_type == 'both':
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        folder_fold = 'fold_' + str(int(fold)) + '_'
        folder_path = folder_dir + folder_fold
        name_prefix = ['NC', 'AD', 'SIMU_AD', 'SIMU_NC', 'CYC_BACK_NC', 'CYC_BACK_AD', 'D_Field_SIMU_AD', 'D_Field_SIMU_NC', 'D_Field_CYC_BACK_NC', 'D_Field_CYC_BACK_AD',]
        xp, xn, s1, s3, s2, s4, v1, v3, v2, v4 = data
        AD_FILE_NAME = file_name[0]
        NC_FILE_NAME = file_name[1]
        num_of_AD = len(AD_FILE_NAME)
        num_of_NC = len(NC_FILE_NAME)
        AD_NAME_TXT = [[] for i in range(len(name_prefix))]
        NC_NAME_TXT = [[] for i in range(len(name_prefix))]
        NAME_TXT = [[] for i in range(len(name_prefix))]
        for name in name_prefix:
            subfolder_dir = folder_path + name
            if not os.path.exists(subfolder_dir):
                os.makedirs(subfolder_dir)
        np.savetxt(folder_dir + 'AD.txt', AD_FILE_NAME, '%s')
        np.savetxt(folder_dir + 'NC.txt', NC_FILE_NAME, '%s')    
        # sleep(1000)
        for i in range(num_of_AD):
            if DEBUG_VERBOSE == True:
                print('AD:', i)
            AD_NAME = str(AD_FILE_NAME[i])
            img = xn[i, 0, :, : ,:]
            nib_img = nib.Nifti1Image(img, np.diag([1, 1, 1, 1]))  
            prefix = name_prefix[1]
            AD_filename = folder_path + prefix + '/' + prefix + '_' + AD_NAME
            NAME_TXT[1].append(prefix + '_' + AD_NAME)
            nib.save(nib_img, AD_filename)

            img = s3[i, 0, :, : ,:]
            nib_img = nib.Nifti1Image(img, np.diag([1, 1, 1, 1]))  
            prefix = name_prefix[3]
            AD_filename = folder_path + prefix + '/' + prefix + '_' + AD_NAME
            NAME_TXT[3].append(prefix + '_' + AD_NAME)
            nib.save(nib_img, AD_filename)

            img = s4[i, 0, :, : ,:]
            nib_img = nib.Nifti1Image(img, np.diag([1, 1, 1, 1]))  
            prefix = name_prefix[5]
            AD_filename = folder_path + prefix + '/' + prefix + '_' + AD_NAME
            NAME_TXT[5].append(prefix + '_' + AD_NAME)
            nib.save(nib_img, AD_filename)

            img = v3[i, :, :, : ,:]
            nib_img = nib.Nifti1Image(img, np.diag([1, 1, 1, 1]))  
            prefix = name_prefix[7]
            AD_filename = folder_path + prefix + '/' + prefix + '_' + AD_NAME
            NAME_TXT[7].append(prefix + '_' + AD_NAME)
            nib.save(nib_img, AD_filename)

            img = v4[i, :, :, : ,:]
            nib_img = nib.Nifti1Image(img, np.diag([1, 1, 1, 1]))  
            prefix = name_prefix[9]
            AD_filename = folder_path + prefix + '/' + prefix + '_' + AD_NAME
            NAME_TXT[9].append(prefix + '_' + AD_NAME)
            nib.save(nib_img, AD_filename)  

        for i in range(num_of_NC): 
            if DEBUG_VERBOSE == True:
                print('NC:', i)
            NC_NAME = NC_FILE_NAME[i]
            img = xp[i, 0, :, : ,:]
            nib_img = nib.Nifti1Image(img, np.diag([1, 1, 1, 1]))  
            prefix = name_prefix[0]
            NC_filename = folder_path + prefix + '/' + prefix + '_' + NC_NAME
            NAME_TXT[0].append(prefix + '_' + NC_NAME)
            nib.save(nib_img, NC_filename)

            img = s1[i, 0, :, : ,:]
            nib_img = nib.Nifti1Image(img, np.diag([1, 1, 1, 1]))  
            prefix = name_prefix[2]
            NC_filename = folder_path + prefix + '/' + prefix + '_' + NC_NAME
            NAME_TXT[2].append(prefix + '_' + NC_NAME)
            nib.save(nib_img, NC_filename)

            img = s2[i, 0, :, : ,:]
            nib_img = nib.Nifti1Image(img, np.diag([1, 1, 1, 1]))  
            prefix = name_prefix[4]
            NC_filename = folder_path + prefix + '/' + prefix + '_' + NC_NAME
            NAME_TXT[4].append(prefix + '_' + NC_NAME)
            nib.save(nib_img, NC_filename)

            img = v1[i, :, :, : ,:]
            nib_img = nib.Nifti1Image(img, np.diag([1, 1, 1, 1]))  
            prefix = name_prefix[6]
            NC_filename = folder_path + prefix + '/' + prefix + '_' + NC_NAME
            NAME_TXT[6].append(prefix + '_' + NC_NAME)
            nib.save(nib_img, NC_filename)

            img = v2[i, :, :, : ,:]
            nib_img = nib.Nifti1Image(img, np.diag([1, 1, 1, 1]))  
            prefix = name_prefix[8]
            NC_filename = folder_path + prefix + '/' + prefix + '_' + NC_NAME
            NAME_TXT[7].append(prefix + '_' + NC_NAME)
            nib.save(nib_img, NC_filename)      

        for i, name in enumerate(name_prefix):
            subfolder_dir = folder_path + name + '/'
            subtxt_filename = subfolder_dir + name + '_file.txt' 
            np.savetxt(subtxt_filename, NAME_TXT[i], '%s') 

def simulate_img_fold(cyc_con, data_fold, label_fold, filename_of_data_fold, data_save_path, eval_fold_list=[0,1,2,3,4], requires_grad=False, mode='eval', verbose=True):
    num_cls = len(data_fold)
    num_fold = len(data_fold[0])
    if DEBUG_VERBOSE == True:
        print(cyc_con.device)
    for fold in eval_fold_list:
        x_AD = data_fold[0][fold]
        x_NC = data_fold[1][fold]
        l_AD = label_fold[0][fold]
        l_NC = label_fold[1][fold]
        f_AD = filename_of_data_fold[0][fold]
        f_NC = filename_of_data_fold[1][fold]
        f_name = [f_AD, f_NC]
        data = []
        num_of_AD = len(x_AD)
        num_of_NC = len(x_NC)
        folder_path = data_save_path
        if verbose == True:
            print('Fold:',fold, ' AD:', num_of_AD)
            print('Fold:',fold, ' NC:', num_of_NC)
            print('data_save_path_fold:', folder_path)
        xp = torch.tensor(x_NC).to(cyc_con.device)
        xn = torch.tensor(x_AD).to(cyc_con.device)
        if verbose == True:
            print('NC:', xp.shape)
            print('AD:', xn.shape)
        option_simu_AD = torch.ones(num_of_NC, 1).to(cyc_con.device) * -1
        option_simu_NC = torch.ones(num_of_AD, 1).to(cyc_con.device) * 1
        option_cyc_back_AD = torch.ones(num_of_AD, 1).to(cyc_con.device) * -1
        option_cyc_back_NC = torch.ones(num_of_NC, 1).to(cyc_con.device) * 1        
        v1, s1 = cyc_con.Generate(xp, option_simu_AD, mode=mode, requires_grad=requires_grad)
        v3, s3 = cyc_con.Generate(xn, option_simu_NC, mode=mode, requires_grad=requires_grad)
        v2, s2 = cyc_con.Generate(s1, option_cyc_back_NC, mode=mode, requires_grad=requires_grad)
        v4, s4 = cyc_con.Generate(s3, option_cyc_back_AD, mode=mode, requires_grad=requires_grad)
        xp = xp.cpu().numpy()
        xn = xn.cpu().numpy()
        v1 = v1.detach().cpu().numpy()
        v2 = v2.detach().cpu().numpy()
        v3 = v3.detach().cpu().numpy()
        v4 = v4.detach().cpu().numpy()
        s1 = s1.detach().cpu().numpy()
        s2 = s2.detach().cpu().numpy()
        s3 = s3.detach().cpu().numpy()
        s4 = s4.detach().cpu().numpy()
        data = [xp, xn, s1, s3, s2, s4, v1, v3, v2, v4]
        if DEBUG_VERBOSE == True:
            print('v1:', v1.shape)
            print('v3:', v3.shape)
        # sleep(1000)
        save_data_to_nii(folder_path, f_name, data, fold, save_type='both')

def main():
    cls = Classifier(name=Classifier_model_name, device=device)

    cls_global_step = 400
    ut.load_model_by_name(cls, global_step=cls_global_step, device=device)
    print('Load model!')
    # print(list(cls.named_modules()))
    # sleep(1000)
    FLAG_COND_TYPE = 'Cond'
    cyc_con_global_step = 950
    trans = Convnet_SkipConnection(in_ch=1, out_ch=3, name='trans', device=device, z_dim=args.z, conv_type=FLAG_COND_TYPE)
    cyc_con = Cycle_Cons_3_Improvements(cls, trans, name=cyc_con_model_name, device=device)

    ut.load_model_by_name(cyc_con, global_step=cyc_con_global_step, device=device)

    print('Load Cyc_con model!')
    eval_fold_list = [4]
    
    data_save_path = IPMI_data_pickle_save_path + 'generate_test_result/' + str(args.run) + '_' + str(cyc_con_global_step) + '/'
    simulate_img_fold(cyc_con, data_fold, label_fold, filename_of_data_fold, data_save_path, eval_fold_list=eval_fold_list, requires_grad=False, mode='eval', verbose=True)


main()