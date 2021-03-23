# utils.py
import numpy as np
import os
import shutil
import sys
import torch
import random
import torch.utils.data as Data
import math
import scipy as sp
from nilearn import image
import nibabel as nib
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import gc
import pandas as pd
import pickle as pkl
# from codebase.models.vae3d import VAE3d

bce = torch.nn.BCEWithLogitsLoss(reduction='none')


def set_seed(x=2020):
	random.seed(x)


''' Save & Load Functions'''
def load_model_by_name(model, global_step, device=None, path="/scratch/users/zucks626/ADNI/IPMI/checkpoints/"):
    """
    Load a model based on its name model.name and the checkpoint iteration step

    Args:
        model: Model: (): A model
        global_step: int: (): Checkpoint iteration
    """
    # path = "/scratch/users/zucks626/ADNI/ae_cls/checkpoints/"
    file_path = path + model.name + "/" + 'model-{:05d}.pt'.format(global_step)
    state = torch.load(file_path, map_location=device)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))

def save_loss_image(losses,path,name):
    length = len(losses)
    plt.cla()
    plt.plot(range(length), losses)
    plt.savefig(path + name)

def sample_image(model, batch, mean, std, path, folder_name):
    x = model.sample_x(batch)
    save_image(x, mean, std, path, folder_name)
    
def save_image(x, mean, std, path, folder_name):
    x = x.squeeze(dim=1)
    x = x * std + mean
    import nibabel as nib
    for i in range(0,x.shape[0]):
        img = nib.Nifti1Image(x[i,:].detach().cpu().clone().numpy(), np.eye(4))
        img.to_filename(path + folder_name+"_"+str(i))

def save_model_by_name_cls(model, global_step, path="/scratch/users/zucks626/ADNI/IPMI/checkpoints/"):
    # path = "/scratch/users/zucks626/ADNI/ae_cls/checkpoints/"
    save_dir = path + model.name + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))

def save_image_pickel_by_name_cls(image_pickle, name, global_step, path="/scratch/users/zucks626/ADNI/IPMI/checkpoints/"):
    # path = "/scratch/users/zucks626/ADNI/ae_cls/checkpoints/"
    save_dir = path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, name + '-{:05d}.pkl'.format(global_step))
    with open(file_path, "wb") as fp:
        pkl.dump(image_pickle, fp)
    print('Saved to {}'.format(file_path))


''' Writing Logs'''
def writelog(filename, step, acc_list, fold_idx_list ,tot_list,total, totlen, test_acc, test_tot, test_idx):
    f = open(filename,'a+')
    line = 'Train step:' +str(step)+ ' Total:' + str(totlen) + ' Acc:' + str(total/totlen) +'\n'
    f.write(line)
    for i in range(len(acc_list)):
        line = 'Batch:'+ str(fold_idx_list[i]) +' Total:' + str(tot_list) + ' Acc:' + str(acc_list[i]) +'\n'
        f.write(line)
    line = 'Test batch:'+ str(test_idx) +' Total:' + str(test_tot) + ' Acc:' + str(test_acc)+'\n\n'
    f.write(line)
    f.close() 
    
def writeacc(filename, step, acc_list, tot_acc, test_acc, fold_idx_list, test_idx):
    f = open(filename,'a+')
    line = str(step) + ' '
    line1 = str(step) + ' '
    for i in range(len(acc_list)):
        line += str(fold_idx_list[i])+' '
        line1 += str(acc_list[i])+' '
    line += 'T ' + str(test_idx) + '\n'
    line1 += str(tot_acc) + ' ' + str(test_acc) + '\n'
    f.write(line)
    f.write(line1)
    f.close()

#TODO: 写分fold 拿label，from both Dataset and small dataset.
''' Get Dataset '''
def get_fileid_lookup_table(path):
    # get Dict with k-v pair (prefix_id, file_name(time ordered) )
    # used in get_Longitudinal_Dataset_from_raw_image(path), to generate prefix_id list w.r.t time
	files = os.listdir(path)
	print("file stored in path=", path)
	lookup_table = {}
	for file in files:
		fileid = file[0:11]
		day = file[11:]
		if lookup_table.get(fileid,-1) == -1:
			lookup_table[fileid] = [day]
		else:
			i = 0
			while i <len(lookup_table[fileid]):
				if day > lookup_table[fileid][i]:
					i += 1
				else:
					break
			lookup_table[fileid].insert(i,day)
	return lookup_table

def get_Longitudinal_dataset_from_raw_image(path):
    # return a Dict with k-v pair: (prefix_id, D)
    # D is a torch.tensor with shape (n_i, 1, 64, 64, 64), n_i is num of data sample
    # from this patient.
    # D is time-ordered.

    lookup_table = get_fileid_lookup_table(path)
    Dataset = {}

    for file_prefixid, file_name_list in lookup_table.items():
        if file_prefixid == 'failed_proc':
            continue
        for i in range(len(file_name_list)):
            filename = file_name_list[i]
            filename_full = path + file_prefixid + filename
            img = nib.load(filename_full)
            img_data = img.get_fdata()
            data = np.copy(img_data)
            # file = file_name_list[i]
            # img = image.get_data(path + file_prefixid + file)
            img1 = torch.tensor(data.reshape(1, 1, 64, 64, 64).astype('float32'))
            # print(img1.dtype)
            # img._mmap.close()
            if i == 0:
                Dataset[file_prefixid] = img1
            else:
                Dataset[file_prefixid] =torch.cat((Dataset[file_prefixid], img1),axis=0)
    return Dataset


def get_Id_attr_lab_pkl_from_label_file(Dataset, label_file_path, save_pkl_path):
    obj = pd.read_csv(label_file_path, ",", header=None)
    obj.columns = ['Subject_ID','DX']

    Subject_ID = {}
    i=0
    for idx in obj['Subject_ID'].values:
            dx = obj['DX'].values[i]
            label = map_lab_label(dx)
            Subject_ID[idx] = label
            i += 1
    print('We go over ', i, ' and get', len(Subject_ID),' subjects matadata!')
    f = open(save_pkl_path + "Id_attr_lab.pkl","wb")
    pkl.dump(Subject_ID, f)
    f.close()
    print('Save Id_attr_lab.pkl success!')



def get_Id_attr_pkl_from_label_file(Dataset, label_file_path, save_pkl_path):
    obj = pd.read_csv(label_file_path, ",", header=None)
    print(obj)
    obj.columns = ['Subject_ID','Sex','DX_GROUP','Age']
    Subject_ID = {}
    for id, data in Dataset.items():
        S_id = id[:-1]
        Subject_ID[S_id] = id

    Sid_attr = {}
    i = 0
    for idx, id_D in Subject_ID.items():
        if idx in obj['Subject_ID'].values:
            i += 1
            Sid_attr[idx] = {}
            index = obj[obj.Subject_ID==idx].index[0]
            Sex = obj['Sex'][index]
            Age = float(obj['Age'][index])
            Group = obj['DX_GROUP'][index]
            Sid_attr[idx]['Sex'] = Sex
            Sid_attr[idx]['Age'] = Age
            Sid_attr[idx]['Group'] = Group
    print('We get ', i, 'subjects matadata!')

    f = open(save_pkl_path + "Id_attr.pkl", "wb")
    pkl.dump(Sid_attr, f)
    f.close()    
    print('Save Id_attr.pkl success!')

def get_dataset_from_idx_file_adni(idx_file_path_1, idx_file_path_2, data_path, remove_file_list=[], data='lab',normalize=True, normalized_method = 'z_score'):
    print("idx_file path=", idx_file_path_1)
    print("data path=", data_path)
    print(remove_file_list)
    dataset = []
    file_idx_1 = np.genfromtxt(idx_file_path_1, dtype='str') 
    file_idx_2 = np.genfromtxt(idx_file_path_2, dtype='str') 
    subject_num = file_idx_1.shape[0]
    patch_x = patch_y = patch_z = 64
    min_x =min_y = min_z=0
    dataset = np.zeros((subject_num, 2, patch_x, patch_y, patch_z))
    # i=0
    # while i< subject_num:
    j=0
    for i in range(len(file_idx_1)):
        subject_idx_1 = file_idx_1[i]
        subject_idx_2 = file_idx_2[i]
        filename_full_1 = data_path + subject_idx_1
        filename_full_2 = data_path + subject_idx_2
        if subject_idx_1 in remove_file_list or subject_idx_2 in remove_file_list:
            print(subject_idx_1)
            continue
        img = nib.load(filename_full_1)
        img_data = img.get_fdata()

        dataset[j,0,:,:,:] = img_data[min_x:min_x+patch_x, min_y:min_y+patch_y, min_z:min_z+patch_z] 
        data = dataset[j,0,:,:,:]
        # print(data[data<-1])
        # print(len(data[data<-1]))
        data[data<1e-6] = 0
        data[data>1e-6] = (data[data>1e-6]-np.mean(data[data>1e-6]))/np.std(data[data>1e-6])   
        # print(np.mean(dataset[j,0,:,:,:]))
        # print(np.std(dataset[j,0,:,:,:]))
        # dataset[j,0,:,:,:] = (dataset[j,0,:,:,:] - np.mean(dataset[j,0,:,:,:])) / np.std(dataset[j,0,:,:,:]) 

        img = nib.load(filename_full_2)
        img_data = img.get_fdata()
        dataset[j,1,:,:,:] = img_data[min_x:min_x+patch_x, min_y:min_y+patch_y, min_z:min_z+patch_z] 
        data = dataset[j,1,:,:,:]
        # print(data[data<-1])
        # print(len(data[data<-1])) 
        data[data<1e-6] = 0
        data[data>1e-6] = (data[data>1e-6]-np.mean(data[data>1e-6]))/np.std(data[data>1e-6])   
        # print(np.mean(dataset[j,1,:,:,:]))
        # print(np.std(dataset[j,1,:,:,:]))        
        # dataset[j,1,:,:,:] = (dataset[j,1,:,:,:] - np.mean(dataset[j,1,:,:,:])) / np.std(dataset[j,1,:,:,:])   
        j+=1     
        if i == 0:
            print(filename_full_1)
    print(j)
    print(subject_num)
    return dataset[:j,:]

def get_dataset_from_idx_file(idx_file_path, data_path, normalize=True, normalized_method='z_score'):
    print("idx_file path=", idx_file_path)
    print("data path=", data_path)
    dataset = []
    file_idx = np.genfromtxt(idx_file_path, dtype='str') 
    subject_num = file_idx.shape[0]
    patch_x = patch_y = patch_z = 64
    min_x = min_y = min_z = 0
    dataset = np.zeros((subject_num, 1, patch_x, patch_y, patch_z))
    i = 0
    for subject_idx in file_idx:
        filename_full = data_path + subject_idx

        img = nib.load(filename_full)
        img_data = img.get_fdata()
        dataset[i, 0, :, :, :] = img_data[min_x:min_x+patch_x, min_y:min_y+patch_y, min_z:min_z+patch_z] 
        data = dataset[i, 0, :, :, :]
        # print(data[data<-1])
        # print(len(data[data<-1]))
        if normalize == True:
            if normalized_method == 'z_score':
                data[data < 1e-6] = 0
                data[data > 1e-6] = (data[data > 1e-6] - np.mean(data[data > 1e-6])) / np.std(data[data > 1e-6])   
        # dataset[i,0,:,:,:] = (dataset[i,0,:,:,:] - np.mean(dataset[i,0,:,:,:])) / np.std(dataset[i,0,:,:,:])        
        if i == 0:
            print(filename_full)
        i += 1
    return dataset

def get_dataset_from_idx_file_and_label_from_Id_attr(idx_file_path, data_path, Id_attr, normalize=True, data_source='adni', label_num=2, normalized_method='z_score', verbose=True, mask_dataset=False):
    # Output: 
    #    dataset - a numpy tensor with shape (num_image, 1, 64, 64, 64)
    #    label - a numpy list recording the label of each image
    #    subject_id - a numpy list recording the subject_id of this image
    print("idx_file_path=", idx_file_path)
    print("data_path=", data_path)
    # dataset = []
    file_idx = np.genfromtxt(idx_file_path, dtype='str') 
    subject_num = file_idx.shape[0]
    print("the num of image is:", subject_num)
    patch_x = patch_y = patch_z = 64
    min_x = min_y = min_z = 0
    dataset = np.zeros((subject_num, 1, patch_x, patch_y, patch_z), dtype=np.float32)
    labels = np.zeros(subject_num)
    subject_id = np.zeros(subject_num)
    i = 0
    num_image_wrt_subject = [0]
    label_wrt_subject = [-1]
    current_subject_id = 0
    last_subject_name = file_idx[0][:10]
    if data_source == 'adni':
        label2num = get_label2num(num_cls=3)
    elif data_source == 'lab':
        label2num = get_label2num_lab(num_cls=label_num)
        # label2num_2 = get_label2num_lab(num_cls=2)
    for subject_idx in file_idx:
        filename_full = data_path + subject_idx
        subject_name = subject_idx[:10]
        if last_subject_name != subject_name:
            num_image_wrt_subject.append(0)
            label_wrt_subject.append(-1)
            current_subject_id += 1
            last_subject_name = subject_name
            
        subject_id[i] = current_subject_id
        if subject_name in Id_attr:
            if data_source == 'adni':
                labels[i] = label2num[Id_attr[subject_name]['Group']]
            elif data_source == 'lab':
                labels[i] = label2num[Id_attr[subject_name]]
            
        ''' '''
        if mask_dataset == False:
            img = nib.load(filename_full)
            img_data = img.get_fdata()
            dataset[i, 0, :, :, :] = img_data[min_x:min_x+patch_x, min_y:min_y+patch_y, min_z:min_z+patch_z] 
            data = dataset[i, 0, :, :, :]
        # print(data[data<-1])
        # print(len(data[data<-1]))
        if mask_dataset == False and normalize == True:
            if normalized_method == 'z_score':
                data[data < 1e-6] = 0
                mean, std = np.mean(data), np.std(data)
                dataset[i, 0, :, :, :] = (data - mean) / std
                print(np.mean(dataset[i,0,:,:,:]), np.std(dataset[i,0,:,:,:]))
                if i % 1000 == 0 and verbose == True:
                    print(data, dataset[i, 0, :, :, :], mean, std)
                # data[data > 1e-6] = (data[data > 1e-6] - np.mean(data[data > 1e-6])) / np.std(data[data > 1e-6])   
        # dataset[i,0,:,:,:] = (dataset[i,0,:,:,:] - np.mean(dataset[i,0,:,:,:])) / np.std(dataset[i,0,:,:,:])        
        
        label_wrt_subject[current_subject_id] = labels[i]
        num_image_wrt_subject[current_subject_id] += 1
        
        if i == 0:
            print(filename_full)
        if verbose == True:
            print('No.', i, subject_idx)
        i += 1
    # sleep(1000)
    return dataset, labels, subject_id, num_image_wrt_subject, label_wrt_subject


def select_data_given_label_list(labels, label_list=[0, 1], statistic=True):
    subject_num = len(labels)
    cls_stats = np.zeros(len(label_list))
    selected_label_list = [[] for i in range(len(label_list))]
    for i in range(subject_num):
        label = labels[i]
        if label not in label_list:
            continue
        idx_in_label_list = np.argwhere(label_list==label)[0][0]
        selected_label_list[idx_in_label_list].append(i)
        cls_stats[idx_in_label_list] += 1 
    
    if statistic == True:
        return selected_label_list, np.asarray(cls_stats)
    else:
        return selected_label_list, None

def get_fold_list_from_dataset_and_label(num_image_wrt_subject, label_wrt_subject, cls_stats, label_list=[0, 1], num_fold=5, shuffle=True, seed=2020):
    sum_of_all_image = sum(cls_stats)
    ratio_of_each_cls = cls_stats / sum_of_all_image
    threshold_of_each_fold = cls_stats // num_fold
    
    num_subject = len(num_image_wrt_subject)  
    num_in_each_fold = np.zeros((len(label_list), num_fold))
    fold_list = np.ones(num_subject) * -1  
    idx = [i for i in range(0, num_subject)]
    if shuffle == True:
        random.seed(seed)
        random.shuffle(idx)
    fold_iter = np.zeros(len(label_list), dtype=int)
    for i in idx:
        label = label_wrt_subject[i]
        num_image = num_image_wrt_subject[i]
        if label not in label_list:
            continue        
        idx_in_label_list = np.argwhere(label_list==label)[0][0]
        fold_list[i] = fold_iter[idx_in_label_list]
        num_in_each_fold[idx_in_label_list, fold_iter[idx_in_label_list]] += num_image
        if num_in_each_fold[idx_in_label_list, fold_iter[idx_in_label_list]] >= threshold_of_each_fold[idx_in_label_list]:
            fold_iter[idx_in_label_list] += 1   
    return fold_list
    
def get_fold_list_4_image_from_wrt_subject(subject_id, fold_list):
    num_of_image = len(subject_id)
    num_of_subject = len(fold_list)
    fold_list_4_image = np.ones(num_of_image)
    for i in range(num_of_image):
        sub_id = int(subject_id[i])
        fold_list_4_image[i] = fold_list[sub_id]
    return fold_list_4_image

def split_dataset_folds_new_true_subject(dataset, label, pair, folds=5, split_ratio=0.1, shuffle=True, seed=2020, BATCH_SIZE=10):
    num = len(pair)
    len_d = dataset.shape[0] 
    # ratio = int(num * split_ratio)
    ratio = int(num // folds)
    ratio_d = int(len_d // folds)
    print('ratio:',ratio)
    print('ratio_d:',ratio_d)
    idx = [i for i in range(0,num)]
    if shuffle == True:
        random.seed(seed)
        random.shuffle(idx)
    Subject_to_folds = {}
    new_idx = [[] for i in range(folds)]
    cnt =0
    for i in range(num):
        idx_subject = idx[i]
        p = pair[idx_subject]
        s = p[2]
        e = p[3]
        l = e-s
        Length_of_folds = [0]*5
        
        if Subject_to_folds.get(p[1],-1) == -1:
            
            potential_idx = random.randint(0,folds-1) 
            
            while Length_of_folds[potential_idx] +l >= ratio_d + 1:
                potential_idx = random.randint(0,folds-1)
            Subject_to_folds[p[1]] = potential_idx
            new_idx[potential_idx].append(idx_subject)
            Length_of_folds[potential_idx] += l

        else:
            cnt += 1
            potential_idx = Subject_to_folds[p[1]] 
            new_idx[potential_idx].append(idx_subject)
            Length_of_folds[potential_idx] += l
    print(cnt)
    len_pair_of_folds = [len(new_idx[i]) for i in range(folds)]
    idx_of_pair_folds = np.cumsum([0]+len_pair_of_folds) 
    new_new_idx = []
    for i in range(folds):
        new_new_idx = new_new_idx + new_idx[i]
    idx = new_new_idx

    data_fold_list = []
    label_fold_list = []
    for i in range(folds):

        data = []
        nl = []
        st = idx_of_pair_folds[i]
        ed = idx_of_pair_folds[i+1]            
        for j in range(st,ed):
                index = idx[j]
                p = pair[idx[j]]
                s = p[2]
                e = p[3]

                data.append(dataset[s:e])
                for k in range(s,e):
                    nl.append(label[k])
        idxs = list(range(len(nl)))
        if shuffle == True:
            random.seed(seed)
            random.shuffle(idxs)

        data = torch.cat(data)[idxs]
        nl =torch.tensor(nl)[idxs]
        print(data.shape)
        print(nl,sum(nl.float())/len(nl))
        train_set = Data.TensorDataset(data)
        label_set = Data.TensorDataset(nl)
        # else:
        #     train_set = Data.TensorDataset(dataset[idx[i*ratio:],:])
        #     label_set = Data.TensorDataset(label[idx[i*ratio:]])           
        train_loader = Data.DataLoader(train_set,batch_size =BATCH_SIZE)
        data_fold_list.append(train_loader)
    	
        label_loader = Data.DataLoader(label_set,batch_size =BATCH_SIZE)
        label_fold_list.append(label_loader)       
    return data_fold_list, label_fold_list

def get_idx_label(label):
    id_idx = []
    cal_idx = []
    for i in range(len(label)):
        if label[i] == 'Normal':
            cal_idx.append(1)
            id_idx.append(0)
        else:
            cal_idx.append(0)
            if label[i] == 'MCI':
                id_idx.append(1)
            elif label[i] == 'AD':
                id_idx.append(2)
    return id_idx, torch.tensor(cal_idx)

def get_pair_idx_label_new(label, pair, idx_list=[0,1]):

    ''' For ADNI: NC=0, MCI=1, AD=2 '''
    pair_new = []
    label_new = [label[i] for i in range(len(label))]

    for i in range(len(pair)):
        p = pair[i]
        s = p[2]
        e = p[3]
        if label[s] in idx_list:
            pair_new.append(p)
            if label[s] ==2:
                for j in range(s,e):
                    label_new[j] = 1 
        else:
            for j in range(s,e):
                label_new[j] = -1
    return pair_new, label_new


''' Dataset Manipulation'''
def get_image_classification_result_given_fold(model, data_loader, label_loader, device):
    # given cls model, image loader(either for a fold or for all dataset), and its corresponding label loader
    # return results of the classification result, in two format
    # 1. data and label gt\pred\res, in tensor
    # 2. data and label gt\pred\res in batched list(compatible with data loader)
    # PS: it's easy to convert format to 1 in one line(since gt\pred\res is all scalar list): torch.cat(type2) -> type1

    batched_data_list = [] 
    batched_label_list = []
    batched_label_pred = []
    batched_label_res = []

    for batch_idx, [d] in enumerate(label_loader):
        batched_label_list.append(d)

    for batch_idx, (x) in enumerate(data_loader):
        x = x[0]
        batched_data_list.append(x)
        x = x.to(device)
        l = batched_label_list[batch_idx].float().to(device)
        acc, tot, pred, res = model.classify(x, l)
        batched_label_pred.append(torch.tensor(pred))
        batched_label_res.append(torch.tensor(res))
    
    return batched_data_list, batched_label_list, batched_label_pred, batched_label_res

def split_data_given_classification_result(data, label, pred, res, idx_attr='res', return_type='all'):
    if idx_attr == 'res':
        r_idx = [i for i in range(len(res)) if res[i] == 1]
        w_idx = [i for i in range(len(res)) if res[i] == 0]
    elif idx_attr == 'pred':
        r_idx = [i for i in range(len(pred)) if pred[i] == 1]
        w_idx = [i for i in range(len(pred)) if pred[i] == 0]
    elif idx_attr == 'label':
        r_idx = [i for i in range(len(label)) if label[i] == 1]
        w_idx = [i for i in range(len(label)) if label[i] == 0]               
    r_data = data[r_idx]
    r_label = label[r_idx]
    r_pred = pred[r_idx]
    r_res = res[r_idx]

    w_data = data[w_idx]
    w_label = label[w_idx]
    w_pred = pred[w_idx]
    w_res = res[w_idx]

    if return_type == 'all':
        return [r_data, r_label, r_pred, r_res], [w_data, w_label, w_pred, w_res]
    elif return_type == 'right':
        return [r_data, r_label, r_pred, r_res], None
    elif return_type == 'wrong':
        return None, [w_data, w_label, w_pred, w_res]

def get_data_pair_for_cyc_con_given_classification_result(data, label, pred, res, p_type='only_pred'):
    # TIPS: if res[i] == 1, then pred[i] == label[i]
    # thus we actually only needs to consider if we want only corrected classified images, or just all image.
    # param: p_type = 'only_pred' or 'res_then_pred'
    if p_type == 'only_pred':
        r_data_pair, w_data_pair = split_data_given_classification_result(data, label, pred, res, idx_attr='pred')
    elif p_type == 'res_then_pred':
        [r_data, r_label, r_pred, r_res], _ = split_data_given_classification_result(data, label, pred, res, idx_attr='res', return_type='right')
        r_data_pair, w_data_pair = split_data_given_classification_result(r_data, r_label, r_pred, r_res, idx_attr='pred')
    return r_data_pair, w_data_pair

def convert_data_pair_to_loader(data_pair, BATCH_SIZE=64, shuffle=False, seed=2020):
    [data, label, pred, res] = data_pair
    num = data.shape[0]
    idx = [i for i in range(0, num)]
    if shuffle == True:
        random.seed(seed)
        random.shuffle(idx)

    data_set = Data.TensorDataset(data[idx, :])
    label_set = Data.TensorDataset(label[idx])
    pred_set = Data.TensorDataset(pred[idx])
    res_set = Data.TensorDataset(res[idx])

    data_loader = Data.DataLoader(data_set, batch_size=BATCH_SIZE) 
    label_loader = Data.DataLoader(label_set, batch_size=BATCH_SIZE) 
    pred_loader = Data.DataLoader(pred_set, batch_size=BATCH_SIZE) 
    res_loader = Data.DataLoader(res_set, batch_size=BATCH_SIZE) 

    return [data_loader, label_loader, pred_loader, res_loader]

# added at Sep.21.2020, for completely new data processing pipeline.
def get_statistic_from_fold_list_4_image(fold_list_4_image, labels, num_cls=2, fold_num=5):
    # Usage: get num of images(+/-) in each fold
    stats_fold_list = np.zeros((num_cls, fold_num), dtype=int)
    num_of_image = len(fold_list_4_image)
    for i in range(num_of_image):
        fold = int(fold_list_4_image[i])
        label = int(labels[i])
        if 0 <= label < num_cls and 0 <= fold < fold_num:  
            stats_fold_list[label, fold] += 1
    return stats_fold_list

# # this function was added at Dec.5.2020, for recover file names for generated data fold.
def get_filename_of_data_fold_from_fold_list_4_image(idx_file_path, fold_list_4_image, labels, num_cls=2, fold_num=5):
    # Usage: get num of images(+/-) in each fold
    file_idx = np.genfromtxt(idx_file_path, dtype='str') 
    stats_fold_list = np.zeros((num_cls, fold_num), dtype=int)
    num_of_image = len(fold_list_4_image)
    filename_of_data_fold = [[[] for i in range(fold_num)] for i in range(num_cls)]
    for i in range(num_of_image):
        filename = file_idx[i]
        fold = int(fold_list_4_image[i])
        label = int(labels[i])
        if 0 <= label < num_cls and 0 <= fold < fold_num:  
            stats_fold_list[label, fold] += 1
            filename_of_data_fold[label][fold].append(filename)

    # Sanity Check: If filename is correctly feeded into each fold
    for i in range(num_cls):
        for j in range(fold_num):
            num_of_images = len(filename_of_data_fold[i][j])
            assert num_of_images == stats_fold_list[i, j]

    return filename_of_data_fold


def get_data_fold_given_fold_list_4_image(dataset, labels, fold_list_4_image, label_list=[0, 1], fold_num=5, return_type='np'):
    # Input:
    #    dataset - tensor with shape (num, 1, x, y, z)
    #    labels - ndarray with shape (num, )
    #    fold_list_4_image - ndarray with shape (num, )
    #    fold_num - integar, num of folds 
    #    return_type - str, either 'np' or 'torch', specifies the types of return data and label
    # return:
    #    data_fold - list with shape (num_cls, fold_num), storing a ndarray of each fold
    #    label_fold - lsit with shape (num_cls, fold_num), storing label w.r.t data_fold

    stats_fold_list = get_statistic_from_fold_list_4_image(fold_list_4_image, labels)
    num_of_image = len(dataset)
    num_cls = len(label_list)
    data_fold = [[] for i in range(num_cls)]
    label_fold = [[] for i in range(num_cls)]
    for i in range(num_cls):
        label = label_list[i]
        for j in range(fold_num):
            num_image = stats_fold_list[i, j]
            data_fold[i].append(np.zeros((num_image, 1, 64, 64, 64)))
            label_fold[i].append(np.ones(num_image) * label)

    fold_cursor = np.zeros_like(stats_fold_list, dtype=int)
    for i in range(num_of_image):
        data = dataset[i, :]
        label = int(labels[i])
        current_fold = int(fold_list_4_image[i])
        # if data should be added into data_fold
        if current_fold != -1 and label in label_list:
            cursor = fold_cursor[label][current_fold]
            data_fold[label][current_fold][cursor, :] = data
            fold_cursor[label][current_fold] += 1

    # Sanity Check: If data is correctly feeded into each fold
    # Also: If return_type = 'torch', then convert data into torch.tensor()
    for i in range(num_cls):
        for j in range(fold_num):
            cursor = fold_cursor[i][j]
            assert cursor == stats_fold_list[i, j]
            # print(i, j, cursor)
            if return_type == 'torch':
                data_fold[i][j] = torch.tensor(data_fold[i][j])
                label_fold[i][j] = torch.tensor(label_fold[i][j])

    return data_fold, label_fold

''' Data Augmentation '''
def augment_by_transformation(data, label, n, return_type='np', verbose=True):
    augment_scale = 1
    # np.random.seed(2022)
    if n <= data.shape[0]:
        return data, label
    else:
        new_label = []
        raw_n = data.shape[0]
        m = n - raw_n
        new_data = np.zeros((m, 1, data.shape[2], data.shape[3], data.shape[4]))
        for i in range(0, m):
            if i % 100 == 0 and verbose == True:
                print('round', i)
            idx = np.random.randint(0, raw_n)
            # new_age = age[idx]
            # new_sex = sex[idx]
            new_data[i] = data[idx].copy()
            new_data[i, 0, :, :, :] = sp.ndimage.interpolation.rotate(new_data[i, 0, :, :, :], np.random.uniform(-0.5, 0.5), axes=(1, 0), reshape=False)
            new_data[i, 0, :, :, :] = sp.ndimage.interpolation.rotate(new_data[i, 0, :, :, :], np.random.uniform(-0.5, 0.5), axes=(0, 2), reshape=False)
            new_data[i, 0, :, :, :] = sp.ndimage.interpolation.rotate(new_data[i, 0, :, :, :], np.random.uniform(-0.5, 0.5), axes=(1, 2), reshape=False)
            new_data[i, 0, :, :, :] = sp.ndimage.shift(new_data[i, 0, :, :, :], np.random.uniform(-0.5, 0.5))
            new_label.append(label[idx])
            # age = np.append(age, new_age)
            # sex = np.append(sex, new_sex)
        # output an example
        # array_img = nib.Nifti1Image(np.squeeze(new_data[3,:,:,:,0]),np.diag([1, 1, 1, 1]))  
        # filename = 'augmented_example.nii.gz'
        # nib.save(array_img,filename)
        # data = np.concatenate((data, new_data), axis=0)
        # label += new_label
        # print(new_data.shape)
        # print(new_data.type)
        if return_type == 'np':
            return new_data, np.asarray(new_label)
        elif return_type == 'torch':
            return torch.tensor(new_data).float(), torch.tensor(new_label)

def augment_data_fold(data_fold, label_fold, num_after_augment=512, return_type='np'):
    num_cls = len(data_fold)
    num_fold = len(data_fold[0])
    augment_data_fold = [[] for i in range(num_cls)]
    augment_label_fold = [[] for i in range(num_cls)]
    for i in range(num_cls):
        for j in range(num_fold):
            print('CLS ', i, ', FOLD', j)
            augment_data, augment_label = augment_by_transformation(data_fold[i][j], label_fold[i][j], num_after_augment, return_type)
            augment_data_fold[i].append(augment_data)
            augment_label_fold[i].append(augment_label)
            print(augment_data.shape)
    return augment_data_fold, augment_label_fold

''' From data and label fold to trainable Loader'''
def split_dataset_folds(dataset, label, folds=5, split_ratio=0.1, shuffle=True,seed=2020, BATCH_SIZE=10):
    num = dataset.shape[0]

    # ratio = int(num * split_ratio)
    ratio = int(num // folds)
    print('ratio:',ratio)
    idx = [i for i in range(0,num)]
    if shuffle == True:
        random.seed(seed)
        random.shuffle(idx)
    data_fold_list = []
    label_fold_list = []
    for i in range(folds):
        if i + 1 < folds:
            train_set = Data.TensorDataset(dataset[idx[i*ratio:(i+1)*ratio],:])
            label_set = Data.TensorDataset(label[idx[i*ratio:(i+1)*ratio]])
        else:
            train_set = Data.TensorDataset(dataset[idx[i*ratio:],:])
            label_set = Data.TensorDataset(label[idx[i*ratio:]])           
        train_loader = Data.DataLoader(train_set,batch_size =BATCH_SIZE)
        data_fold_list.append(train_loader)
    	
        label_loader = Data.DataLoader(label_set,batch_size =BATCH_SIZE)
        label_fold_list.append(label_loader)       
    return data_fold_list, label_fold_list

def merge_data_fold_with_augmented_data(data_fold, label_fold, augment_data_fold, augment_label_fold):
    num_cls = len(data_fold)
    num_fold = len(data_fold[0])
    for i in range(num_cls):
        for j in range(num_fold):
            data_fold[i][j] = np.concatenate([data_fold[i][j], augment_data_fold[i][j]])
            label_fold[i][j] = np.concatenate([label_fold[i][j], augment_label_fold[i][j]])
    return data_fold, label_fold   

def merge_data_fold_wrt_label(data_fold, label_fold):
    num_cls = len(data_fold)
    num_fold = len(label_fold[0])
    new_data_fold = []
    new_label_fold = []
    for j in range(num_fold):
        new_data = []
        new_label = []
        for i in range(num_cls):
            new_data.append(data_fold[i][j])
            new_label.append(label_fold[i][j])
        new_data = np.concatenate(new_data)
        print(new_data.shape)
        new_label = np.concatenate(new_label)
        new_data_fold.append(new_data)
        new_label_fold.append(new_label)
    return new_data_fold, new_label_fold

def flipped_data_fold(data_fold, label_fold):
    num_fold = len(data_fold)
    for i in range(num_fold):
        flipped_data = np.flip(data_fold[i], axis=1)
        data_fold[i] = np.concatenate([data_fold[i], flipped_data])
        label_fold[i] = np.concatenate([label_fold[i], label_fold[i]])
    return data_fold, label_fold

def split_data_fold(data_fold, label_fold, num_fold=5, shuffle=True, seed=2020, BATCH_SIZE=10):
    data_fold_list = []
    label_fold_list = []
    for i in range(num_fold):   
        num_image = data_fold[i].shape[0]
        idx = [i for i in range(0, num_image)]
        if shuffle == True:
            random.seed(seed)
            random.shuffle(idx)
        train_set = Data.TensorDataset(torch.tensor(data_fold[i][idx, :]))
        train_loader = Data.DataLoader(train_set, batch_size=BATCH_SIZE)
        label_set = Data.TensorDataset(torch.tensor(label_fold[i][idx]))
        label_loader = Data.DataLoader(label_set, batch_size=BATCH_SIZE)    
        data_fold_list.append(train_loader)
        label_fold_list.append(label_loader) 
    return data_fold_list, label_fold_list

def split_test_dataset(test_dataset, shuffle=False, seed=2020, BATCH_SIZE=10):
    test_num_image = test_dataset.shape[0]
    idx = [i for i in range(0, test_num_image)]
    if shuffle == True:
        random.seed(seed)
        random.shuffle(idx)
    data_set = Data.TensorDataset(torch.tensor(test_dataset[idx, :]))
    data_loader = Data.DataLoader(data_set, batch_size=BATCH_SIZE)
    return data_loader

''' Loss Helper '''
def mseloss(x, xhat, dim=5):
    loss = torch.nn.MSELoss(reduction='none')
    output = loss(x, xhat)
    output = output.mean(tuple(range(1,dim)))
    return output

def l1_loss(x, xhat, dim=5):
    loss = torch.nn.L1Loss(reduction='none')
    output = loss(x, xhat)
    output = output.mean(tuple(range(1,dim)))
    return output



''' Others Functionality '''
def get_label2num(num_cls=2):
    if num_cls == 2:
        D = {'AD':0, 'Normal':1}
    elif num_cls == 3:
        D = {'AD':0, 'Normal':1, 'MCI':2}
    return D

def map_lab_label(label):
    if label != 'C' and label != 'E' and label != 'HE':
        return 'UA'
    else:
        return label

def get_label2num_lab(num_cls=2):
    # only works for C vs E + HE
    if num_cls == 3:
        D = {'C':0, 'E':1, 'HE':2, 'UA':-1}
    elif num_cls == 2:
        D = {'C':0, 'E':1, 'HE':1, 'UA':-1}
    return D    


''' cls logit score'''
