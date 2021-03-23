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
# from unet_model import Cycle_Cons_3_Improvements, Convnet_SkipConnection
from unet_model_cond_warp import Cycle_Cons_3_Improvements, Convnet_SkipConnection
from torch import  nn, optim
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score


parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=512,    help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=7500, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=59, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=56,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=15,     help="Flag for training")
parser.add_argument('--Type',     type=str, default='LAB_CLS',     help="Flag for training")
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
LAB_path_group_data = "/home/groups/kpohl/t1_data/lab_data/img_64_longitudinal/"

LAB_data_name_list_txt = "/home/users/zucks626/miccai/lab_list.txt"

IPMI_data_pickle_save_path = "/scratch/users/zucks626/ADNI/IPMI/data_pickle_saved/"
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

label_file_path = IPMI_data_pickle_save_path + 'demographics_lab.csv'
if FLAG_Save_data_pickle == True:
    ut.get_Id_attr_lab_pkl_from_label_file(Dataset, label_file_path, IPMI_data_pickle_save_path) 
else:
    f = open(IPMI_data_pickle_save_path + "Id_attr_lab.pkl","rb")
    Id_attr = pkl.load(f)
    f.close() 
    print('Load Id_attr_lab.pkl success!')  

FLAG_Save_data_pickle = True
if FLAG_Save_data_pickle == True:
    pass
else:
    f = open(IPMI_data_pickle_save_path + "information_related_to_test_dataset.pkl","rb")
    information_related_to_test_dataset = pkl.load(f)
    f.close() 
    print('Load test visualization dataset success!')  
# test_dataset = information_related_to_test_dataset[0]
# num_of_test_pair = test_dataset.shape[0]
# print('test visualization dataset shape is:', test_dataset.shape) # shape: (BS, 2, 64, 64, 64)
# test_visualization_loader = ut.split_test_dataset(test_dataset, BATCH_SIZE=10)
FLAG_Save_data_pickle = False

Type = args.Type    
if Type == 'LAB_CLS':
    FLAG_TEST_FOLD = 4
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_orig_data_fold_fp32.pkl", "rb")
    data_fold = pkl.load(f)
    f.close()  
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_orig_label_fold.pkl", "rb")
    label_fold = pkl.load(f)
    f.close()  
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_augment_data_fold_0_fp32.pkl", "rb")
    augment_data_fold_0 = pkl.load(f)
    f.close()  
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_augment_label_fold_0.pkl", "rb")
    augment_label_fold_0 = pkl.load(f)
    f.close()  
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_augment_data_fold_1_fp32.pkl", "rb")
    augment_data_fold_1 = pkl.load(f)
    f.close()  
    f = open(IPMI_data_pickle_save_path + "LAB_CE+HE_augment_label_fold_1.pkl", "rb")
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
    # sleep(100000)
    flipped = False
    if flipped == True:
        data_fold, label_fold = ut.flipped_data_fold(data_fold, label_fold)
    # for i in range(5):
    #     print(data_fold[i].shape, label_fold[i].shape)
    train_loader_list, train_label_loader_list = ut.split_data_fold(data_fold, label_fold, num_fold=5, BATCH_SIZE=args.BATCH_SIZE)
    # sleep(10000)

def train_cls(cls, dataset, label, test_data, test_label, device, tqdm, fold, iter_max, iter_save, lr_decay_step, schedule=False):

    model = cls
    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    if schedule == True:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.65, verbose=True, patience=5, threshold=1e-3)

    # load label at first(5 fold)
    all_label_list = []
    for j in range(fold):
        test_label_list = []
        label_loader = label[j]
        for batch_idx, [d] in enumerate(label_loader):
            test_label_list.append(d)
        all_label_list.append(test_label_list)

    # load label at first(5 fold)
    all_label_test_list = []
    for j in range(fold):
        test_label_list = []
        label_loader = test_label[j]
        for batch_idx, [d] in enumerate(label_loader):
            test_label_list.append(d)
        all_label_test_list.append(test_label_list)
    
    
    i = 0
    lr = 5e-4
    for j in range(0, 5):
        print("Now test fold: ", str(j))
        # test_loader = dataset[j]
        test_loader = test_data[j]
        test_label_list = all_label_test_list[j]

        path = "/scratch/users/zucks626/ADNI/IPMI/checkpoints/"
        save_dir = path + model.name + "/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = save_dir+'log_'+str(j)
        f = open(filename,'w')
        f.close()
        filename = save_dir+'acc_'+str(j)
        f = open(filename,'w')
        f.close()
        if j + 1 < fold:
            train_loader_list = dataset[:j] + dataset[j+1:]
            train_label_list = all_label_list[:j] + all_label_list[j+1:]
        else:
            train_loader_list =  dataset[:j]
            train_label_list = all_label_list[:j]
        losses = []
        train_acc = []
        test_acc = [] 
        with tqdm(total=iter_max+20) as pbar:
            while True:
                yu = [] 
                for k in range(fold-1):
                    print('fold ', str(k))
                    train_loader = train_loader_list[k]
                    label_list = train_label_list[k]
                    
                    for batch_idx, (xu) in enumerate(train_loader):
                        i += 1 # i is num of gradient steps taken by end of loop iteration
                        xu = xu[0].to(device)
                        l = label_list[batch_idx].float().to(device)
                        # print(sum(l)/len(l))
                        optimizer.zero_grad()
                        if i == 1:
                            fold_idx_list = []
                            acc_list = []
                            tot_list = []
                            totlen = 0
                            total = 0
                            for k1 in range(fold - 1):
                                loader = train_loader_list[k1]
                                label_l = train_label_list[k1]
                                acc, tot = test_fold(model, loader, label_l, device)
                                total += acc * tot
                                totlen += tot
                                fold_idx = k1 + 1 if k1 < j else k1 + 2
                                print('Batch ', fold_idx ," acc:", acc)
                                fold_idx_list.append(fold_idx)
                                acc_list.append(acc)
                                tot_list.append(tot)

                            path = "/scratch/users/zucks626/ADNI/IPMI/checkpoints/"
                            save_dir = path + model.name + "/"
                            print('Train Batch '," acc:",total/totlen)
                            acc, tot = test_fold_realdata_balanced(model,test_loader,test_label_list,device)
                            print('Test Batch acc:', acc)
                            train_acc.append(total/totlen)
                            test_acc.append(acc)
                            # sleep(10000)
                            ut.writelog(save_dir +'log_'+str(j),i-1,acc_list,fold_idx_list,tot_list,total,totlen,acc,tot,j)
                            ut.writeacc(save_dir +'acc_'+str(j), i-1, acc_list, total/totlen, acc, fold_idx_list, j)

                        loss = model(xu, l, 'train')   
                        loss_temp = loss.detach().cpu()
                        losses.append(loss_temp)
                        print(loss)
                        print("Begin Backward")
                        loss.backward()
                        optimizer.step()
                        print("Finish Backward")
 
                        pbar.set_postfix(
                                loss='{:.2e}'.format(loss))
                        pbar.update(1)

                        if i % 25 == 0 or i == 1: 
                            fold_idx_list = []
                            acc_list = []
                            tot_list = []
                            totlen = 0
                            total = 0
                            for k1 in range(fold - 1):
                                loader = train_loader_list[k1]
                                label_l = train_label_list[k1]
                                acc, tot = test_fold(model, loader, label_l, device)
                                total += acc * tot
                                totlen += tot
                                fold_idx = k1+1 if k1 < j else k1 + 2
                                print('Batch ', fold_idx, " acc:", acc)
                                fold_idx_list.append(fold_idx)
                                acc_list.append(acc)
                                tot_list.append(tot)

                            path = "/scratch/users/zucks626/ADNI/IPMI/checkpoints/"
                            save_dir = path + model.name + "/"
                            print('Train Batch '," acc:",total/totlen)
                            acc, tot = test_fold_realdata_balanced(model,test_loader,test_label_list,device)
                            print('Test Batch acc:', acc)
                            train_acc.append(total/totlen)
                            test_acc.append(acc)
                            if schedule == True:
                                scheduler.step(acc)
                            ut.writelog(save_dir +'log_'+str(j),i,acc_list,fold_idx_list,tot_list,total,totlen,acc,tot,j)
                            ut.writeacc(save_dir +'acc_'+str(j), i, acc_list, total/totlen, acc, fold_idx_list, j)
                        # if i > 150:
                        #     lr = lr /2
                        #     for param_group in optimizer.param_groups:
                        #         param_group['lr'] = lr
                        # if i > 300:
                        #     lr = lr *2
                        #     for param_group in optimizer.param_groups:
                        #         param_group['lr'] = lr
                        if i % lr_decay_step ==0 and schedule==False:
                            lr = lr / 2
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr
                        
                        # Save model
                        if i % iter_save == 0:
                            ut.save_model_by_name_cls(cls, i)
                if i >= iter_max:
                # if i >= 4:
                    path = "/scratch/users/zucks626/ADNI/IPMI/checkpoints/"
                    save_dir = path + model.name + "/"
                    f = open(save_dir + "losses_" + str(j) + ".pkl", "wb")
                    pkl.dump(losses, f)
                    f.close()
                    ut.save_loss_image(losses, save_dir, 'Batch_' + str(j) + '_losses.png')
                    ut.save_loss_image(train_acc, save_dir, 'Batch_' + str(j) + '_train_acc.png')
                    ut.save_loss_image(test_acc, save_dir, 'Batch_' + str(j) + '_test_acc.png')
                    # ut.save_loss_image(vae_losses,save_dir, 'vae_losses.png')
                    # ut.save_loss_image(np.log(reg_losses),save_dir, 'reg_losses.png')
                    # ut.save_loss_image(relu_losses,save_dir, 'relu_losses.png')
                    print('save to:', save_dir + 'Batch_' + str(j) + '_losses.png')
                    i = 0

                    model = Classifier(name=model.name, device=model.device)
                    lr = 5*1e-4
                    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
                    # if vae == None:
                    #     ut.load_model_by_name_ae(model.vae, global_step=iter_restart, device=device)
                    #     model = Classifier(model.vae, clses =2, name = model.name,z_dim=model.z_dim,device =device,requires_grad=requires_grad).to(device)
                    # else:
                    #     vae_new = VAE3d(z_dim=model.z_dim, name=model.name, device=device, nn='v13')
                    #     model = Classifier(vae_new, clses =2, name = model.name,z_dim=model.z_dim,device =device,requires_grad=requires_grad).to(device)

                    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5*1e-4, weight_decay=1*1e-4)

                    break
    return


def test_visualization_dataset(cyc_con, data_loader, step):

    device = cyc_con.device
    test_dir_name = "test_visualization/"
    xn_list = []
    xp_list = []
    s_list_1 = []
    s_list_2 = []
    s_list_3 = []
    s_list_4 = []
    v_list_1 = []
    v_list_2 = []
    v_list_3 = []
    v_list_4 = []
    lp_c_list = []
    lp_d_list = []
    with torch.no_grad():
        for batch_idx, (test_data) in enumerate(data_loader):
            test_data = test_data[0]
            xn = test_data[:, 0, :].reshape(-1, 1, 64, 64, 64)
            xp = test_data[:, 1, :].reshape(-1, 1, 64, 64, 64)
            batch_size = test_data.shape[0]
            pos2neg = torch.ones(batch_size, 1).to(device) * -1
            neg2pos = torch.ones(batch_size, 1).to(device) * 1 
            xp = torch.tensor(xp).to(device)
            vec_field_1 = cyc_con.simulate(xp, pos2neg, mode='eval')
            # grad_loss = self.GRAD_Loss.loss(vec_field)
            simulated_1 = cyc_con.ST(xp, vec_field_1)
            # simulated_1 = cyc_con.trans.eval()(xp, pos2neg)
            vec_field_2 = cyc_con.simulate(simulated_1, neg2pos, mode='eval')
            simulated_2 = cyc_con.ST(simulated_1, vec_field_2)
            # simulated_2 = cyc_con.trans.eval()(simulated_1, neg2pos)
            simulated_1 = simulated_1.cpu().numpy()
            simulated_2 = simulated_2.cpu().numpy()
            # c = cyc_con.trans.eval()(xp, neg2pos)
            # lp_c = c.cpu().numpy()
            xp = xp.cpu().numpy()

            xn = torch.tensor(xn).to(device)
            vec_field_3 = cyc_con.simulate(xn, neg2pos, mode='eval')
            # grad_loss = self.GRAD_Loss.loss(vec_field)
            simulated_3 = cyc_con.ST(xn, vec_field_3)
            # simulated_1 = cyc_con.trans.eval()(xp, pos2neg)
            vec_field_4 = cyc_con.simulate(simulated_3, pos2neg, mode='eval')
            simulated_4 = cyc_con.ST(simulated_3, vec_field_4)
            # simulated_3 = cyc_con.trans.eval()(xn, neg2pos)
            # simulated_4 = cyc_con.trans.eval()(simulated_3, pos2neg)
            simulated_3 = simulated_3.cpu().numpy()
            simulated_4 = simulated_4.cpu().numpy()

            vec_field_1 = vec_field_1.cpu().numpy()
            vec_field_2 = vec_field_2.cpu().numpy()  
            vec_field_3 = vec_field_3.cpu().numpy()
            vec_field_4 = vec_field_4.cpu().numpy()          
            # d = cyc_con.trans.eval()(xn, pos2neg)
            # lp_d = d.cpu().numpy()
            xn = xn.cpu().numpy()
            xn_list.append(xn)
            xp_list.append(xp)
            s_list_1.append(simulated_1)
            s_list_2.append(simulated_2)
            s_list_3.append(simulated_3)
            s_list_4.append(simulated_4)
            v_list_1.append(vec_field_1)
            v_list_2.append(vec_field_2)
            v_list_3.append(vec_field_3)
            v_list_4.append(vec_field_4)
            # lp_c_list.append(lp_c)
            # lp_d_list.append(lp_d)
        xp = np.concatenate(xp_list)
        xn = np.concatenate(xn_list)
        s1 = np.concatenate(s_list_1)
        s2 = np.concatenate(s_list_2)
        s3 = np.concatenate(s_list_3)
        s4 = np.concatenate(s_list_4)
        v1 = np.concatenate(v_list_1)
        v2 = np.concatenate(v_list_2)
        v3 = np.concatenate(v_list_3)
        v4 = np.concatenate(v_list_4)
        # lp_c = np.concatenate(lp_c_list)
        # lp_d = np.concatenate(lp_d_list)
        print('generate images success!')
        name = 'xp'
        ut.save_image_pickel_by_name_cls(xp, name, step, path=IPMI_save + cyc_con.name + "/" + test_dir_name)                    
        name = 's1'
        ut.save_image_pickel_by_name_cls(s1, name, step, path=IPMI_save + cyc_con.name + "/" + test_dir_name)
        name = 's2'
        ut.save_image_pickel_by_name_cls(s2, name, step, path=IPMI_save + cyc_con.name + "/" + test_dir_name)
        name = 'xn'
        ut.save_image_pickel_by_name_cls(xn, name, step, path=IPMI_save + cyc_con.name + "/" + test_dir_name)
        name = 's3'
        ut.save_image_pickel_by_name_cls(s3, name, step, path=IPMI_save + cyc_con.name + "/" + test_dir_name)
        name = 's4'
        ut.save_image_pickel_by_name_cls(s4, name, step, path=IPMI_save + cyc_con.name + "/" + test_dir_name)
        name = 'v1'
        ut.save_image_pickel_by_name_cls(v1, name, step, path=IPMI_save + cyc_con.name + "/" + test_dir_name)
        name = 'v2'
        ut.save_image_pickel_by_name_cls(v2, name, step, path=IPMI_save + cyc_con.name + "/" + test_dir_name)
        name = 'v3'
        ut.save_image_pickel_by_name_cls(v3, name, step, path=IPMI_save + cyc_con.name + "/" + test_dir_name)
        name = 'v4'
        ut.save_image_pickel_by_name_cls(v4, name, step, path=IPMI_save + cyc_con.name + "/" + test_dir_name)

        # xn_list.append(xn.cpu().numpy())
        # xp_list.append(xp.cpu().numpy())
        # simulated_list_1.append(simulated_1.cpu().numpy())
        # simulated_list_2.append(simulated_2.cpu().numpy())
        # simulated_list_3.append(simulated_3.cpu().numpy())
        # simulated_list_4.append(simulated_4.cpu().numpy())
        # lp_c.append(c.cpu().numpy())
        # lp_d.append(d.cpu().numpy())

def test_fold(model, test_loader, test_label_list, device):
    t_acc = 0
    t_tot = 0 
    for batch_idx, (xu) in enumerate(test_loader):
        # i += 1 # i is num of gradient steps taken by end of loop iteration       
        xu = xu[0].to(device)
        l = test_label_list[batch_idx].float().to(device)
        b = np.random.rand()

        acc, tot, pred, res = model.classify(xu, l)
        t_acc += acc * tot
        t_tot += tot
        # list_f = [0, 1, 2, 3, 4]
        # list_f.pop(np.random.randint(2, 5))

        # xu_f = torch.flip(xu, [4])
        # acc, tot, pred, res = model.classify(xu_f, l)
        # t_acc += acc * tot
        # t_tot += tot
    acc = t_acc/t_tot
    return acc, t_tot

def test_fold_realdata_balanced(model, test_loader, test_label_list, device, num_cls=2):
    t_acc = 0
    t_tot = 0 
    total_labels = []
    total_preds = []
    for batch_idx, (xu) in enumerate(test_loader):
        # i += 1 # i is num of gradient steps taken by end of loop iteration       
        xu = xu[0].to(device)
        l = test_label_list[batch_idx].float()
        # b = np.random.rand()
        total_labels.append(l)
        # total_labels.append(l)
        l = l.to(device)
        acc, tot, pred, res = model.classify(xu, l)
        total_preds.append(torch.tensor(pred))
        t_acc += acc * tot
        t_tot += tot
        # list_f = [0, 1, 2, 3, 4]
        # list_f.pop(np.random.randint(2, 5))

        # xu_f = torch.flip(xu, [4])
        # acc, tot, pred, res = model.classify(xu_f, l)
        # total_preds.append(torch.tensor(pred))
        # t_acc += acc * tot
        # t_tot += tot
    acc = t_acc/t_tot
    total_labels = torch.cat(total_labels)
    total_preds = torch.cat(total_preds)
    # confusion_matrixs = confusion_matrix(total_labels, total_preds) 
    f1_scores = f1_score(total_labels, total_preds, average='weighted')
    avg_recall_scores = recall_score(total_labels, total_preds, average='weighted')
    print('f1_score is:', f1_scores, 'recall_scores is:', avg_recall_scores, 'acc is:', acc)

    return acc, t_tot

def prepare_train_data_based_on_trained_classifier(model, data_loader_list, label_loader_list, device, pipeline='push_to_all', p_type='res_then_pred', BATCH_SIZE=64, shuffle=False, seed=2020):
    # pipiline = 'push_to_all' or 'stay_in_fold'
    folds = len(data_loader_list)
    all_data_list = []
    all_label_list = []
    all_pred_list = []
    all_res_list = []

    for cur_fold in range(folds):
        data_loader = data_loader_list[cur_fold]
        label_loader = label_loader_list[cur_fold]
        batched_data_list, batched_label_list, batched_pred_list, batched_res_list = ut.get_image_classification_result_given_fold(model, data_loader, label_loader, device)
        # print(batched_data_list[0].shape)
        data_list = torch.cat(batched_data_list) 
        label_list = torch.cat(batched_label_list) 
        pred_list = torch.cat(batched_pred_list) 
        res_list = torch.cat(batched_res_list)
        all_data_list.append(data_list)
        all_label_list.append(label_list)
        all_pred_list.append(pred_list)
        all_res_list.append(res_list)
    
    if pipeline == 'push_to_all':
        data = torch.cat(all_data_list)
        label = torch.cat(all_label_list)
        pred = torch.cat(all_pred_list)
        res = torch.cat(all_res_list)
        # sleep(10000)
        r_data_pair, w_data_pair = ut.get_data_pair_for_cyc_con_given_classification_result(data, label, pred, res, p_type=p_type)
        # list - len=4, [data, label, pred, res]
        # print(r_data_pair[0].shape, w_data_pair[0].shape)
        # sleep(10000)
        r_loader = ut.convert_data_pair_to_loader(r_data_pair, BATCH_SIZE=BATCH_SIZE, shuffle=shuffle, seed=seed)
        w_loader = ut.convert_data_pair_to_loader(w_data_pair, BATCH_SIZE=BATCH_SIZE, shuffle=shuffle, seed=seed)
        return r_loader, w_loader, r_data_pair, w_data_pair
    elif pipeline == 'stay_in_fold':
        r_loader_list = []
        w_loader_list = []
        for cur_fold in range(folds):
            data = all_data_list[cur_fold]
            label = all_label_list[cur_fold]
            pred = all_pred_list[cur_fold]
            res = all_res_list[cur_fold]
            r_data_pair, w_data_pair = ut.get_data_pair_for_cyc_con_given_classification_result(data, label, pred, res, p_type=p_type)
            r_loader = ut.convert_data_pair_to_loader(r_data_pair, BATCH_SIZE=BATCH_SIZE, shuffle=shuffle, seed=seed)
            w_loader = ut.convert_data_pair_to_loader(w_data_pair, BATCH_SIZE=BATCH_SIZE, shuffle=shuffle, seed=seed)   
            r_loader_list.append(r_loader)
            w_loader_list.append(w_loader)
        return r_loader_list, w_loader_list         

def train_cyc_con(cyc_con, pos_loader, neg_loader, discriminator, r_data_pair, w_data_pair, gan_mode='non_saturate', iter_max=1001, verbose=True, optimizer_list=None, pipeline='push_to_all', BATCH_SIZE=40):
    if optimizer_list == None:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, cyc_con.parameters()), lr=2e-4, weight_decay=1e-4)
        optimizer_1 = optim.Adam(filter(lambda p: p.requires_grad, cyc_con.parameters()), lr=1e-4, weight_decay=1e-4)
        optimizer_2 = optim.Adam(filter(lambda p: p.requires_grad, cyc_con.parameters()), lr=1e-4, weight_decay=1e-4)
        optimizer_3 = optim.Adam(filter(lambda p: p.requires_grad, cyc_con.parameters()), lr=2e-4, weight_decay=1e-4)
        optimizer_4 = optim.Adam(filter(lambda p: p.requires_grad, cyc_con.parameters()), lr=1e-4, weight_decay=1e-4)
        optimizer_5 = optim.Adam(filter(lambda p: p.requires_grad, cyc_con.parameters()), lr=1e-4, weight_decay=1e-4) 
        optimizer_6_d = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=1e-4, weight_decay=1e-4) 
        optimizer_6_g = optim.Adam(filter(lambda p: p.requires_grad, cyc_con.parameters()), lr=1e-4, weight_decay=1e-4)    
    else:
        optimizer = optimizer_list[0]
        optimizer_1 = optimizer_list[1]
        optimizer_2 = optimizer_list[2]
        optimizer_3 = optimizer_list[3]
        optimizer_4 = optimizer_list[4]
        optimizer_5 = optimizer_list[5] 
        optimizer_6 = optimizer_list[6]  
    
    losses_cyc = []
    c_acc_list = []
    d_acc_list = []
    ctrans_bce_losses = []
    dtrans_bce_losses = []
    cdtrans_losses = []
    dctrans_losses = []

    adv_g_loss_p_list = []
    adv_d_loss_p_list = []
    adv_g_loss_n_list = []
    adv_d_loss_n_list = []

    grad_losses_c = []
    cyc_grad_losses_c = []
    grad_losses_d = []
    cyc_grad_losses_d = []

    xn_list = []
    xp_list = []
    simulated_list_1 = []
    simulated_list_2 = []
    simulated_list_3 = []
    simulated_list_4 = []
    lp_c = []
    lp_d = []

    w_dw = 1
    w_cw = 1
    w_cdtrans = 1
    w_dctrans = 1
    w_clf_c = 1
    w_clf_d = 1
    w_adv_g_loss = 0.1
    w_adv_d_loss = 0.5

    w_grad_loss = 0.0005
    w_cyc_grad_loss = 0.01

    num_p_data = r_data_pair[0].shape[0]
    num_n_data = w_data_pair[0].shape[0]
    # global test_dataset
    # test_dataset = torch.tensor(test_dataset)

    # cyc_con.con_enc = Contrastive_Encoding(enc=None, z_dim=512, device=cyc_con.device)
    # optimizer_6 = optim.Adam(filter(lambda p: p.requires_grad, cyc_con.parameters()), lr=1e-4, weight_decay=1e-4)
    init_step = -1


    if pipeline == 'push_to_all':
        [r_data_loader, r_label_loader, r_pred_loader, r_res_loader] = pos_loader
        [w_data_loader, w_label_loader, w_pred_loader, w_res_loader] = neg_loader
        p_label_list = []
        p_pred_list = []
        p_res_list = []
        n_label_list = []
        n_pred_list = []
        n_res_list = []         
        for batch_idx, [d] in enumerate(r_label_loader):
            p_label_list.append(d)
        for batch_idx, [d] in enumerate(r_pred_loader):
            p_pred_list.append(d)
        for batch_idx, [d] in enumerate(r_res_loader):
            p_res_list.append(d)
        for batch_idx, [d] in enumerate(w_label_loader):
            n_label_list.append(d)
        for batch_idx, [d] in enumerate(w_pred_loader):
            n_pred_list.append(d)
        for batch_idx, [d] in enumerate(w_res_loader):
            n_res_list.append(d)

        xn_list = []
        for batch_idx, (xn) in enumerate(w_data_loader):
            xn_list.append(xn[0])
        neg_batch_num = len(xn_list)
        print(neg_batch_num, xn_list[0].shape)
        cnt_neg_batch = 0 
        step = 0
        while True:
            if step >= iter_max:
                break
            for batch_idx, (xp) in enumerate(r_data_loader):
                idx_p = np.random.permutation(num_p_data)
                idx_p = idx_p[:BATCH_SIZE]
                idx_n = np.random.permutation(num_n_data)
                idx_n = idx_n[:BATCH_SIZE]
                idx_p_r = np.random.permutation(num_p_data)
                idx_p_r = idx_p_r[:BATCH_SIZE]
                idx_n_r = np.random.permutation(num_n_data)
                idx_n_r = idx_n_r[:BATCH_SIZE]
                r_xp = r_data_pair[0][idx_p]
                r_xn = w_data_pair[0][idx_n]
                r_xp_1 = r_data_pair[0][idx_p_r]
                r_xn_1 = w_data_pair[0][idx_n_r]

                if verbose == True:
                    print('idx_n:', idx_n)
                    print('idx_p:', idx_p)
                    print('shape of xn and xp:', r_xn.shape, r_xp.shape)
                xp = xp[0].to(device)
                lp = p_label_list[batch_idx].to(device)
                # print(lp)
                predp = p_pred_list[batch_idx].to(device)
                resp = p_res_list[batch_idx].to(device)


                # if step % 50 == 0:
                if step % 50 == 0 and step > 0:
                    r_logits, r_trans_logits, r_cyc_back_logits = get_logits_from_loader(cyc_con.e_cls, cyc_con, r_data_loader, -1)

                    w_logits, w_trans_logits, w_cyc_back_logits = get_logits_from_loader(cyc_con.e_cls, cyc_con, w_data_loader, 1)
                    name = 'r_logits_r'
                    ut.save_image_pickel_by_name_cls([r_logits, r_trans_logits, r_cyc_back_logits], name, step, path=IPMI_save + cyc_con.name + "/") 
                    name = 'w_logits_r'
                    ut.save_image_pickel_by_name_cls([w_logits, w_trans_logits, w_cyc_back_logits], name, step, path=IPMI_save + cyc_con.name + "/") 
                    r_logits, r_trans_logits, r_cyc_back_logits = get_logits_from_loader(cyc_con.e_cls, cyc_con, r_data_loader, 1)
                    w_logits, w_trans_logits, w_cyc_back_logits = get_logits_from_loader(cyc_con.e_cls, cyc_con, w_data_loader, -1)
                    name = 'r_logits_w'
                    ut.save_image_pickel_by_name_cls([r_logits, r_trans_logits, r_cyc_back_logits], name, step, path=IPMI_save + cyc_con.name + "/") 
                    name = 'w_logits_w'
                    ut.save_image_pickel_by_name_cls([w_logits, w_trans_logits, w_cyc_back_logits], name, step, path=IPMI_save + cyc_con.name + "/") 
                    # sleep(10000)
                    # w_logits, w_trans_logits, w_cyc_back_logits, r_logits, r_trans_logits, r_cyc_back_logits = get_test_logits_from_loader(cyc_con.e_cls, cyc_con, test_visualization_loader)
                    # name = 'test_logits_r'
                    # ut.save_image_pickel_by_name_cls([r_logits, r_trans_logits, r_cyc_back_logits], name, step, path=IPMI_save + cyc_con.name + "/") 
                    # name = 'test_logits_w'
                    # ut.save_image_pickel_by_name_cls([w_logits, w_trans_logits, w_cyc_back_logits], name, step, path=IPMI_save + cyc_con.name + "/") 
 
                    # sleep(10000)
                    # test_visualization_dataset(cyc_con, test_visualization_loader, step)
                    # sleep(10000)
                if step % 50 == 0:
                    # neg_lp_loss, dtrans_bce_loss, cdtrans_loss, acc_d = cyc_con.forward_no_grad(xn, 1, verbose=True)
                    # # sleep(10000)
                    # pos_lp_loss, ctrans_bce_loss, dctrans_loss, acc_c = cyc_con.forward_no_grad(xp, -1, verbose=True)
                    # d_loss_n, g_loss_n = cyc_con.forward_no_grad_adv(xn,  1, discriminator, verbose=True, gan_mode=gan_mode)
                    # d_loss_p, g_loss_p = cyc_con.forward_no_grad_adv(xp, -1, discriminator, verbose=True, gan_mode=gan_mode)
                    # with torch.no_grad():
                    #     batch_size_p = xp.shape[0]
                    #     batch_size_n = xn.shape[0]
                    #     pos2neg_p = torch.ones(batch_size_p, 1).to(device) * -1
                    #     neg2pos_p = torch.ones(batch_size_p, 1).to(device) * 1 
                    #     pos2neg_n = torch.ones(batch_size_n, 1).to(device) * -1
                    #     neg2pos_n = torch.ones(batch_size_n, 1).to(device) * 1 
                    #     simulated_1 = cyc_con.trans.eval()(xp, pos2neg_p)
                    #     simulated_2 = cyc_con.trans.eval()(simulated_1, neg2pos_p)
                    #     simulated_3 = cyc_con.trans.eval()(xn, neg2pos_n)
                    #     simulated_4 = cyc_con.trans.eval()(simulated_3, pos2neg_n)
                    #     c = cyc_con.trans.eval()(xp, neg2pos_p)
                    #     d = cyc_con.trans.eval()(xn, pos2neg_n)
                    #     xn_list.append(xn.cpu().numpy())
                    #     xp_list.append(xp.cpu().numpy())
                    #     simulated_list_1.append(simulated_1.cpu().numpy())
                    #     simulated_list_2.append(simulated_2.cpu().numpy())
                    #     simulated_list_3.append(simulated_3.cpu().numpy())
                    #     simulated_list_4.append(simulated_4.cpu().numpy())
                    #     lp_c.append(c.cpu().numpy())
                    #     lp_d.append(d.cpu().numpy())
                    if step != 0:
                        ut.save_model_by_name_cls(cyc_con, step)
                        # ut.save_model_by_name_cls(discriminator, step)
                    # if verbose == True:
                    #     print(step, 'd_acc', acc_d, 'c_acc', acc_c, 'dc_loss', dctrans_loss.data.cpu().numpy(), 'cd_loss', cdtrans_loss.data.cpu().numpy())
                    #     print('c_lp:', neg_lp_loss.data.cpu().numpy(), 'd_lp:',pos_lp_loss.data.cpu().numpy(), 'bce_c_loss:', ctrans_bce_loss.data.cpu().numpy(), 'bce_d_loss:', dtrans_bce_loss.data.cpu().numpy()) 
                    #     print('d_loss_p:', d_loss_p, 'g_loss_p:', g_loss_p, 'd_loss_n:', d_loss_n, 'g_loss_p:', g_loss_p)
                    # sleep(10000)


                if step > init_step:
                    for j in range(1):
                        optimizer_1.zero_grad()
                        temp_no_grad_list = ['lp', 'trans_bce']
                        pos_lp_loss, ctrans_bce_loss, dctrans_loss, acc_c, grad_loss, cyc_grad_loss = cyc_con.forward_no_grad(xp, -1, with_no_grad_list=temp_no_grad_list)
                        dctrans_loss = w_dctrans * dctrans_loss  #+ w_cyc_grad_loss * cyc_grad_loss #+ w_grad_loss * grad_loss
                        dctrans_loss.backward()
                        optimizer_1.step()  
                for j in range(1):
                    optimizer.zero_grad()
                    temp_no_grad_list = ['lp', 'trans']
                    pos_lp_loss, ctrans_bce_loss, dctrans_loss, acc_c, grad_loss, cyc_grad_loss = cyc_con.forward_no_grad(xp, -1, with_no_grad_list=temp_no_grad_list)
                    ctrans_bce_loss = w_clf_c * ctrans_bce_loss + w_grad_loss * grad_loss
                    ctrans_bce_loss.backward()
                    optimizer.step()
                # if step % 2 == 1 and step > 0:
                if step % 10 == 0 and verbose == True:
                    print('step:', step, 'ctrans_bce_loss is:', ctrans_bce_loss)
                r_xp = r_xp.to(device)
                r_xn_1 = r_xn_1.to(device)


                if step > init_step:
                    for j in range(1):
                        optimizer_1.zero_grad()
                        temp_no_grad_list = ['lp', 'trans_bce']
                        pos_lp_loss, ctrans_bce_loss, dctrans_loss, acc_c, grad_loss, cyc_grad_loss = cyc_con.forward_no_grad(r_xp, -1, with_no_grad_list=temp_no_grad_list)
                        dctrans_loss = w_dctrans * dctrans_loss  # + w_cyc_grad_loss * cyc_grad_loss # + w_grad_loss * grad_loss
                        dctrans_loss.backward()
                        optimizer_1.step()  
                for j in range(1):
                    optimizer.zero_grad()
                    temp_no_grad_list = ['lp', 'trans']
                    pos_lp_loss, ctrans_bce_loss, dctrans_loss, acc_c, grad_loss, cyc_grad_loss = cyc_con.forward_no_grad(r_xp, -1, with_no_grad_list=temp_no_grad_list)
                    ctrans_bce_loss = w_clf_c * ctrans_bce_loss + w_grad_loss * grad_loss
                    ctrans_bce_loss.backward()
                    optimizer.step()
                # for j in range(2):
                #     optimizer_6_d.zero_grad()
                #     temp_no_grad_list = ['adv']
                #     d_loss_p, _ = cyc_con.forward_no_grad_adv_random(r_xp, r_xn_1, -1, discriminator, with_no_grad_list=temp_no_grad_list, gan_mode=gan_mode)
                #     if step < 10 or d_loss_p >= 0.3:
                #         temp_no_grad_list = []
                #         d_loss_p, _ = cyc_con.forward_no_grad_adv_random(r_xp, r_xn_1, -1, discriminator, with_no_grad_list=temp_no_grad_list, gan_mode=gan_mode)
                #         d_loss_p = w_adv_d_loss * d_loss_p
                #         d_loss_p.backward()
                #         optimizer_6_d.step() 
                #     if j + 1 < 2: 
                #         idx_p = np.random.permutation(num_p_data)
                #         idx_p = idx_p[:BATCH_SIZE]
                #         idx_n_r = np.random.permutation(num_n_data)
                #         idx_n_r = idx_n_r[:BATCH_SIZE]
                #         r_xp = r_data_pair[0][idx_p]
                #         r_xn_1 = w_data_pair[0][idx_n_r]
                #         r_xp = r_xp.to(device)
                #         r_xn_1 = r_xn_1.to(device)
                # i = 1
                # if step > 30:
                #     i = 2
                # # if step > 500:
                # #     i = 3
                # for j in range(i):
                #     optimizer_6_g.zero_grad()
                #     temp_no_grad_list = ['adv']
                #     _, g_loss_p = cyc_con.forward_no_grad_adv_random(r_xp, r_xn_1, -1, discriminator, with_no_grad_list=temp_no_grad_list, gan_mode=gan_mode)
                #     if g_loss_p >= 0.2:
                #         temp_no_grad_list = []
                #         _, g_loss_p = cyc_con.forward_no_grad_adv_random(r_xp, r_xn_1, -1, discriminator, with_no_grad_list=temp_no_grad_list, gan_mode=gan_mode)
                #         g_loss_p = w_adv_g_loss * g_loss_p
                #         g_loss_p.backward()
                #         if step > 5:
                #             optimizer_6_g.step()   
                #             if j +1 < i:
                #                 idx_p = np.random.permutation(num_p_data)
                #                 idx_p = idx_p[:BATCH_SIZE]
                #                 idx_n_r = np.random.permutation(num_n_data)
                #                 idx_n_r = idx_n_r[:BATCH_SIZE]
                #                 r_xp = r_data_pair[0][idx_p]
                #                 r_xn_1 = w_data_pair[0][idx_n_r]
                #                 r_xp = r_xp.to(device)
                #                 r_xn_1 = r_xn_1.to(device) 

                # torch.cuda.empty_cache()                 
                # pos_lp_loss, ctrans_bce_loss, dctrans_loss, acc_c = cyc_con.forward_no_grad(xp, -1)
                # for j in range(1):
                #     optimizer_6.zero_grad()
                #     temp_no_grad_list = []
                #     neg_con_loss = cyc_con.forward_no_grad_con(xn, 1, with_no_grad_list=temp_no_grad_list)
                #     neg_con_loss = 1 * neg_con_loss
                #     neg_con_loss.backward()
                #     optimizer_6.step()  
                # for j in range(1):
                #     optimizer_2.zero_grad()
                #     temp_no_grad_list = ['trans_bce', 'trans']
                #     pos_lp_loss, ctrans_bce_loss, dctrans_loss, acc_c = cyc_con.forward_no_grad(xp, -1, with_no_grad_list=temp_no_grad_list)
                #     pos_lp_loss = w_dw * pos_lp_loss
                #     pos_lp_loss.backward()
                #     optimizer_2.step()       
                xn = xn_list[cnt_neg_batch].clone().to(device)
                ln = n_label_list[cnt_neg_batch].to(device)
                # print(ln)
                # sleep(10000)
                predn = n_pred_list[cnt_neg_batch].to(device)
                resn = n_res_list[cnt_neg_batch].to(device)
                cnt_neg_batch = (cnt_neg_batch + 1) % neg_batch_num                    

                if step > init_step:
                    for j in range(1):
                        optimizer_4.zero_grad()
                        temp_no_grad_list = ['lp', 'trans_bce']
                        neg_lp_loss, dtrans_bce_loss, cdtrans_loss, acc_d, grad_loss, cyc_grad_loss = cyc_con.forward_no_grad(xn, 1, with_no_grad_list=temp_no_grad_list)
                        dctrans_loss = w_cdtrans * cdtrans_loss # + w_cyc_grad_loss * cyc_grad_loss # + w_grad_loss * grad_loss
                        dctrans_loss.backward()
                        optimizer_4.step()  
                for j in range(1):
                    optimizer_3.zero_grad()
                    temp_no_grad_list = ['lp', 'trans']
                    neg_lp_loss, dtrans_bce_loss, cdtrans_loss, acc_d, grad_loss, cyc_grad_loss = cyc_con.forward_no_grad(xn, 1, with_no_grad_list=temp_no_grad_list)
                    dtrans_bce_loss = w_clf_d * dtrans_bce_loss + w_grad_loss * grad_loss
                    dtrans_bce_loss.backward()
                    optimizer_3.step()
                # if step % 2 == 0 and step > 0: 
                r_xn = r_xn.to(device)
                r_xp_1 = r_xp_1.to(device)

                if step > init_step:
                    for j in range(1):
                        optimizer_4.zero_grad()
                        temp_no_grad_list = ['lp', 'trans_bce']
                        neg_lp_loss, dtrans_bce_loss, cdtrans_loss, acc_d, grad_loss, cyc_grad_loss = cyc_con.forward_no_grad(r_xn, 1, with_no_grad_list=temp_no_grad_list)
                        dctrans_loss = w_cdtrans * cdtrans_loss #+ w_cyc_grad_loss * cyc_grad_loss # + w_grad_loss * grad_loss
                        dctrans_loss.backward()
                        optimizer_4.step()  
                if np.random.rand() < 0.7:
                    for j in range(1):
                        optimizer_3.zero_grad()
                        temp_no_grad_list = ['lp', 'trans']
                        neg_lp_loss, dtrans_bce_loss, cdtrans_loss, acc_d, grad_loss, cyc_grad_loss = cyc_con.forward_no_grad(r_xn, 1, with_no_grad_list=temp_no_grad_list)
                        dtrans_bce_loss = w_clf_d * dtrans_bce_loss + w_grad_loss * grad_loss
                        dtrans_bce_loss.backward()
                        optimizer_3.step()
                # for j in range(2):
                #     optimizer_6_d.zero_grad()
                #     temp_no_grad_list = ['adv']
                #     d_loss_n, _ = cyc_con.forward_no_grad_adv_random(r_xn, r_xp_1, 1, discriminator, with_no_grad_list=temp_no_grad_list, gan_mode=gan_mode)
                #     if step < 10 or d_loss_n > 0.3:
                #         temp_no_grad_list = []
                #         d_loss_n, _ = cyc_con.forward_no_grad_adv_random(r_xn, r_xp_1, 1, discriminator, with_no_grad_list=temp_no_grad_list, gan_mode=gan_mode)
                #         d_loss_n = w_adv_d_loss * d_loss_n
                #         d_loss_n.backward()
                #         optimizer_6_d.step() 
                #         if j + 1 < 2: 
                #             idx_n = np.random.permutation(num_n_data)
                #             idx_n = idx_n[:BATCH_SIZE]
                #             idx_p_r = np.random.permutation(num_p_data)
                #             idx_p_r = idx_p_r[:BATCH_SIZE]
                #             r_xn = w_data_pair[0][idx_n]
                #             r_xp_1 = r_data_pair[0][idx_p_r]
                #             r_xp_1 = r_xp_1.to(device)
                #             r_xn = r_xn.to(device)  

                # for j in range(i):
                #     optimizer_6_g.zero_grad()
                #     temp_no_grad_list = ['adv']
                #     _, g_loss_n = cyc_con.forward_no_grad_adv_random(r_xn, r_xp_1, 1, discriminator, with_no_grad_list=temp_no_grad_list, gan_mode=gan_mode)
                #     if g_loss_n > 0.2:
                #         temp_no_grad_list = []
                #         _, g_loss_n = cyc_con.forward_no_grad_adv_random(r_xn, r_xp_1, 1, discriminator, with_no_grad_list=temp_no_grad_list, gan_mode=gan_mode)
                #         g_loss_n = w_adv_g_loss * g_loss_n
                #         g_loss_n.backward()
                #         if step > 5:
                #             optimizer_6_g.step() 
                #             if j + 1 < i:
                #                 idx_n = np.random.permutation(num_n_data)
                #                 idx_n = idx_n[:BATCH_SIZE]
                #                 idx_p_r = np.random.permutation(num_p_data)
                #                 idx_p_r = idx_p_r[:BATCH_SIZE]
                #                 r_xn = w_data_pair[0][idx_n]
                #                 r_xp_1 = r_data_pair[0][idx_p_r]
                #                 r_xp_1 = r_xp_1.to(device)
                #                 r_xn = r_xn.to(device)  
                # torch.cuda.empty_cache()

                # if step % 50 >= 49: 
                #     test_xp_idx = np.random.permutation(num_of_test_pair)
                #     test_xp_idx = test_xp_idx[:BATCH_SIZE]
                #     test_xn_idx = np.random.permutation(num_of_test_pair)
                #     test_xn_idx = test_xn_idx[:BATCH_SIZE]
                #     test_xn = test_dataset[test_xn_idx,0,:,:,:].unsqueeze(1).to(device)
                #     test_xp = test_dataset[test_xp_idx,1,:,:,:].unsqueeze(1).to(device)
                #     # print(test_xn.shape)
                #     # print(test_xp.shape)
                #     # sleep(1000)
                #     for j in range(1):
                #         optimizer_3.zero_grad()
                #         temp_no_grad_list = ['lp', 'trans']
                #         neg_lp_loss, dtrans_bce_loss, cdtrans_loss, acc_d, grad_loss, cyc_grad_loss = cyc_con.forward_no_grad(test_xn, 1, with_no_grad_list=temp_no_grad_list)
                #         dtrans_bce_loss = w_clf_d * dtrans_bce_loss # + w_grad_loss * grad_loss
                #         dtrans_bce_loss.backward()
                #         optimizer_3.step()
                #     for j in range(1):
                #         optimizer_4.zero_grad()
                #         temp_no_grad_list = ['lp', 'trans_bce']
                #         neg_lp_loss, dtrans_bce_loss, cdtrans_loss, acc_d, grad_loss, cyc_grad_loss = cyc_con.forward_no_grad(test_xn, 1, with_no_grad_list=temp_no_grad_list)
                #         dctrans_loss = w_cdtrans * cdtrans_loss #+ w_grad_loss * grad_loss + w_cyc_grad_loss * cyc_grad_loss
                #         dctrans_loss.backward()
                #         optimizer_4.step()   
                #     for j in range(1):
                #         optimizer.zero_grad()
                #         temp_no_grad_list = ['lp', 'trans']
                #         pos_lp_loss, ctrans_bce_loss, dctrans_loss, acc_c, grad_loss, cyc_grad_loss = cyc_con.forward_no_grad(test_xp, -1, with_no_grad_list=temp_no_grad_list)
                #         ctrans_bce_loss = w_clf_c * ctrans_bce_loss # + w_grad_loss * grad_loss
                #         ctrans_bce_loss.backward()
                #         optimizer.step()
                #     for j in range(1):
                #         optimizer_1.zero_grad()
                #         temp_no_grad_list = ['lp', 'trans_bce']
                #         pos_lp_loss, ctrans_bce_loss, dctrans_loss, acc_c, grad_loss, cyc_grad_loss = cyc_con.forward_no_grad(test_xp, -1, with_no_grad_list=temp_no_grad_list)
                #         dctrans_loss = w_dctrans * dctrans_loss # + w_grad_loss * grad_loss + w_cyc_grad_loss * cyc_grad_loss
                #         dctrans_loss.backward()
                #         optimizer_1.step()   
                #     # sleep(1000)

                # d_loss_n, g_loss_n = cyc_con.forward_no_grad_adv(r_xn, 1, discriminator, gan_mode=gan_mode)
                # d_loss_p, g_loss_p = cyc_con.forward_no_grad_adv(r_xp, -1, discriminator, gan_mode=gan_mode) 
                neg_lp_loss, dtrans_bce_loss, cdtrans_loss, acc_d, grad_loss_d, cyc_grad_loss_d = cyc_con.forward_no_grad(r_xn, 1)    
                pos_lp_loss, ctrans_bce_loss, dctrans_loss, acc_c, grad_loss_c, cyc_grad_loss_c = cyc_con.forward_no_grad(r_xp, -1)
                # adv_d_loss_n_list.append(d_loss_n.cpu().numpy())
                # adv_d_loss_p_list.append(d_loss_p.cpu().numpy())
                # adv_g_loss_n_list.append(g_loss_n.cpu().numpy())
                # adv_g_loss_p_list.append(g_loss_p.cpu().numpy())
                cdtrans_losses.append(cdtrans_loss.cpu().numpy())
                dctrans_losses.append(dctrans_loss.cpu().numpy())
                ctrans_bce_losses.append(ctrans_bce_loss.cpu().numpy())
                dtrans_bce_losses.append(dtrans_bce_loss.cpu().numpy())
                c_acc_list.append(acc_c)
                d_acc_list.append(acc_d)  
                grad_losses_d.append(grad_loss_d) 
                grad_losses_c.append(grad_loss_c)  
                cyc_grad_losses_d.append(cyc_grad_loss_d) 
                cyc_grad_losses_c.append(cyc_grad_loss_c)           
                # for j in range(2):
                #     optimizer_5.zero_grad()
                #     temp_no_grad_list = ['trans_bce', 'trans']
                #     neg_lp_loss, dtrans_bce_loss, cdtrans_loss, acc_d = cyc_con.forward_no_grad(xn, 1, with_no_grad_list=temp_no_grad_list)
                #     neg_lp_loss = w_cw * neg_lp_loss
                #     neg_lp_loss.backward()
                #     optimizer_5.step()  


                # for j in range(1):
                #     optimizer_6.zero_grad()
                #     temp_no_grad_list = []
                #     pos_con_loss = cyc_con.forward_no_grad_con(xp, -1, with_no_grad_list=temp_no_grad_list)
                #     pos_con_loss = 1 * pos_con_loss
                #     pos_con_loss.backward()
                #     optimizer_6.step()  

                if verbose == True:
                    print('cur_step is:', step)
                    print(step, 'd_acc', acc_d, 'c_acc', acc_c, 'dc_loss', dctrans_loss.data.cpu().numpy(), 'cd_loss', cdtrans_loss.data.cpu().numpy())
                    print('d_lp:', neg_lp_loss.data.detach().cpu().numpy(), 'c_lp:',pos_lp_loss.data.detach().cpu().numpy(), 'bce_c_loss:', ctrans_bce_loss.data.cpu().numpy(), 'bce_d_loss:', dtrans_bce_loss.data.cpu().numpy()) 
                    # print('d_loss_p:', d_loss_p, 'g_loss_p:', g_loss_p, 'd_loss_n:', d_loss_n, 'g_loss_n:', g_loss_n)
                    print('grad_loss_c:', grad_loss_c, 'grad_loss_d:', grad_loss_d, 'cyc_grad_loss_c:', cyc_grad_loss_c, 'cyc_grad_loss_d:', cyc_grad_loss_d)
                    # print('pos con_loss:', pos_con_loss.data.detach().cpu().numpy(), 'neg con_loss:', neg_con_loss.data.detach().cpu().numpy())
                # if step % 50 == 0:
                if step % 50 == 0 and step > 0:
                    name = 'cdtrans_losses'
                    ut.save_image_pickel_by_name_cls(cdtrans_losses, name, step, path=IPMI_save + cyc_con.name + "/")                    
                    name = 'dctrans_losses'
                    ut.save_image_pickel_by_name_cls(dctrans_losses, name, step, path=IPMI_save + cyc_con.name + "/")
                    name = 'ctrans_bce_losses'
                    ut.save_image_pickel_by_name_cls(ctrans_bce_losses, name, step, path=IPMI_save + cyc_con.name + "/")
                    name = 'dtrans_bce_losses'
                    ut.save_image_pickel_by_name_cls(dtrans_bce_losses, name, step, path=IPMI_save + cyc_con.name + "/")
                    name = 'grad_losses_c'
                    ut.save_image_pickel_by_name_cls(cdtrans_losses, name, step, path=IPMI_save + cyc_con.name + "/")                    
                    name = 'grad_losses_d'
                    ut.save_image_pickel_by_name_cls(dctrans_losses, name, step, path=IPMI_save + cyc_con.name + "/")
                    name = 'cyc_grad_losses_c'
                    ut.save_image_pickel_by_name_cls(ctrans_bce_losses, name, step, path=IPMI_save + cyc_con.name + "/")
                    name = 'cyc_grad_losses_d'
                    ut.save_image_pickel_by_name_cls(dtrans_bce_losses, name, step, path=IPMI_save + cyc_con.name + "/")
                    # name = 'adv_g_loss_p'
                    # ut.save_image_pickel_by_name_cls(adv_g_loss_p_list, name, step, path=IPMI_save + cyc_con.name + "/")
                    # name = 'adv_g_loss_n'
                    # ut.save_image_pickel_by_name_cls(adv_g_loss_n_list, name, step, path=IPMI_save + cyc_con.name + "/")
                    # name = 'adv_d_loss_p'
                    # ut.save_image_pickel_by_name_cls(adv_d_loss_p_list, name, step, path=IPMI_save + cyc_con.name + "/")
                    # name = 'adv_d_loss_n'
                    # ut.save_image_pickel_by_name_cls(adv_d_loss_n_list, name, step, path=IPMI_save + cyc_con.name + "/")    
                    # sleep(10000)    

                # if step % 100 == 0 and step > 0:
                # # if step % 100 == 0:
                #     print(len(xp_list), step)
                #     name = 'xp'
                #     ut.save_image_pickel_by_name_cls(xp_list[-1], name, step, path=IPMI_save + cyc_con.name + "/")                    
                #     name = 's1'
                #     ut.save_image_pickel_by_name_cls(simulated_list_1[-1], name, step, path=IPMI_save + cyc_con.name + "/")
                #     name = 's2'
                #     ut.save_image_pickel_by_name_cls(simulated_list_2[-1], name, step, path=IPMI_save + cyc_con.name + "/")
                #     name = 'xn'
                #     ut.save_image_pickel_by_name_cls(xn_list[-1], name, step, path=IPMI_save + cyc_con.name + "/")
                #     name = 's3'
                #     ut.save_image_pickel_by_name_cls(simulated_list_3[-1], name, step, path=IPMI_save + cyc_con.name + "/")
                #     name = 's4'
                #     ut.save_image_pickel_by_name_cls(simulated_list_4[-1], name, step, path=IPMI_save + cyc_con.name + "/")
                #     name = 'lp_c'
                #     ut.save_image_pickel_by_name_cls(lp_c[-1], name, step, path=IPMI_save + cyc_con.name + "/")
                #     name = 'lp_d'
                #     ut.save_image_pickel_by_name_cls(lp_d[-1], name, step, path=IPMI_save + cyc_con.name + "/")
                #     # sleep(10000)
                #     # sleep(10000)
                #     # with open(IPMI_save + "s1.pkl", "wb") as fp:
                #     #     pkl.dump(simulated_list_1, fp)
                #     # with open(IPMI_save + "s2.pkl", "wb") as fp:
                #     #     pkl.dump(simulated_list_2, fp)
                #     # with open(IPMI_save + "s3.pkl", "wb") as fp:
                #     #     pkl.dump(simulated_list_3, fp)
                #     # with open(IPMI_save + "s4.pkl", "wb") as fp:
                #     #     pkl.dump(simulated_list_4, fp)
                step += 1
                


    elif pipeline == 'stay_in_fold':
        print(1)


def get_classified_result_image_given_fold_list(model, dataset_list, label_list):
    all_groundtruth_label_list = []
    all_predicted_label = []
    all_predict

    for j in range(fold):
        test_label_list = []
        label_loader = label[j]
        for batch_idx, [d] in enumerate(label_loader):
            test_label_list.append(d)
        all_groundtruth_label_list.append(test_label_list)
    folds = len(dataset_list)

    for cur_fold in range(folds):
        data_loader = dataset_list[i]
        label_list = all_groundtruth_label_list[cur_fold]
        for batch_idx, (xu) in enumerate(data_loader):
            xu = xu[0].to(device)
            l = label_list[batch_idx].float().to(device)
            # print(sum(l)/len(l))
            # optimizer.zero_grad()
            # if i == 1:
            fold_idx_list = []
            acc_list = []
            tot_list = []
            totlen = 0
            total = 0
            for k1 in range(fold - 1):
                loader = train_loader_list[k1]
                label_l = train_label_list[k1]
                acc, tot = test_fold(model, loader, label_l, device)
                total += acc * tot
                totlen += tot
                fold_idx = k1 + 1 if k1 < j else k1 + 2
                print('Batch ', fold_idx ," acc:", acc)
                fold_idx_list.append(fold_idx)
                acc_list.append(acc)
                tot_list.append(tot)

                path = "/scratch/users/zucks626/ADNI/IPMI/checkpoints/"
                save_dir = path + model.name + "/"
                print('Train Batch '," acc:",total/totlen)
                acc, tot = test_fold(model,test_loader,test_label_list,device)
                print('Test Batch acc:', acc)
                train_acc.append(total/totlen)
                test_acc.append(acc)

                # ut.writelog(save_dir +'log_'+str(j),i-1,acc_list,fold_idx_list,tot_list,total,totlen,acc,tot,j)
                # ut.writeacc(save_dir +'acc_'+str(j), i-1, acc_list, total/totlen, acc, fold_idx_list, j)

                # loss = cls(xu, l)   
                # loss_temp = loss.detach().cpu()
                # losses.append(loss_temp)       

def get_logits_from_loader(cls, cyc_con, data_loader, option):
    logits = []
    trans_logits = []
    cyc_back_logits = []
    device = cyc_con.device
    FLAG = option

    with torch.no_grad():
        for batch_idx, (x) in enumerate(data_loader):
            x = x[0].to(device)
            bs = x.shape[0]
            batch_size = bs
            if FLAG == 1:
                option_rec = torch.ones(batch_size, 1).to(device) * -1
                option_simu = torch.ones(batch_size, 1).to(device) * 1
            elif FLAG == -1:
                option_rec = torch.ones(batch_size, 1).to(device) * 1
                option_simu = torch.ones(batch_size, 1).to(device) * -1

            y = torch.ones(bs).to(device)
            _, _, _, _, logit = cls.classify_logit(x, y)
            logits.append(logit.cpu().numpy())
            vec_field = cyc_con.simulate(x, option_simu, mode='eval')
            # grad_loss = self.GRAD_Loss.loss(vec_field)
            trans_x = cyc_con.ST(x, vec_field)
            # trans_x = cyc_con.trans.eval()(x, option_simu)
            _, _, _, _, trans_logit = cls.classify_logit(trans_x, y)
            trans_logits.append(trans_logit.cpu().numpy())
            cyc_vec_field = cyc_con.simulate(trans_x, option_rec, mode='eval')
            # cyc_grad_loss = self.GRAD_Loss(cyc_vec_field)
            cyc_back_x = cyc_con.ST(trans_x, cyc_vec_field)
            # cyc_back_x = cyc_con.trans.eval()(trans_x, option_rec)
            _, _, _, _, cyc_back_logit = cls.classify_logit(cyc_back_x, y)
            cyc_back_logits.append(cyc_back_logit.cpu().numpy())
    return np.concatenate(logits), np.concatenate(trans_logits), np.concatenate(cyc_back_logits)

def get_test_logits_from_loader(cls, cyc_con, test_data_loader):
    n_logits = []
    n_trans_logits = []
    n_cyc_back_logits = []
    p_logits = []
    p_trans_logits = []
    p_cyc_back_logits = []
    device = cyc_con.device

    with torch.no_grad():
        for batch_idx, (x) in enumerate(test_data_loader):
            x = x[0].to(device)
            xn = x[:,0,:,:,:].unsqueeze(1)
            xp = x[:,1,:,:,:].unsqueeze(1)
            bs = x.shape[0]
            batch_size = bs
            option_pos2neg = torch.ones(batch_size, 1).to(device) * -1
            option_neg2pos = torch.ones(batch_size, 1).to(device) * 1
            y = torch.ones(bs).to(device)

            _, _, _, _, p_logit = cls.classify_logit(xp, y)
            p_logits.append(p_logit.cpu().numpy())
            vec_field_p = cyc_con.simulate(xp, option_pos2neg, mode='eval')
            # grad_loss = self.GRAD_Loss.loss(vec_field)
            trans_xp = cyc_con.ST(xp, vec_field_p)
            # trans_xp = cyc_con.trans.eval()(xp, option_pos2neg)
            _, _, _, _, p_trans_logit = cls.classify_logit(trans_xp, y)
            p_trans_logits.append(p_trans_logit.cpu().numpy())
            cyc_vec_field_p = cyc_con.simulate(trans_xp, option_neg2pos, mode='eval')
            # cyc_grad_loss = self.GRAD_Loss(cyc_vec_field)
            cyc_back_xp = cyc_con.ST(trans_xp, cyc_vec_field_p)
            # cyc_back_xp = cyc_con.trans.eval()(trans_xp, option_neg2pos)
            _, _, _, _, p_cyc_back_logit = cls.classify_logit(cyc_back_xp, y)
            p_cyc_back_logits.append(p_cyc_back_logit.cpu().numpy())

            _, _, _, _, n_logit = cls.classify_logit(xn, y)
            n_logits.append(n_logit.cpu().numpy())
            vec_field_n = cyc_con.simulate(xn, option_neg2pos, mode='eval')
            # grad_loss = self.GRAD_Loss.loss(vec_field)
            trans_xn = cyc_con.ST(xn, vec_field_n)
            # trans_xn = cyc_con.trans.eval()(xn, option_neg2pos)
            _, _, _, _, n_trans_logit = cls.classify_logit(trans_xn, y)
            n_trans_logits.append(n_trans_logit.cpu().numpy())
            cyc_vec_field_n = cyc_con.simulate(trans_xn, option_pos2neg, mode='eval')
            # cyc_grad_loss = self.GRAD_Loss(cyc_vec_field)
            cyc_back_xn = cyc_con.ST(trans_xn, cyc_vec_field_n)
            # cyc_back_xn = cyc_con.trans.eval()(trans_xn, option_pos2neg)
            _, _, _, _, n_cyc_back_logit = cls.classify_logit(cyc_back_xn, y)
            n_cyc_back_logits.append(n_cyc_back_logit.cpu().numpy())
    return np.concatenate(n_logits), np.concatenate(n_trans_logits), np.concatenate(n_cyc_back_logits), np.concatenate(p_logits), np.concatenate(p_trans_logits), np.concatenate(p_cyc_back_logits)


def main():
    cls = Classifier(name=Classifier_model_name, device=device)
    # train_cls(cls, test_loader_list, test_label_loader_list, test_loader_list, test_label_loader_list, device=device, tqdm=tqdm.tqdm, fold=5, iter_max=1010, iter_save=50, lr_decay_step=1500, schedule=False)

    # train_cls(cls, train_loader_list, train_label_loader_list, test_loader_list, test_label_loader_list, device=device, tqdm=tqdm.tqdm, fold=5, iter_max=1010, iter_save=25, lr_decay_step=300, schedule=False)
    # sleep(1000)
    global_step = 450
    ut.load_model_by_name(cls, global_step=global_step, device=device)
    # train_cls(cls)
    print('Load model!')
    # sleep(10000)

    # FLAG_COND_TYPE = 'Cond_sim'
    FLAG_COND_TYPE = 'Cond'
    trans = Convnet_SkipConnection(in_ch=1, out_ch=3, name='trans', device=device, z_dim=args.z, conv_type=FLAG_COND_TYPE)
    cyc_con = Cycle_Cons_3_Improvements(cls, trans, name=cyc_con_model_name, device=device)
    discriminator = Discriminator(name=Discriminator_model_name, device=device)
    pos_loader, neg_loader, r_data_pair, w_data_pair = prepare_train_data_based_on_trained_classifier(cls, train_loader_list, train_label_loader_list, device=device, pipeline='push_to_all', p_type='res_then_pred', BATCH_SIZE=args.BATCH_SIZE, shuffle=False, seed=2020)
    print(r_data_pair[0].shape)
    print(w_data_pair[0].shape)
    # sleep(10000)
    ut.load_model_by_name(cyc_con, global_step=1700, device=device)
    # ut.load_model_by_name(discriminator, global_step=100, device=device)
    print('Load Cyc_con model!')
    
    train_cyc_con(cyc_con, pos_loader, neg_loader, discriminator, r_data_pair, w_data_pair, iter_max=1501, verbose=True, optimizer_list=None, pipeline='push_to_all', BATCH_SIZE=args.BATCH_SIZE)


main()