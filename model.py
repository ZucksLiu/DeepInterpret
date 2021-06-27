import torch
from torch.nn import functional as F
import numpy as np
from torch import autograd, nn, optim
import utils as ut

class Classifier(nn.Module):
    def __init__(self, vae, z_dim, clses, requires_grad=True, label=1, name ="Classifier", y_dim=0,input_channel_size=1,device ='cpu'):
        super().__init__()
        self.vae = vae
        if requires_grad == False:
            for p in self.parameters():
                p.requires_grad = False
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.device =device
        self.name =name
        self.cls = clses
        self.Label = label
        # self.net = nn.Sequential(
        #     nn.Linear(z_dim, 1),
        #     nn.ReLU(),
        # )
        self.lambda_cos = 0.1
        self.lambda_ip = 2
        self.Linear1 = nn.Linear(z_dim,64)
        self.dropout = nn.Dropout(p=0.3)
        # self.weights = torch.nn.Parameter(torch.randn(1,1,z_dim,dtype=torch.float, requires_grad=requires_grad).to(device))
        # print(self.weights.norm())
        # self.weights.data = self.weights.data.detach() / (self.weights.data.detach().norm() + 1e-10)
        # nn.Module.register_parameter(self,'d_weights',self.weights)
        if self.cls== 2:
            self.Linear2 = nn.Linear(64,1).to(self.device)
        else:
            self.Linear2 = nn.Linear(z_dim,self.cls).to(self.device)
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.Sigmoid = nn.Sigmoid()
        torch.nn.init.kaiming_normal_(self.Linear1.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.Linear2.weight, a=0, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.xavier_uniform_(Linear1.weight)
        # torch.nn.init.xavier_uniform_(.weight)
        # self.Linear3 = nn.Linear(32,32).to(self.device)
        # self.Relu = nn.ReLU()
        # self.net =self.net.float() 
    def classify(self, x, y):
        z = self.vae.enc.encode_nkl(x)
        # h = F.relu(z)
        # h = self.dropout.eval()(h)
        h = self.Linear1(z)
        h = F.relu(h)

        h = self.Linear2(h)
        h = h.squeeze(dim=-1)        
        # loss = self.bce(input=h, target=x)
        pred = self.Sigmoid(h).detach()
        pred = [self.Label if pred[i] >=0.5 else 0 for i in range(len(pred))]
        # print(y)
        # print(pred)
        res = [1 if pred[i]==y[i] else 0 for i in range(len(pred))]
        acc = sum(res)/len(pred)
        # print('total:',len(pred),'acc for batch:', acc)
        return acc, len(pred), pred, res           
    def forward(self, x, y):
        z = self.vae.enc.encode_nkl(x)
        # h = F.relu(z)
        # h = self.dropout(h)
        h = self.Linear1(z)
        h = F.relu(h)
        
        h = self.Linear2(h)
        h = h.squeeze(dim=-1)
        loss = self.bce(input=h, target=y)
        pred = self.Sigmoid(h).detach()
        pred = [self.Label if pred[i] >=0.5 else 0 for i in range(len(pred))]
        print('total:',len(pred),'acc for batch:', sum([1 if pred[i]==y[i] else 0 for i in range(len(pred))])/len(pred))
        return loss.mean()

class Encoder(nn.Module):
    def __init__(self, z_dim, input_channel_size=1, device ='cpu'):
        super().__init__()
        self.z_dim = z_dim
        self.device =device
        self.net = nn.Sequential(
            nn.Conv3d(input_channel_size,16,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(2),
        )
        self.net =self.net.float().to(self.device)

        self.Linear1 = nn.Linear(1024,512).to(self.device)
        self.Linear2 = nn.Linear(1024,2*z_dim).to(self.device)
        self.dropout1 = nn.Dropout(p=0.5).to(self.device)
        self.dropout2 = nn.Dropout(p=0.5).to(self.device)
        self._initialize_weights()

    def _initialize_weights(self):
        cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m,nn.ConvTranspose3d):
                cnt += 1
                nn.init.xavier_uniform_(m.weight, gain=1)
        # print(cnt)

    def encode_kl(self, x):
        h = self.net(x.float())
        batch_size = h.shape[0]
        h = h.reshape(batch_size,-1)
        h = self.Linear2(h)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

    def forward(self, x):
        h = self.net(x.float())
        batch_size = h.shape[0]
        h = h.reshape(batch_size,-1)
        h = self.Linear1(h)
        h = F.relu(h)
        # h = F.tanh(h)
        return h

class Decoder(nn.Module):
    def __init__(self, z_dim, fp_dim=1024, output_channel_size=1, device ='cpu'):
        super().__init__()
        self.z_dim = z_dim
        self.device =device
        self.Linear1 = nn.Linear(z_dim, fp_dim).to(self.device)
        self.dropout1 = nn.Dropout(p=0.5)
        self.net = nn.Sequential(
            nn.ConvTranspose3d(16,16,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose3d(16,64,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose3d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose3d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose3d(16, output_channel_size, 3, padding=1),
        )
        self.net = self.net.float().to(device)
        self._initialize_weights()

    def _initialize_weights(self):
        cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                cnt += 1
                nn.init.xavier_uniform_(m.weight, gain=1)
        # print(cnt)

    def forward(self, z):
        h = self.Linear1(z)
        h = h.reshape(h.shape[0], 16, 4, 4, 4)
        return self.net(h)

    def decode_with_dense(self, z):
        h = self.Linear1(z)
        h = F.tanh(h)
        h = h.reshape(h.shape[0], 16, 4, 4, 4)
        return self.net(h)

class CD_Simu(nn.Module):
    def __init__(self, enc=None, dec=None, name='CD_Simu', z_dim=512, device='cpu'):
        super().__init__()
        self.device =device
        self.name = name
        if enc == None:
            self.enc = Encoder(z_dim = z_dim, device=device)
        else:
            self.enc = enc
        if dec == None:
            self.dec = Decoder(z_dim = z_dim, device=device)
        else:
            self.dec = dec
        self.enc = self.enc.float().to(device)
        self.dec = self.dec.float().to(device)
                
    def forward(self, x, verbose=False):
        batch_size = x.shape[0]
        shape = x.shape
        h = self.enc(x.float())
        rec_img = self.dec(h)
        return rec_img

class Cycle_Cons(nn.Module):
    def __init__(self, cls, ctrans, dtrans, name='Cyc_cons', requires_grad=False, device ='cpu'):
        super().__init__()
        self.device = device
        self.e_cls = cls
        self.name = name
        if requires_grad == False:
            for p in self.parameters():
                p.requires_grad = False
        self.ctrans = ctrans
        self.dtrans = dtrans
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.e_cls = self.e_cls.to(device)
        self.ctrans = self.ctrans.to(device)
        self.dtrans = self.dtrans.to(device)
               
    def forward_c(self, xp, verbose=False):
        batch_size = xp.shape[0]
        neg_y = torch.zeros(batch_size).to(self.device)
        pos_lp_loss = ut.mseloss(xp, self.dtrans(xp)).mean()
        simu_c = self.ctrans(xp)
        ctrans_bce_loss, acc_c = self.e_cls(simu_c, neg_y, verbose)
        dctrans_loss = ut.mseloss(xp, self.dtrans(simu_c)).mean()
        return pos_lp_loss, ctrans_bce_loss, dctrans_loss, acc_c

    def forward_c_no_grad(self, xp, verbose=False, with_no_grad_list=['pos_lp', 'dctrans', 'ctrans_bce']):
        batch_size = xp.shape[0]
        neg_y = torch.zeros(batch_size).to(self.device)
        if 'pos_lp' in with_no_grad_list:
            with torch.no_grad():
                pos_lp_loss = ut.mseloss(xp, self.dtrans(xp)).mean()
        else:
            pos_lp_loss = ut.mseloss(xp, self.dtrans(xp)).mean()

        if 'dctrans' not in with_no_grad_list or 'ctrans_bce' not in with_no_grad_list: 
            simu_c = self.ctrans(xp)
            if 'dctrans' in with_no_grad_list:
                with torch.no_grad():
                    dctrans_loss = ut.mseloss(xp, self.dtrans(simu_c)).mean()
            else:
                dctrans_loss = ut.mseloss(xp, self.dtrans(simu_c)).mean()
            if 'ctrans_bce' in with_no_grad_list:
                with torch.no_grad():
                    ctrans_bce_loss, acc_c = self.e_cls(simu_c, neg_y, verbose)
            else:
                ctrans_bce_loss, acc_c = self.e_cls(simu_c, neg_y, verbose)
        elif 'dctrans' in with_no_grad_list and 'ctrans_bce' in with_no_grad_list:
            with torch.no_grad():
                simu_c = self.ctrans(xp)
                ctrans_bce_loss, acc_c = self.e_cls(simu_c, neg_y, verbose)
                dctrans_loss = ut.mseloss(xp, self.dtrans(simu_c)).mean()

        return pos_lp_loss, ctrans_bce_loss, dctrans_loss, acc_c


    def forward_d(self, xn, verbose=False):
        batch_size = xn.shape[0]
        pos_y = torch.ones(batch_size).to(self.device)
        neg_lp_loss = ut.mseloss(xn, self.ctrans(xn)).mean()
        simu_d = self.dtrans(xn)
        print(simu_d.shape)
        dtrans_bce_loss, acc_d = self.e_cls(simu_d, pos_y, verbose)
        cdtrans_loss = ut.mseloss(xn, self.ctrans(simu_d)).mean()
        return neg_lp_loss, dtrans_bce_loss, cdtrans_loss, acc_d

    def forward_d_no_grad(self, xn, verbose=False, with_no_grad_list=['neg_lp', 'cdtrans', 'dtrans_bce']):
        batch_size = xn.shape[0]
        pos_y = torch.ones(batch_size).to(self.device)
        if 'neg_lp' in with_no_grad_list:
            with torch.no_grad():
                neg_lp_loss = ut.mseloss(xn, self.ctrans(xn)).mean()
        else:
            neg_lp_loss = ut.mseloss(xn, self.ctrans(xn)).mean()

        if 'cdtrans' not in with_no_grad_list or 'dtrans_bce' not in with_no_grad_list: 
            simu_d = self.dtrans(xn)
            if 'cdtrans' in with_no_grad_list:
                with torch.no_grad():
                    cdtrans_loss = ut.mseloss(xn, self.ctrans(simu_d)).mean()
            else:
                cdtrans_loss = ut.mseloss(xn, self.ctrans(simu_d)).mean()
            if 'dtrans_bce' in with_no_grad_list:
                with torch.no_grad():
                    dtrans_bce_loss, acc_d = self.e_cls(simu_d, pos_y, verbose)
            else:
                dtrans_bce_loss, acc_d = self.e_cls(simu_d, pos_y, verbose)
        elif 'cdtrans' in with_no_grad_list and 'dtrans_bce' in with_no_grad_list:
            with torch.no_grad():
                simu_d = self.dtrans(xn)
                dtrans_bce_loss, acc_d = self.e_cls(simu_d, pos_y, verbose)
                cdtrans_loss = ut.mseloss(xn, self.ctrans(simu_d)).mean()

        return neg_lp_loss, dtrans_bce_loss, cdtrans_loss, acc_d

    def forward_all(self, xp, xn, verbose=False):
        batch_size = xp.shape[0]
        neg_y = torch.zeros(batch_size)
        pos_y = torch.ones(batch_size)
        pos_lp_loss = ut.mseloss(xp, self.dtrans(xp)).mean()
        neg_lp_loss = ut.mseloss(xn, self.ctrans(xn)).mean()
        simu_d = self.dtrans(xn)
        simu_c = self.ctrans(xp)
        ctrans_bce_loss,acc_c = self.e_cls(simu_c, neg_y, verbose)
        dtrans_bce_loss,acc_d = self.e_cls(simu_d, pos_y, verbose)
        cdtrans_loss = ut.mseloss(xn, self.ctrans(simu_d)).mean()
        dctrans_loss = ut.mseloss(xp, self.dtrans(simu_c)).mean()
        return pos_lp_loss, neg_lp_loss, ctrans_bce_loss, dtrans_bce_loss, cdtrans_loss, dctrans_loss, [acc_c, acc_d] 
        
    def loss_all(self, xp, xn, verbose=False):
        batch_size = xp.shape[0]
        neg_y = torch.zeros(batch_size)
        pos_y = torch.ones(batch_size)
        pos_lp_loss = ut.mseloss(xp, self.dtrans(xp)).mean()
        neg_lp_loss = ut.mseloss(xn, self.ctrans(xn)).mean()
        simu_d = self.dtrans(xn)
        simu_c = self.ctrans(xp)
        ctrans_bce_loss, acc_c = self.e_cls(simu_c, neg_y, verbose)
        dtrans_bce_loss, acc_d = self.e_cls(simu_d, pos_y, verbose)
        cdtrans_loss = ut.mseloss(xn, self.ctrans(simu_d)).mean()
        dctrans_loss = ut.mseloss(xp, self.dtrans(simu_c)).mean()
        print('shape of dctrans_loss:', dctrans_loss.shape)
        if verbose == True:
            print('cd_loss:', cdtrans_loss, 'dc_loss:', dctrans_loss, 'c_simu_loss:', ctrans_bce_loss, 'd_simu_loss:', dtrans_bce_loss)
        loss = 0.01 * pos_lp_loss + 0.01 * neg_lp_loss + ctrans_bce_loss + dtrans_bce_loss + cdtrans_loss + dctrans_loss
        return loss, acc_c/batch_size, acc_d/batch_size

class Classifier(nn.Module):
    def __init__(self, input_channel_size=1, label=1, name='Classifier', requires_grad=True, device ='cpu'):
        super().__init__()
        self.device = device
        self.name = name
        self.net = nn.Sequential(
            nn.Conv3d(input_channel_size,16,3,padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # nn.Dropout3d(p=0.1),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # nn.Dropout3d(p=0.1),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # nn.Dropout3d(p=0.1),
            nn.Conv3d(64, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
        ).to(device)
        # self.Linear = nn.Linear(1024, 128).to(self.device)
        self.Linear1 = nn.Linear(1024, 64).to(device)
        self.Linear2 = nn.Linear(64, 1).to(device)
        self.Sigmoid = nn.Sigmoid()
        self.net = self.net.float().to(device)
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.Label = label
        self.dropout = nn.Dropout(p=0.05)
        self._initialize_weights()
        torch.nn.init.kaiming_normal_(self.Linear1.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.Linear2.weight, a=0, mode='fan_in', nonlinearity='relu')
        for p in self.parameters():
            p.requires_grad = True
        # self.dropout = nn.Dropout(p=0.3)

    def _initialize_weights(self):
        cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m,nn.ConvTranspose3d):
                cnt += 1
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0.0)
        # print(cnt)

    def classify(self, x, y):
        h = self.net.eval()(x.float())
        batch_size = h.shape[0]
        h = h.reshape(batch_size, -1)
        h = self.dropout.eval()(h)
        h = self.Linear1(h)
        h = F.relu(h)
        h = self.Linear2(h)
        h = h.squeeze(dim=-1)
        pred = self.Sigmoid(h).detach()
        pred = [self.Label if pred[i] >= 0.5 else 0 for i in range(len(pred))]
        res = [1 if pred[i]==y[i] else 0 for i in range(len(pred))]
        acc = sum(res) / len(pred)
        return acc, len(pred), pred, res  

    def logits(self, x):
        h = self.net.eval()(x.float())
        batch_size = h.shape[0]
        h = h.reshape(batch_size, -1)
        h = self.dropout.eval()(h)
        h = self.Linear1(h)
        h = F.relu(h)
        h = self.Linear2(h)
        h = h.squeeze(dim=-1)
        logits = h
        return logits 

    def classify_logit(self, x, y):
        h = self.net.eval()(x.float())
        batch_size = h.shape[0]
        h = h.reshape(batch_size, -1)
        h = self.dropout.eval()(h)
        h = self.Linear1(h)
        h = F.relu(h)
        h = self.Linear2(h)
        h = h.squeeze(dim=-1)
        logits = h
        sigmoid_logits = self.Sigmoid(h)
        pred = [self.Label if sigmoid_logits[i] >= 0.5 else 0 for i in range(len(sigmoid_logits))]
        res = [1 if pred[i]==y[i] else 0 for i in range(len(pred))]
        acc = sum(res) / len(pred)
        return acc, len(pred), pred, res, logits 

    def classify_logit_after_sigmoid(self, x, y):
        h = self.net.eval()(x.float())
        batch_size = h.shape[0]
        h = h.reshape(batch_size, -1)
        h = self.dropout.eval()(h)
        h = self.Linear1(h)
        h = F.relu(h)
        h = self.Linear2(h)
        h = h.squeeze(dim=-1)
        logits = self.Sigmoid(h)
        pred = [self.Label if logits[i] >= 0.5 else 0 for i in range(len(logits))]
        res = [1 if pred[i]==y[i] else 0 for i in range(len(pred))]
        acc = sum(res) / len(pred)
        return acc, len(pred), pred, res, logits 

    def forward(self, x, y, forward_type='eval', verbose=False):
        if forward_type == 'train':
            h = self.net.train()(x.float())
        elif forward_type == 'eval':
            h = self.net.eval()(x.float())
        batch_size = h.shape[0]
        h = h.reshape(batch_size, -1)
        if forward_type == 'train':
            h = self.dropout.train()(h)
        elif forward_type == 'eval':
            h = self.dropout.eval()(h)
        h = self.Linear1(h)
        h = F.relu(h)
        h = self.Linear2(h)
        h = h.squeeze(dim=-1)
        # print(h.shape)
        # y.requires_grad = True
        loss = self.bce(input=h, target=y)
        pred = self.Sigmoid(h).detach()
        # print(pred)
        pred = [self.Label if pred[i] >=0.5 else 0 for i in range(len(pred))]
        acc_num = sum([1 if pred[i]==y[i] else 0 for i in range(len(pred))])
        if verbose == True:
            print('total:', len(pred), 'acc for batch:', acc_num/len(pred))
        # print('mean:', loss.mean())
        if forward_type == 'train':
            return loss.mean()
        elif forward_type == 'eval':
            return loss.mean(), acc_num/len(pred)


class Discriminator(nn.Module):
    def __init__(self, input_channel_size=1, label=1, name='Classifier', requires_grad=True, device ='cpu'):
        super().__init__()
        self.device = device
        self.name = name
        self.net = nn.Sequential(
            nn.Conv3d(input_channel_size,16,3,padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # nn.Dropout3d(p=0.1),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # nn.Dropout3d(p=0.1),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # nn.Dropout3d(p=0.1),
            nn.Conv3d(64, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2),
        ).to(device)
        # self.Linear = nn.Linear(1024, 128).to(self.device)
        self.Linear1 = nn.Linear(512, 64).to(device)
        self.Linear2 = nn.Linear(64, 1).to(device)
        self.Sigmoid = nn.Sigmoid()
        self.net = self.net.float().to(device)
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.Label = label
        self.dropout = nn.Dropout(p=0.05)
        self._initialize_weights()
        torch.nn.init.kaiming_normal_(self.Linear1.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.Linear2.weight, a=0, mode='fan_in', nonlinearity='relu')
        for p in self.parameters():
            p.requires_grad = True
        # self.dropout = nn.Dropout(p=0.3)

    def _initialize_weights(self):
        cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m,nn.ConvTranspose3d):
                cnt += 1
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0.0)
        # print(cnt)

    def classify(self, x, y):
        h = self.net.eval()(x.float())
        batch_size = h.shape[0]
        h = h.reshape(batch_size, -1)
        h = self.dropout.eval()(h)
        h = self.Linear1(h)
        h = F.relu(h)
        h = self.Linear2(h)
        h = h.squeeze(dim=-1)
        pred = self.Sigmoid(h).detach()
        pred = [self.Label if pred[i] >= 0.5 else 0 for i in range(len(pred))]
        res = [1 if pred[i]==y[i] else 0 for i in range(len(pred))]
        acc = sum(res) / len(pred)
        return acc, len(pred), pred, res  

    def logits(self, x, mode='train'):
        if mode == 'train':
            h = self.net(x.float())
        elif mode == 'eval':
            h = self.net.eval()(x.float())
        batch_size = h.shape[0]
        h = h.reshape(batch_size, -1)
        if mode == 'train':
            h = self.dropout(h)
        elif mode == 'eval':
            h = self.dropout.eval()(h)
        h = self.Linear1(h)
        h = F.relu(h)
        h = self.Linear2(h)
        h = h.squeeze(dim=-1)
        logits = h
        return logits 

    def classify_logit(self, x, y):
        h = self.net.eval()(x.float())
        batch_size = h.shape[0]
        h = h.reshape(batch_size, -1)
        h = self.dropout.eval()(h)
        h = self.Linear1(h)
        h = F.relu(h)
        h = self.Linear2(h)
        h = h.squeeze(dim=-1)
        logits = h
        sigmoid_logits = self.Sigmoid(h)
        pred = [self.Label if sigmoid_logits[i] >= 0.5 else 0 for i in range(len(sigmoid_logits))]
        res = [1 if pred[i]==y[i] else 0 for i in range(len(pred))]
        acc = sum(res) / len(pred)
        return acc, len(pred), pred, res, logits 

    def classify_logit_after_sigmoid(self, x, y):
        h = self.net.eval()(x.float())
        batch_size = h.shape[0]
        h = h.reshape(batch_size, -1)
        h = self.dropout.eval()(h)
        h = self.Linear1(h)
        h = F.relu(h)
        h = self.Linear2(h)
        h = h.squeeze(dim=-1)
        logits = self.Sigmoid(h)
        pred = [self.Label if logits[i] >= 0.5 else 0 for i in range(len(logits))]
        res = [1 if pred[i]==y[i] else 0 for i in range(len(pred))]
        acc = sum(res) / len(pred)
        return acc, len(pred), pred, res, logits 

    def forward(self, x, y, forward_type='eval', verbose=False):
        if forward_type == 'train':
            h = self.net(x.float())
        elif forward_type == 'eval':
            h = self.net.eval()(x.float())
        batch_size = h.shape[0]
        h = h.reshape(batch_size, -1)
        if forward_type == 'train':
            h = self.dropout(h)
        elif forward_type == 'eval':
            h = self.dropout.eval()(h)
        h = self.Linear1(h)
        h = F.relu(h)
        h = self.Linear2(h)
        h = h.squeeze(dim=-1)
        # print(h.shape)
        # y.requires_grad = True
        loss = self.bce(input=h, target=y)
        pred = self.Sigmoid(h).detach()
        # print(pred)
        pred = [self.Label if pred[i] >=0.5 else 0 for i in range(len(pred))]
        acc_num = sum([1 if pred[i]==y[i] else 0 for i in range(len(pred))])
        if verbose == True:
            print('total:', len(pred), 'acc for batch:', acc_num/len(pred))
        # print('mean:', loss.mean())
        if forward_type == 'train':
            return loss.mean()
        elif forward_type == 'eval':
            return loss.mean(), acc_num/len(pred)
