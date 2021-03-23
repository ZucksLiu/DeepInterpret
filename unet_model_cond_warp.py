import torch
from torch.nn import functional as F
import numpy as np
from torch import autograd, nn, optim
import utils as ut
from CondConv import CondConv3D
from CondConv_simple import CondConv3D_sim

FLAG_COND_TYPE = 'Cond_sim'

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

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, device='cpu', mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).to(device)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) 
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) 
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) 

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

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

class Convnet_SkipConnection(nn.Module):
    def __init__(self, in_ch, out_ch, gpu_ids=[], name='SkipCon_Simu', device='cpu', z_dim=512, fp_dim=1024, conv_type='Cond'):
        super(Convnet_SkipConnection, self).__init__()
        self.z_dim = z_dim
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.input = bare_conv(in_ch, 16).to(self.device)     # (bs, 16, 64, 64, 64)
        self.down1 = single_conv_down(16, 32).to(self.device) # (bs, 32, 32, 32, 32)
        self.down2 = single_conv_down(32, 64).to(self.device) # (bs, 64, 16, 16, 16)
        self.down3 = single_conv_down(64, 16).to(self.device) # (bs, 16, 8, 8, 8)
        self.down4 = single_conv_down(16, 16).to(self.device) # (bs, 16, 4, 4, 4)

        self.up_input = bare_conv(16, 16).to(self.device) # (bs, 16, 4, 4, 4)
        self.up1_noconv = noconv_up(16).to(self.device)   # (bs, 16, 8, 8, 8)
        self.up2 = single_conv_up(16, 64).to(self.device) # (bs, 64, 16, 16, 16)
        self.up3 = single_conv_up(64, 32).to(self.device) # (bs, 32, 32, 32, 32)
        self.up4 = single_conv_up(32, 16).to(self.device) # (bs, 16, 64, 64, 64)
        self.output = bare_conv(16, out_ch).to(self.device)

        self.in_Linear1 = nn.Linear(1024, z_dim).to(self.device)
        self.in_Linear2 = nn.Linear(1024, 2 * z_dim).to(self.device)
        self.in_dropout1 = nn.Dropout(p=0.5).to(self.device)
        self.in_dropout2 = nn.Dropout(p=0.5).to(self.device)

        self.out_Linear1 = nn.Linear(z_dim + 1, fp_dim).to(self.device)
        self.out_dropout1 = nn.Dropout(p=0.5).to(self.device)
    
    def forward(self, x, option):
        d0 = self.input(x.float(), option)
        d1 = self.down1(d0, option)
        d2 = self.down2(d1, option)
        d3 = self.down3(d2, option)
        d4 = self.down4(d3, option)
        batch_size = d4.shape[0]
        z1 = d4.reshape(batch_size, -1)
        z = self.in_Linear1(z1)
        z = torch.cat([z, option], axis=1)
        z = F.relu(z)
        z = self.out_Linear1(z)
        z = z.reshape(batch_size, 16, 4, 4, 4)
        u4 = self.up_input(z, option) + d4
        u3 = self.up1_noconv(u4) + d3
        u2 = self.up2(u3, option) + d2
        u1 = self.up3(u2, option) + d1
        u0 = self.up4(u1, option) + d0
        output = self.output(u0, option)
        return output


class double_conv(nn.Module):
    ''' Conv => Batch_Norm => ReLU => Conv2d => Batch_Norm => ReLU
    '''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv.apply(self.init_weights)
    
    def forward(self, x):
        x = self.conv(x)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            init.xavier_normal(m.weight)
            init.constant(m.bias,0)


class single_conv_down(nn.Module):
    '''  ReLU => Batch_Norm => MaxPool3d => Conv 
    '''
    def __init__(self, in_ch, out_ch, num_experts=3, embeddings=1, dropout_rate=0.1, conv_type='Cond'):
        super(single_conv_down, self).__init__()
        if conv_type == 'Cond':
            self.feed = nn.Sequential(
                nn.BatchNorm3d(in_ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool3d(2),
            )
            self.conv = CondConv3D(in_ch, out_ch, 3, padding=1, embeddings=embeddings, num_experts=num_experts, dropout_rate=dropout_rate)
        elif conv_type == 'Cond_sim':
            self.feed = nn.Sequential(
                nn.BatchNorm3d(in_ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool3d(2),
            )
            self.conv = CondConv3D_sim(in_ch, out_ch, 3, padding=1, embeddings=embeddings, num_experts=num_experts, dropout_rate=dropout_rate)                
        # self.conv.apply(self.init_weights)
    
    def forward(self, x, embedding):
        x = self.feed(x)
        x = self.conv(x, embedding)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv3d:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias,0)


class single_conv_up(nn.Module):
    '''  Conv => ReLU => Batch_Norm => ReLU => Upsampling
    '''
    def __init__(self, in_ch, out_ch, num_experts=3, embeddings=1, dropout_rate=0.1, conv_type='Cond'):
        super(single_conv_up, self).__init__()
        if conv_type == 'Cond':
            self.conv = CondConv3D(in_ch, out_ch, 3, padding=1, embeddings=embeddings, num_experts=num_experts, dropout_rate=dropout_rate)
            self.feed = nn.Sequential(
                
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Upsample(scale_factor=2),   
            )        
        elif conv_type == 'Cond_sim':
            self.conv = CondConv3D_sim(in_ch, out_ch, 3, padding=1, embeddings=embeddings, num_experts=num_experts, dropout_rate=dropout_rate)
            self.feed = nn.Sequential(
                
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Upsample(scale_factor=2),   
            )        
        # self.conv.apply(self.init_weights)
    
    def forward(self, x, embedding):
        x = self.conv(x, embedding)
        x = self.feed(x)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv3d:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias,0)


class bare_conv(nn.Module):
    ''' bare conv layer
        as input, let input 1 channels image to 16 channels
        as output, let 16 channels to output 1 channels image
    '''
    def __init__(self, in_ch, out_ch, num_experts=3, embeddings=1, dropout_rate=0.1, conv_type='Cond'):
        super(bare_conv, self).__init__()
        if conv_type == 'Cond':
            self.conv = CondConv3D(in_ch, out_ch, 3, padding=1, embeddings=embeddings, num_experts=num_experts, dropout_rate=dropout_rate)
        elif conv_type == 'Cond_sim':
            self.conv = CondConv3D_sim(in_ch, out_ch, 3, padding=1, embeddings=embeddings, num_experts=num_experts, dropout_rate=dropout_rate)

    def forward(self, x, embedding):
        x = self.conv(x, embedding)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv3d:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias,0)


class noconv_up(nn.Module):
    ''' before output image
        ReLU => BN => Upsample
    '''
    def __init__(self, in_ch):
        super(noconv_up, self).__init__()
        self.act_bn_mp = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2),
        )
    def forward(self, x):
        x = self.act_bn_mp(x)
        return x




class Cycle_Cons_3_Improvements(nn.Module):
    def __init__(self, cls, trans, name='Cyc_cons', requires_grad=False, device='cpu', grad_penalty='l1', weight_grad=0.001, b=10):
        super().__init__()
        self.device = device
        self.e_cls = cls
        self.name = name
        if requires_grad == False:
            for p in self.parameters():
                p.requires_grad = False
        self.e_cls.eval()
        self.trans = trans
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.e_cls = self.e_cls.to(device)
        self.trans = self.trans.to(device)
        self.b = b
        self.GRAD_Loss = Grad(grad_penalty, weight_grad)
        self.SpatialTransformer = SpatialTransformer(size=np.array([64, 64, 64]), device=device).to(device)

    def simulate(self, img, option, mode='train'):
        if mode == 'train':
            vec_field = self.trans.train()(img, option)
        elif mode == 'eval':
            vec_field = self.trans.eval()(img, option)
        return vec_field

    def ST(self, src, flow):
        return self.SpatialTransformer(src, flow)

    def Generate(self, img, option, mode='eval', requires_grad=False):
        if requires_grad == False:
            with torch.no_grad():
                vec_field = self.simulate(img, option, mode)
                simulated_img = self.ST(img, vec_field)
        else:
            vec_field = self.simulate(img, option, mode)
            simulated_img = self.ST(img, vec_field)
        return vec_field, simulated_img

    def shift_logit(self, logit_orig, logit_simu, option_simu, b):
        # logit_orig: a tensor with_no_grad
        # logit_simu: a tensor that needs to be optimized
        return F.relu((logit_orig - logit_simu) * option_simu.reshape(-1) + b)

    def forward_no_grad(self, x, option, verbose=False, with_no_grad_list=['lp', 'trans', 'trans_bce'], b=12.5, query_loss='shift'):
        # option: (neg -> pos) 0 -> 1: 1, (pos -> neg) 1 -> 0: -1
        batch_size = x.shape[0]
        FLAG = option
        if FLAG == 1:
            orig_y = torch.zeros(batch_size).to(self.device)
            simu_y = torch.ones(batch_size).to(self.device)
            option_rec = torch.ones(batch_size, 1).to(self.device) * -1
            option_simu = torch.ones(batch_size, 1).to(self.device) * 1
        elif FLAG == -1:
            orig_y = torch.ones(batch_size).to(self.device)
            simu_y = torch.zeros(batch_size).to(self.device)
            option_rec = torch.ones(batch_size, 1).to(self.device) * 1
            option_simu = torch.ones(batch_size, 1).to(self.device) * -1
        with torch.no_grad():
            _, _, _, _, logit_orig = self.e_cls.classify_logit(x, orig_y)
        if verbose == True:
            print('logit_orig:', logit_orig)

        if 'lp' in with_no_grad_list:
            with torch.no_grad():
                vec_field = self.simulate(x, option_simu, mode='eval')
                grad_loss = self.GRAD_Loss.loss(vec_field)
                simu_img = self.ST(x, vec_field)
                lp_loss = ut.mseloss(x, simu_img).mean()
                # lp_loss = ut.mseloss(x, self.trans.eval()(x, option_rec)).mean()
                
        else:
            vec_field = self.simulate(x, option_simu, mode='train')
            grad_loss = self.GRAD_Loss.loss(vec_field)
            simu_img = self.ST(x, vec_field)
            lp_loss = ut.mseloss(x, simu_img).mean()
            # lp_loss = ut.mseloss(x, self.trans(x, option_rec)).mean()
        if verbose == True:
            print('lp_loss:', lp_loss)
        cyc_grad_loss = 0
        if 'trans' not in with_no_grad_list or 'trans_bce' not in with_no_grad_list: 
            vec_field = self.simulate(x, option_simu, mode='train')
            grad_loss = self.GRAD_Loss.loss(vec_field)
            simu_img = self.ST(x, vec_field)
            # simu_img = self.trans(x, option_simu)
            if 'trans' in with_no_grad_list:
                with torch.no_grad():
                    cyc_vec_field = self.simulate(simu_img, option_rec, mode='eval')
                    cyc_grad_loss = self.GRAD_Loss.loss(cyc_vec_field)
                    cyc_back_img = self.ST(simu_img, cyc_vec_field)
                    trans_loss = ut.mseloss(x, cyc_back_img).mean()
                    # trans_loss = ut.mseloss(x, self.trans.eval()(simu_img, option_rec)).mean()
            else:
                cyc_vec_field = self.simulate(simu_img, option_rec, mode='train')
                cyc_grad_loss = self.GRAD_Loss.loss(cyc_vec_field)
                cyc_back_img = self.ST(simu_img, cyc_vec_field)
                trans_loss = ut.mseloss(x, cyc_back_img).mean()
                # trans_loss = ut.mseloss(x, self.trans(simu_img, option_rec)).mean()
            if 'trans_bce' in with_no_grad_list:
                with torch.no_grad():
                    if query_loss == 'shift':
                        acc, _, _, _, logit_simu = self.e_cls.classify_logit(simu_img, simu_y)
                        shift_bce_loss = self.shift_logit(logit_orig, logit_simu, option_simu, b)
                        trans_bce_loss = shift_bce_loss.mean()
                    elif query_loss == 'bce':
                        # print('yes!')
                        trans_bce_loss, acc = self.e_cls(simu_img, simu_y, forward_type='eval', verbose=verbose)

            else:
                if query_loss == 'shift':
                    acc, _, _, _, logit_simu = self.e_cls.classify_logit(simu_img, simu_y)
                    shift_bce_loss = self.shift_logit(logit_orig, logit_simu, option_simu, b)
                    trans_bce_loss = shift_bce_loss.mean()
                elif query_loss == 'bce':
                    # print('yesyes')
                    trans_bce_loss, acc = self.e_cls(simu_img, simu_y, forward_type='eval', verbose=verbose)

                
        elif 'trans' in with_no_grad_list and 'trans_bce' in with_no_grad_list:
            with torch.no_grad():
                vec_field = self.simulate(x, option_simu, mode='eval')
                grad_loss = self.GRAD_Loss.loss(vec_field)
                simu_img = self.ST(x, vec_field)
                # simu_img = self.trans.eval()(x, option_simu)
                cyc_vec_field = self.simulate(simu_img, option_rec, mode='eval')
                cyc_grad_loss = self.GRAD_Loss.loss(cyc_vec_field)
                cyc_back_img = self.ST(simu_img, cyc_vec_field)  
                trans_loss = ut.mseloss(x, cyc_back_img).mean()              
                # trans_loss = ut.mseloss(x, self.trans.eval()(simu_img, option_rec)).mean()
                if query_loss == 'shift':
                    acc, _, _, _, logit_simu = self.e_cls.classify_logit(simu_img, simu_y)
                    shift_bce_loss = self.shift_logit(logit_orig, logit_simu, option_simu, b)
                    if verbose == True:
                        print('shift_bce_loss:', shift_bce_loss)
                    trans_bce_loss = shift_bce_loss.mean()
                elif query_loss == 'bce':
                    # print('yesyesyes!')
                    trans_bce_loss, acc = self.e_cls(simu_img, simu_y, forward_type='eval', verbose=verbose)

        if verbose == True:
            print('acc:', acc)
            print('trans_bce_loss:', trans_bce_loss)
        return lp_loss, trans_bce_loss, trans_loss, acc, grad_loss, cyc_grad_loss

    def forward_no_grad_adv(self, x, option, d, verbose=False, with_no_grad_list=['adv'], gan_mode='non_saturate'):
        # option: (neg -> pos) 0 -> 1: 1, (pos -> neg) 1 -> 0: -1
        batch_size = x.shape[0]
        FLAG = option
        if FLAG == 1:
            orig_y = torch.zeros(batch_size).to(self.device)
            simu_y = torch.ones(batch_size).to(self.device)
            option_rec = torch.ones(batch_size, 1).to(self.device) * -1
            option_simu = torch.ones(batch_size, 1).to(self.device) * 1
        elif FLAG == -1:
            orig_y = torch.ones(batch_size).to(self.device)
            simu_y = torch.zeros(batch_size).to(self.device)
            option_rec = torch.ones(batch_size, 1).to(self.device) * 1
            option_simu = torch.ones(batch_size, 1).to(self.device) * -1
        # with torch.no_grad():
        #     _, _, _, _, logit_orig = self.e_cls.classify_logit(x, orig_y)
        # if verbose == True:
        #     print('logit_orig:', logit_orig)
        if 'adv' in with_no_grad_list:
            with torch.no_grad():
                vec_field = self.simulate(x, option_simu, mode='eval')
                grad_loss = self.GRAD_Loss.loss(vec_field)
                simu_img = self.ST(x, vec_field)
                # simu_img = self.trans.eval()(x, option_simu)
                d_real = d.logits(x, mode='eval')
                d_fake = d.logits(simu_img, mode='eval')
                if gan_mode == 'non_saturate':
                    d_loss = (-F.logsigmoid(d_real).mean() - F.logsigmoid(-d_fake).mean()) / 2
                    g_loss = -F.logsigmoid(d_fake).mean() + grad_loss
                elif gan_mode == 'wgan_gp':
                    alpha = torch.rand(batch_size, 1, 1, 1, 1, device=self.device)
                    x_r = alpha * simu_img + (1 - alpha) * x
                    d_r = d(x_r)
                    grad = torch.autograd.grad(d_r.sum(), x_r, create_graph=True)
                    grad_norm = grad[0].reshape(batch_size, -1).norm(dim=1)
                    d_loss = (d_fake - d_real).mean() + 10 * ((grad_norm -1)**2).mean()
                    g_loss = -d_fake.mean()                
        else:
            vec_field = self.simulate(x, option_simu, mode='train')
            grad_loss = self.GRAD_Loss.loss(vec_field)
            simu_img = self.ST(x, vec_field)
            # simu_img = self.trans(x, option_simu)
            d_real = d.logits(x, mode='train')
            d_fake = d.logits(simu_img, mode='train')
            if gan_mode == 'non_saturate':
                d_loss = (-F.logsigmoid(d_real).mean() - F.logsigmoid(-d_fake).mean()) / 2
                g_loss = -F.logsigmoid(d_fake).mean() + grad_loss
            elif gan_mode == 'wgan_gp':
                alpha = torch.rand(batch_size, 1, 1, 1, 1, device=self.device)
                x_r = alpha * simu_img + (1 - alpha) * x
                d_r = d(x_r)
                grad = torch.autograd.grad(d_r.sum(), x_r, create_graph=True)
                grad_norm = grad[0].reshape(batch_size, -1).norm(dim=1)
                d_loss = (d_fake - d_real).mean() + 10 * ((grad_norm -1)**2).mean()
                g_loss = -d_fake.mean()
        return d_loss, g_loss

    def forward_no_grad_adv_random(self, x, r_x, option, d, verbose=False, with_no_grad_list=['adv'], gan_mode='non_saturate'):
        # option: (neg -> pos) 0 -> 1: 1, (pos -> neg) 1 -> 0: -1
        batch_size = x.shape[0]
        FLAG = option
        if FLAG == 1:
            orig_y = torch.zeros(batch_size).to(self.device)
            simu_y = torch.ones(batch_size).to(self.device)
            option_rec = torch.ones(batch_size, 1).to(self.device) * -1
            option_simu = torch.ones(batch_size, 1).to(self.device) * 1
        elif FLAG == -1:
            orig_y = torch.ones(batch_size).to(self.device)
            simu_y = torch.zeros(batch_size).to(self.device)
            option_rec = torch.ones(batch_size, 1).to(self.device) * 1
            option_simu = torch.ones(batch_size, 1).to(self.device) * -1
        # with torch.no_grad():
        #     _, _, _, _, logit_orig = self.e_cls.classify_logit(x, orig_y)
        # if verbose == True:
        #     print('logit_orig:', logit_orig)
        if 'adv' in with_no_grad_list:
            with torch.no_grad():
                vec_field = self.simulate(r_x, option_simu, mode='eval')
                grad_loss = self.GRAD_Loss.loss(vec_field)
                simu_img = self.ST(r_x, vec_field)
                # simu_img = self.trans.eval()(r_x, option_simu)
                d_real = d.logits(x, mode='eval')
                d_fake = d.logits(simu_img, mode='eval')
                if gan_mode == 'non_saturate':
                    d_loss = (-F.logsigmoid(d_real).mean() - F.logsigmoid(-d_fake).mean()) / 2
                    g_loss = -F.logsigmoid(d_fake).mean() + grad_loss
                elif gan_mode == 'wgan_gp':
                    # alpha = torch.rand(batch_size, 1, 1, 1, 1, device=self.device)
                    # x_r = alpha * simu_img + (1 - alpha) * x
                    # d_r = d.logits(x_r)
                    # grad = torch.autograd.grad(d_r.sum(), x_r, create_graph=True)
                    # grad_norm = grad[0].reshape(batch_size, -1).norm(dim=1)
                    d_loss = (d_fake - d_real).mean() #+ 10 * ((grad_norm -1)**2).mean()
                    g_loss = -d_fake.mean()                
        else:
            vec_field = self.simulate(r_x, option_simu, mode='train')
            grad_loss = self.GRAD_Loss.loss(vec_field)
            simu_img = self.ST(r_x, vec_field)
            # simu_img = self.trans(r_x, option_simu)
            d_real = d.logits(x, mode='train')
            d_fake = d.logits(simu_img, mode='train')
            if gan_mode == 'non_saturate':
                d_loss = (-F.logsigmoid(d_real).mean() - F.logsigmoid(-d_fake).mean()) / 2
                g_loss = -F.logsigmoid(d_fake).mean() + grad_loss
            elif gan_mode == 'wgan_gp':
                alpha = torch.rand(batch_size, 1, 1, 1, 1, device=self.device)
                x_r = alpha * simu_img + (1 - alpha) * x
                d_r = d.logits(x_r)
                grad = torch.autograd.grad(d_r.sum(), x_r, create_graph=True)
                grad_norm = grad[0].reshape(batch_size, -1).norm(dim=1)
                d_loss = (d_fake - d_real).mean() + 10 * ((grad_norm -1)**2).mean()
                g_loss = -d_fake.mean()
        return d_loss, g_loss

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
