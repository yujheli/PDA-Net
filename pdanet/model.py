import os,sys
import itertools
import numpy as np
import math
import random
import copy
from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch.nn import functional as F

import cpdgan.utils.util as util
from pdanet.networks import get_norm_layer, init_weights, CustomPoseGenerator, NLayerDiscriminator, \
                            remove_module_key, set_bn_fix, get_scheduler, print_network
from pdanet.losses import GANLoss, TripletLoss, MMDLoss
from reid.models import create
from reid.models.embedding import EltwiseSubEmbed
from reid.models.multi_branch import SiameseNet

class PDANetModel(object):

    def __init__(self, opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints, opt.name)
        self.norm_layer = get_norm_layer(norm_type=opt.norm)

        self._init_models()
        self._init_losses()
        self._init_cross_optimizers()


    def _init_models(self):
        #G For source
        self.net_G = CustomPoseGenerator(self.opt.pose_feature_size, 2048, self.opt.noise_feature_size,
                                dropout=self.opt.drop, norm_layer=self.norm_layer, fuse_mode=self.opt.fuse_mode, connect_layers=self.opt.connect_layers)
        #G For target
        self.tar_net_G = CustomPoseGenerator(self.opt.pose_feature_size, 2048, self.opt.noise_feature_size,
                                dropout=self.opt.drop, norm_layer=self.norm_layer, fuse_mode=self.opt.fuse_mode, connect_layers=self.opt.connect_layers)
        
        #We share same E for cross-dataset
        e_base_model = create(self.opt.arch, cut_at_pooling=True)
        e_embed_model = EltwiseSubEmbed(use_batch_norm=True, use_classifier=True, num_features=2048, num_classes=2)
        self.net_E = SiameseNet(e_base_model, e_embed_model)
        
        #Di For source
        di_base_model = create(self.opt.arch, cut_at_pooling=True)
        di_embed_model = EltwiseSubEmbed(use_batch_norm=True, use_classifier=True, num_features=2048, num_classes=1)
        self.net_Di = SiameseNet(di_base_model, di_embed_model)
        
        #Di For target
        di_base_model = create(self.opt.arch, cut_at_pooling=True)
        di_embed_model = EltwiseSubEmbed(use_batch_norm=True, use_classifier=True, num_features=2048, num_classes=1)
        self.tar_net_Di = SiameseNet(di_base_model, di_embed_model)
        
        #We share same Dp for cross-dataset
        self.net_Dp = NLayerDiscriminator(3+18, norm_layer=self.norm_layer)
        
        
        #Load model
        if self.opt.stage==1:
            self._load_state_dict(self.net_E, self.opt.netE_pretrain)
            self._load_state_dict(self.net_G, self.opt.netG_pretrain)
            self._load_state_dict(self.net_Di, self.opt.netDi_pretrain)
            self._load_state_dict(self.net_Dp, self.opt.netDp_pretrain)
            
            self._load_state_dict(self.tar_net_Di, self.opt.tar_netDi_pretrain)
            self._load_state_dict(self.tar_net_G, self.opt.tar_netG_pretrain)
            
        elif self.opt.stage==2:
            self._load_state_dict(self.net_E, self.opt.netE_pretrain)
            self._load_state_dict(self.net_G, self.opt.netG_pretrain)
            self._load_state_dict(self.net_Di, self.opt.netDi_pretrain)
            self._load_state_dict(self.net_Dp, self.opt.netDp_pretrain)
            
            self._load_state_dict(self.tar_net_Di, self.opt.tar_netDi_pretrain)
            self._load_state_dict(self.tar_net_G, self.opt.tar_netG_pretrain)
        else:
            assert('unknown training stage')

        self.net_E = torch.nn.DataParallel(self.net_E).cuda()
        self.net_G = torch.nn.DataParallel(self.net_G).cuda()
        self.net_Di = torch.nn.DataParallel(self.net_Di).cuda()
        self.net_Dp = torch.nn.DataParallel(self.net_Dp).cuda()
        
        self.tar_net_G = torch.nn.DataParallel(self.tar_net_G).cuda()
        self.tar_net_Di = torch.nn.DataParallel(self.tar_net_Di).cuda()
            
    def reset_model_status(self):
        if self.opt.stage==1:
            self.net_G.train()
            self.tar_net_G.train()
            self.net_Dp.train()
            self.net_E.eval()
            self.net_Di.train()
            self.net_Di.apply(set_bn_fix)
            self.tar_net_Di.train()
            self.tar_net_Di.apply(set_bn_fix)
            
        elif self.opt.stage==2:
            self.net_E.train()
            self.net_G.train()
            self.tar_net_G.train()
            self.net_Di.train()
            self.tar_net_Di.train()
            self.net_Dp.train()
            self.net_E.apply(set_bn_fix)
            self.net_Di.apply(set_bn_fix)
            self.tar_net_Di.apply(set_bn_fix)
            
    def _load_state_dict(self, net, path):
        state_dict = remove_module_key(torch.load(path))
        net.load_state_dict(state_dict)

    def _init_losses(self):
        self.criterion_Triplet = TripletLoss(margin=self.opt.tri_margin)
        self.criterion_MMD = MMDLoss(sigma_list=[1, 2, 10])
        #Smooth label is a mechanism
        if self.opt.smooth_label:
            self.criterionGAN_D = GANLoss(smooth=True).cuda()
            self.rand_list = [True] * 1 + [False] * 10000
        else:
            self.criterionGAN_D = GANLoss(smooth=False).cuda()
            self.rand_list = [False]
        self.criterionGAN_G = GANLoss(smooth=False).cuda()
    
    #For cross dataset optimizers
    def _init_cross_optimizers(self):

        if self.opt.stage==1:
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                                lr=self.opt.lr*0.1, betas=(0.5, 0.999))
            self.optimizer_tar_G = torch.optim.Adam(self.tar_net_G.parameters(),
                                                lr=self.opt.lr*0.1, betas=(0.5, 0.999))
            self.optimizer_Di = torch.optim.SGD(self.net_Di.parameters(),
                                                lr=self.opt.lr*0.01, momentum=0.9, weight_decay=1e-4)
            self.optimizer_tar_Di = torch.optim.SGD(self.tar_net_Di.parameters(),
                                                lr=self.opt.lr*0.01, momentum=0.9, weight_decay=1e-4)
            self.optimizer_Dp = torch.optim.SGD(self.net_Dp.parameters(),
                                                lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)
        elif self.opt.stage==2:
            param_groups = [{'params': self.net_E.module.base_model.parameters(), 'lr_mult': 0.1},
                            {'params': self.net_E.module.embed_model.parameters(), 'lr_mult': 1.0},
                            {'params': self.net_G.parameters(), 'lr_mult': 0.1}]
            
            param_tar_groups = [{'params': self.net_E.module.base_model.parameters(), 'lr_mult': 0.1},
                            {'params': self.net_E.module.embed_model.parameters(), 'lr_mult': 1.0},
                            {'params': self.tar_net_G.parameters(), 'lr_mult': 0.1}]
            
            self.optimizer_G = torch.optim.Adam(param_groups,
                                                lr=self.opt.lr*0.1, betas=(0.5, 0.999))
            self.optimizer_tar_G = torch.optim.Adam(param_tar_groups,
                                                lr=self.opt.lr*0.1, betas=(0.5, 0.999))
            self.optimizer_Di = torch.optim.SGD(self.net_Di.parameters(),
                                                lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)
            self.optimizer_tar_Di = torch.optim.SGD(self.tar_net_Di.parameters(),
                                                lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)
            self.optimizer_Dp = torch.optim.SGD(self.net_Dp.parameters(),
                                                lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)
            self.optimizer_E = torch.optim.Adam(self.net_E.module.base_model.parameters(),
                                                lr=self.opt.lr*0.1, betas=(0.5, 0.999))

        self.schedulers = []
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_Di)
        self.optimizers.append(self.optimizer_Dp)
        
        self.optimizers.append(self.optimizer_tar_G)
        self.optimizers.append(self.optimizer_tar_Di)
        
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, self.opt))        

    def set_input(self, input):
        input1, input2 = input
        labels = (input1['pid']==input2['pid']).long()
        noise = torch.randn(labels.size(0), self.opt.noise_feature_size)

        # keep the same pose map for persons with the same identity
        mask = labels.view(-1,1,1,1).expand_as(input1['posemap'])
        input2['posemap'] = input1['posemap']*mask.float() + input2['posemap']*(1-mask.float())
        mask = labels.view(-1,1,1,1).expand_as(input1['target'])
        input2['target'] = input1['target']*mask.float() + input2['target']*(1-mask.float())

        origin = torch.cat([input1['origin'], input2['origin']])
        target = torch.cat([input1['target'], input2['target']])
        posemap = torch.cat([input1['posemap'], input2['posemap']])
        noise = torch.cat((noise, noise))

        self.origin = origin.cuda()
        self.target = target.cuda()
        self.posemap = posemap.cuda()
        self.labels = labels.cuda()
        self.noise = noise.cuda()
    
    
    #For cross-dataset
    def set_inputs(self, target_data, source_data):
        #=============source data===============================
        input1, input2 = source_data
        labels = (input1['pid']==input2['pid']).long()
        noise = torch.randn(labels.size(0), self.opt.noise_feature_size)

        # keep the same pose map for persons with the same identity
        mask = labels.view(-1,1,1,1).expand_as(input1['posemap'])
        input2['posemap'] = input1['posemap']*mask.float() + input2['posemap']*(1-mask.float())
        mask = labels.view(-1,1,1,1).expand_as(input1['target'])
        input2['target'] = input1['target']*mask.float() + input2['target']*(1-mask.float())

        origin = torch.cat([input1['origin'], input2['origin']])
        target = torch.cat([input1['target'], input2['target']])
        posemap = torch.cat([input1['posemap'], input2['posemap']])
        noise = torch.cat((noise, noise))
        plabels = torch.cat((input1['pid'].long(), input2['pid'].long())) # Used for triplet loss

        self.s_origin = origin.cuda()
        self.s_target = target.cuda()
        self.s_posemap = posemap.cuda()
        self.s_labels = labels.cuda()
        self.s_noise = noise.cuda()
        self.s_plabels = plabels.cuda() # Used for triplet loss
        
        #=============target data===============================
        input1, input2 = target_data
        noise = torch.randn(input1['origin'].size(0), self.opt.noise_feature_size)

        
        origin = torch.cat([input1['origin'], input2['origin']])
        target = torch.cat([input1['target'], input2['target']])
        posemap = torch.cat([input1['posemap'], input2['posemap']])
        noise = torch.cat((noise, noise))
        
        self.t_origin = origin.cuda()
        self.t_target = target.cuda()
        self.t_posemap = posemap.cuda()
        self.t_noise = noise.cuda()
        
    #forward cross
    def forward_cross(self):
        #source
        A = Variable(self.s_origin)
        B_map = Variable(self.s_posemap)
        z = Variable(self.s_noise)
        bs = A.size(0)

        A_id1, A_id2, self.s_id_score = self.net_E(A[:bs//2], A[bs//2:])
        A_id = torch.cat((A_id1, A_id2))
        self.s_A_id = A_id
        self.s_fake = self.net_G(B_map, A_id.view(A_id.size(0), A_id.size(1), 1, 1), z.view(z.size(0), z.size(1), 1, 1))
        
        #source to target
        A_id1_st = A_id1[:bs//2]
        z_st = z[:bs//2]
        self.st_fake = self.tar_net_G(B_map[:bs//2], A_id1_st.view(A_id1_st.size(0), A_id1_st.size(1), 1, 1), z_st.view(z_st.size(0), z_st.size(1), 1, 1))
#         self.st_fake = self.tar_net_G(B_map, A_id.view(A_id.size(0), A_id.size(1), 1, 1), z.view(z.size(0), z.size(1), 1, 1))
        
        #target
        A = Variable(self.t_origin)
        B_map = Variable(self.t_posemap)
        z = Variable(self.t_noise)
        bs = A.size(0)

        A_id1, A_id2, self.t_id_score = self.net_E(A[:bs//2], A[bs//2:])
        A_id = torch.cat((A_id1, A_id2))
        self.t_A_id = A_id
        self.t_fake = self.tar_net_G(B_map, A_id.view(A_id.size(0), A_id.size(1), 1, 1), z.view(z.size(0), z.size(1), 1, 1))
    
    
    def backward_Dp(self):
        real_pose = torch.cat((Variable(self.posemap), Variable(self.target)),dim=1)
        fake_pose = torch.cat((Variable(self.posemap), self.fake.detach()),dim=1)
        pred_real = self.net_Dp(real_pose)
        pred_fake = self.net_Dp(fake_pose)

        if random.choice(self.rand_list):
            loss_D_real = self.criterionGAN_D(pred_fake, True)
            loss_D_fake = self.criterionGAN_D(pred_real, False)
        else:
            loss_D_real = self.criterionGAN_D(pred_real, True)
            loss_D_fake = self.criterionGAN_D(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.loss_Dp = loss_D.data[0]

    def backward_Di(self):
        _, _, pred_real = self.net_Di(Variable(self.origin), Variable(self.target))
        _, _, pred_fake = self.net_Di(Variable(self.origin), self.fake.detach())
        if random.choice(self.rand_list):
            loss_D_real = self.criterionGAN_D(pred_fake, True)
            loss_D_fake = self.criterionGAN_D(pred_real, False)
        else:
            loss_D_real = self.criterionGAN_D(pred_real, True)
            loss_D_fake = self.criterionGAN_D(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.loss_Di = loss_D.data[0]
        
    def backward_E_MMD(self):
        
        loss_mmd = self.criterion_MMD(self.s_A_id, self.t_A_id)
        self.loss_mmd = loss_mmd.item()
        loss_mmd = loss_mmd * self.opt.lambda_mmd
        loss_mmd.backward(retain_graph=True)
    
    def backward_s_G(self):
        
#         self.s_plabels
#         self.criterion_Triplet
#         self.s_A_id
#         self.criterion_Triplet = TripletLoss(margin=opt.tri_margin)
#         self.criterion_MMD = MMDLoss(sigma_list=[1, 2, 10])
    
        loss_tri, prec = self.criterion_Triplet(self.s_A_id, self.s_plabels)
        
        
        loss_v = F.cross_entropy(self.s_id_score, Variable(self.s_labels).view(-1))
        loss_r = F.l1_loss(self.s_fake, Variable(self.s_target))
        fake_1 = self.s_fake[:self.s_fake.size(0)//2]
        fake_2 = self.s_fake[self.s_fake.size(0)//2:]

        _, _, pred_fake_Di = self.net_Di(Variable(self.s_origin), self.s_fake)
        pred_fake_Dp = self.net_Dp(torch.cat((Variable(self.s_posemap),self.s_fake),dim=1))
        loss_G_GAN_Di = self.criterionGAN_G(pred_fake_Di, True)
        loss_G_GAN_Dp = self.criterionGAN_G(pred_fake_Dp, True)

        loss_G = loss_G_GAN_Di + loss_G_GAN_Dp + \
                loss_r * self.opt.lambda_recon + \
                loss_v * self.opt.lambda_veri  + \
                loss_tri * self.opt.lambda_tri
        
        loss_G.backward()

        del self.s_id_score
        self.loss_s_G = loss_G.item()
        self.loss_s_v = loss_v.item()

        self.loss_s_r = loss_r.item()
        self.loss_s_G_GAN_Di = loss_G_GAN_Di.item()
        self.loss_s_G_GAN_Dp = loss_G_GAN_Dp.item()
        self.loss_s_tri = loss_tri.item()

    
    def backward_t_G(self):

        loss_r = F.l1_loss(self.t_fake, Variable(self.t_target))
        fake_1 = self.t_fake[:self.t_fake.size(0)//2]
        fake_2 = self.t_fake[self.t_fake.size(0)//2:]

        _, _, pred_fake_Di = self.tar_net_Di(Variable(self.t_origin), self.t_fake)
        pred_fake_Dp = self.net_Dp(torch.cat((Variable(self.t_posemap),self.t_fake),dim=1))
        loss_G_GAN_Di = self.criterionGAN_G(pred_fake_Di, True)
        loss_G_GAN_Dp = self.criterionGAN_G(pred_fake_Dp, True)

        loss_G = loss_G_GAN_Di + loss_G_GAN_Dp + \
                loss_r * self.opt.lambda_recon
        
        loss_G.backward()

        del self.t_id_score
        self.loss_t_G = loss_G.item()

        self.loss_t_r = loss_r.item()
        self.loss_t_G_GAN_Di = loss_G_GAN_Di.item()
        self.loss_t_G_GAN_Dp = loss_G_GAN_Dp.item()

    
    def backward_s_Di(self):
        _, _, pred_real = self.net_Di(Variable(self.s_origin), Variable(self.s_target))
        _, _, pred_fake = self.net_Di(Variable(self.s_origin), self.s_fake.detach())
        if random.choice(self.rand_list):
            loss_D_real = self.criterionGAN_D(pred_fake, True)
            loss_D_fake = self.criterionGAN_D(pred_real, False)
        else:
            loss_D_real = self.criterionGAN_D(pred_real, True)
            loss_D_fake = self.criterionGAN_D(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
#         self.loss_s_Di = loss_D.data[0]
        self.loss_s_Di = loss_D.item()

    
    def backward_t_Di(self):
        _, _, pred_real = self.tar_net_Di(Variable(self.t_origin), Variable(self.t_target))
        _, _, pred_fake = self.tar_net_Di(Variable(self.t_origin), self.t_fake.detach())
        if random.choice(self.rand_list):
            loss_D_real = self.criterionGAN_D(pred_fake, True)
            loss_D_fake = self.criterionGAN_D(pred_real, False)
        else:
            loss_D_real = self.criterionGAN_D(pred_real, True)
            loss_D_fake = self.criterionGAN_D(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
#         self.loss_t_Di = loss_D.data[0]
        self.loss_t_Di = loss_D.item()

    def backward_cross_Dp(self):
        real_pose = torch.cat((Variable(self.s_posemap), Variable(self.s_target)),dim=1)
        fake_pose = torch.cat((Variable(self.s_posemap), self.s_fake.detach()),dim=1)
        pred_real = self.net_Dp(real_pose)
        pred_fake = self.net_Dp(fake_pose)

        if random.choice(self.rand_list):
            loss_D_real = self.criterionGAN_D(pred_fake, True)
            loss_D_fake = self.criterionGAN_D(pred_real, False)
        else:
            loss_D_real = self.criterionGAN_D(pred_real, True)
            loss_D_fake = self.criterionGAN_D(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
#         self.loss_Dp = loss_D.data[0]
        self.loss_Dp = loss_D.item()

        real_pose = torch.cat((Variable(self.t_posemap), Variable(self.t_target)),dim=1)
        fake_pose = torch.cat((Variable(self.t_posemap), self.t_fake.detach()),dim=1)
        pred_real = self.net_Dp(real_pose)
        pred_fake = self.net_Dp(fake_pose)

        if random.choice(self.rand_list):
            loss_D_real = self.criterionGAN_D(pred_fake, True)
            loss_D_fake = self.criterionGAN_D(pred_real, False)
        else:
            loss_D_real = self.criterionGAN_D(pred_real, True)
            loss_D_fake = self.criterionGAN_D(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
#         self.loss_Dp += loss_D.data[0]
        self.loss_Dp += loss_D.item()

    
    def optimize_cross_parameters(self):
        self.forward_cross()
        
        self.optimizer_E.zero_grad()
        self.backward_E_MMD()
        self.optimizer_E.step()
        
        
        self.optimizer_Di.zero_grad()
        self.backward_s_Di()
        self.optimizer_Di.step()

        self.optimizer_Dp.zero_grad()
        self.backward_cross_Dp()
        self.optimizer_Dp.step()

        self.optimizer_G.zero_grad()
        self.backward_s_G()
        self.optimizer_G.step()
        
        self.optimizer_tar_Di.zero_grad()
        self.backward_t_Di()
        self.optimizer_tar_Di.step()
        
        self.optimizer_tar_G.zero_grad()
        self.backward_t_G()
        self.optimizer_tar_G.step()
        
    
    def get_current_errors(self):
        return OrderedDict([('G_v', self.loss_v),
                            ('G_r', self.loss_r),
                            ('G_sp', self.loss_sp),
                            ('G_gan_Di', self.loss_G_GAN_Di),
                            ('G_gan_Dp', self.loss_G_GAN_Dp),
                            ('D_i', self.loss_Di),
                            ('D_p', self.loss_Dp)
                            ])
    
    def get_current_cross_errors(self):
        return OrderedDict([('G_tar_rec', self.loss_t_r),
                            ('G_tar_gan_Di', self.loss_t_G_GAN_Di),
                            ('G_tar_gan_Dp', self.loss_t_G_GAN_Dp),
                            ('tar_D_i', self.loss_t_Di),
                            ('D_p', self.loss_Dp),
                            ('G_src_v', self.loss_s_v),
                            ('G_src_rec', self.loss_s_r),
                            ('G_src_gan_Di', self.loss_s_G_GAN_Di),
                            ('G_src_gan_Dp', self.loss_s_G_GAN_Dp),
                            ('src_D_i', self.loss_s_Di),
                            ('MMD', self.loss_mmd),
                            ('src_tri', self.loss_s_tri)
                            ])
    
    def get_current_visuals(self):
        input = util.tensor2im(self.origin)
        target = util.tensor2im(self.target)
        fake = util.tensor2im(self.fake)
        map = self.posemap.sum(1)
        map[map>1]=1
        map = util.tensor2im(torch.unsqueeze(map,1))
        return OrderedDict([('input', input), ('posemap', map), ('fake', fake), ('target', target)])
    
    
    def get_tf_visuals(self):
        input = util.tensor2ims(self.origin)
        target = util.tensor2ims(self.target)
        fake = util.tensor2ims(self.fake)
        map = self.posemap.sum(1)
        map[map>1]=1
        map = util.tensor2ims(torch.unsqueeze(map,1))
        return OrderedDict([('input', input), ('posemap', map), ('fake', fake), ('target', target)])
    
    
    #     self.t_origin = origin.cuda()
    #     self.t_target = target.cuda()
    #     self.t_posemap = posemap.cuda()
    #     self.t_labels = labels.cuda()
    #     self.t_noise = noise.cuda()
    def get_tf_cross_visuals(self):
        src_input = util.tensor2ims(self.s_origin)
        src_target = util.tensor2ims(self.s_target)
        src_fake = util.tensor2ims(self.s_fake.data)
        src_map = self.s_posemap.sum(1)
        src_map[src_map>1]=1
        src_map = util.tensor2ims(torch.unsqueeze(src_map,1))
        
        tar_input = util.tensor2ims(self.t_origin)
        tar_target = util.tensor2ims(self.t_target)
        tar_fake = util.tensor2ims(self.t_fake.data)
        tar_map = self.t_posemap.sum(1)
        tar_map[tar_map>1]=1
        tar_map = util.tensor2ims(torch.unsqueeze(tar_map,1))
        
        src2tgt_fake = util.tensor2ims(self.st_fake.data)
        
        return OrderedDict([('src_input', src_input),
                            ('src_posemap', src_map),
                            ('src_fake', src_fake),
                            ('src_target', src_target),
                            ('tar_input', tar_input),
                            ('tar_posemap', tar_map),
                            ('tar_fake', tar_fake),
                            ('tar_target', tar_target),
                            ('src2tgt_fake', src2tgt_fake)
                           ])
    
    def save(self, epoch):
        self.save_network(self.net_E, 'E', epoch)
        self.save_network(self.net_G, 'G', epoch)
        self.save_network(self.net_Di, 'Di', epoch)
        self.save_network(self.net_Dp, 'Dp', epoch)
        
        self.save_network(self.tar_net_G, 'tar_G', epoch)
        self.save_network(self.tar_net_Di, 'tar_Di', epoch)

    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        
        
        
    def forward_test_cross(self):
        #source
        A = Variable(self.s_origin)
        B_map = Variable(self.s_posemap)
        z = Variable(self.s_noise)
        bs = A.size(0)

        A_id1, A_id2, self.s_id_score = self.net_E(A[:bs//2], A[bs//2:])
        A_id = torch.cat((A_id1, A_id2))
        self.s_A_id = A_id
        self.s_fake = self.net_G(B_map, A_id.view(A_id.size(0), A_id.size(1), 1, 1), z.view(z.size(0), z.size(1), 1, 1))
        
        #source to target
#         A_id1_st = A_id1[:bs//]
#         z_st = z[:bs//]
#         self.st_fake = self.tar_net_G(B_map[:bs//], A_id1_st.view(A_id1_st.size(0), A_id1_st.size(1), 1, 1), z_st.view(z_st.size(0), z_st.size(1), 1, 1))
        self.st_fake = self.tar_net_G(B_map, A_id.view(A_id.size(0), A_id.size(1), 1, 1), z.view(z.size(0), z.size(1), 1, 1))
        
        #target
        A = Variable(self.t_origin)
        B_map = Variable(self.s_posemap)
        z = Variable(self.t_noise)
        bs = A.size(0)

        A_id1, A_id2, self.t_id_score = self.net_E(A[:bs//2], A[bs//2:])
        A_id = torch.cat((A_id1, A_id2))
        self.t_A_id = A_id
        self.t_fake = self.tar_net_G(B_map, A_id.view(A_id.size(0), A_id.size(1), 1, 1), z.view(z.size(0), z.size(1), 1, 1))
        
        self.ts_fake = self.net_G(B_map, A_id.view(A_id.size(0), A_id.size(1), 1, 1), z.view(z.size(0), z.size(1), 1, 1))
        
    
    def get_test_cross_visuals(self):
        src_input = util.tensor2ims(self.s_origin)
        src_target = util.tensor2ims(self.s_target)
        src_fake = util.tensor2ims(self.s_fake.data)
        src_map = self.s_posemap.sum(1)
        src_map[src_map>1]=1
        src_map = util.tensor2ims(torch.unsqueeze(src_map,1))
        
        tar_input = util.tensor2ims(self.t_origin)
        tar_target = util.tensor2ims(self.t_target)
        tar_fake = util.tensor2ims(self.t_fake.data)
        tar_map = self.t_posemap.sum(1)
        tar_map[tar_map>1]=1
        tar_map = util.tensor2ims(torch.unsqueeze(tar_map,1))
        
        src2tgt_fake = util.tensor2ims(self.st_fake.data)
        tgt2src_fake = util.tensor2ims(self.ts_fake.data)
        
        return OrderedDict([('src_input', src_input),
                            ('src_posemap', src_map),
                            ('src_fake', src_fake),
                            ('src_target', src_target),
                            ('tar_input', tar_input),
                            ('tar_fake', tar_fake),
                            ('src2tgt_fake', src2tgt_fake),
                            ('tgt2src_fake', tgt2src_fake),
                           ])
    