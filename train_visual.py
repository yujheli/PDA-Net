import os, sys
import os.path as osp
import time

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.autograd import Variable

from reid import datasets
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomPairSampler, RandomPosSampler
from reid.utils.data import transforms as T
from reid.utils.serialization import load_checkpoint
from reid.evaluators import CascadeEvaluator

from cpdgan.options import Options
# from cpdgan.utils.visualizer import Visualizer
from cpdgan.model import CPDGANModel
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter 

def get_data(name, data_dir, height, width, batch_size, workers, pose_aug):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)

    # use combined trainval set for training as default
    train_loader = DataLoader(
        Preprocessor(dataset.trainval, root=dataset.images_dir, with_pose=True, pose_root=dataset.poses_dir,
                    pid_imgs=dataset.trainval_query, height=height, width=width, pose_aug=pose_aug),
        sampler=RandomPosSampler(dataset.trainval, neg_pos_ratio=6),
        batch_size=batch_size, num_workers=workers, pin_memory=False)

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return dataset, train_loader, test_loader

def get_cross_data(opt):
    height = opt.height
    width = opt.width
    batch_size = opt.batch_size
    workers = opt.workers
    pose_aug = opt.pose_aug
    
    
    
    source_root = osp.join(opt.dataroot, opt.source)
    target_root = osp.join(opt.dataroot, opt.target)
    
#     root = osp.join(opt.dataroot, opt.dataset)
    source_dataset = datasets.create(opt.source, source_root)
    target_dataset = datasets.create(opt.target, target_root)

    # use combined trainval set for training as default
    train_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    source_train_loader = DataLoader(
        Preprocessor(source_dataset.trainval, root=source_dataset.images_dir, with_pose=True, pose_root=source_dataset.poses_dir,
                    pid_imgs=source_dataset.trainval_query, height=height, width=width, pose_aug=pose_aug, transform=train_transformer),
        sampler=RandomPosSampler(source_dataset.trainval, neg_pos_ratio=6),
        batch_size=batch_size, num_workers=workers, pin_memory=False)
    
    target_train_loader = DataLoader(
        Preprocessor(target_dataset.trainval, root=target_dataset.images_dir, with_pose=True, pose_root=target_dataset.poses_dir,
                    pid_imgs=target_dataset.trainval_query, height=height, width=width, pose_aug=pose_aug, transform=train_transformer),
        sampler=RandomPosSampler(target_dataset.trainval, neg_pos_ratio=6),
        batch_size=batch_size, num_workers=workers, pin_memory=False)
    
    
#     train_loader = DataLoader(
#         Preprocessor(dataset.trainval, root=dataset.images_dir, with_pose=True, pose_root=dataset.poses_dir,
#                     pid_imgs=dataset.trainval_query, height=height, width=width, pose_aug=pose_aug),
#         sampler=RandomPairSampler(dataset.trainval, neg_pos_ratio=3),
#         batch_size=batch_size, num_workers=workers, pin_memory=False)

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    source_test_loader = DataLoader(
        Preprocessor(list(set(source_dataset.query) | set(source_dataset.gallery)),
                     root=source_dataset.images_dir, transform=test_transformer),
        batch_size=batch_size*4, num_workers=workers,
        shuffle=False, pin_memory=False)
    
    target_test_loader = DataLoader(
        Preprocessor(list(set(target_dataset.query) | set(target_dataset.gallery)),
                     root=target_dataset.images_dir, transform=test_transformer),
        batch_size=batch_size*4, num_workers=workers,
        shuffle=False, pin_memory=False)

#     test_loader = DataLoader(
#         Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
#                      root=dataset.images_dir, transform=test_transformer),
#         batch_size=batch_size, num_workers=workers,
#         shuffle=False, pin_memory=False)

    return source_dataset, source_train_loader, source_test_loader, target_dataset, target_train_loader, target_test_loader


def print_current_errors(epoch, i, errors, t):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)

    print(message)

def main():
    opt = Options().parse()
    source_dataset, source_train_loader, source_test_loader, target_dataset, target_train_loader, target_test_loader = get_cross_data(opt)
    
#     dataset, train_loader, test_loader = get_data(opt.dataset, opt.dataroot, opt.height, opt.width, opt.batch_size, opt.workers, opt.pose_aug)

    dataset_size = len(source_dataset.trainval)*4
    print('#souce training images = %d' % dataset_size)

    model = CPDGANModel(opt)
#     visualizer = Visualizer(opt)
    writer = SummaryWriter()

    evaluator = CascadeEvaluator(
                    torch.nn.DataParallel(model.net_E.module.base_model).cuda(),
                    model.net_E.module.embed_model,
                    embed_dist_fn=lambda x: F.softmax(Variable(x), dim=1).data[:, 0])
    
    if opt.stage==0:
        print('Test with baseline model:')
        top1, mAP = evaluator.evaluate(target_test_loader, target_dataset.query, target_dataset.gallery, second_stage=False, rerank_topk=100, dataset=opt.dataset)
        message = '\n Test with baseline model:  mAP: {:5.1%}  top1: {:5.1%}\n'.format(mAP, top1)
        print(message)
#         visualizer.print_reid_results(message)

    total_steps = 0
    best_mAP = 0
    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        model.reset_model_status()
        
        
        target_iter = enumerate(target_train_loader)

        for i, source_data in enumerate(source_train_loader):
            
            """ Load Target Data """
            try:
                _, target_data = next(target_iter)
            except:
                target_iter = enumerate(target_train_loader)
                _, target_data = next(target_iter)
#             print("target datatype",target_data[0].keys())
        
            iter_start_time = time.time()
#             visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            
            model.set_inputs(target_data,source_data)
            model.forward_test_cross()
            
            
            #Display visual results
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                
                #TB visualization
                visual_data = model.get_test_cross_visuals()
                for key, value in visual_data.items():
                    writer.add_image(key, make_grid(value, nrow=24), total_steps)
                    save_image(value,'./save_images/{}_{}.png'.format(total_steps,key),nrow=24)
                    
#                 visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            
#             #Plot curves
            if total_steps % opt.print_freq == 0:
# #                 errors = model.get_current_errors()
#                 errors = model.get_current_cross_errors()
#                 #TB scalar
#                 for key, value in errors.items():
#                     writer.add_scalar(key, value, total_steps)
                
                t = (time.time() - iter_start_time) / opt.batch_size
                message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
                print(message)
#                 print_current_errors(epoch, epoch_iter, errors, t)
#                 visualizer.print_current_errors(epoch, epoch_iter, errors, t)
#                 if opt.display_id > 0:
#                     visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

#         if epoch % opt.save_step == 0:
#             print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
#             model.save(epoch)

#         if epoch % opt.eval_step == 0 and opt.stage!=1:
            
#             top1, mAP = evaluator.evaluate(target_test_loader, target_dataset.query, target_dataset.gallery, second_stage=False, rerank_topk=100, dataset=opt.dataset)
# #             mAP = evaluator.evaluate(val_loader, dataset.val, dataset.val, top1=False)
            
#             writer.add_scalar('tar_rank1', top1, epoch)
#             writer.add_scalar('tar_mAP', mAP, epoch)
            
#             is_best = mAP > best_mAP
#             best_mAP = max(mAP, best_mAP)
#             if is_best:
#                 model.save('best')
#             message = '\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.format(epoch, mAP, best_mAP, ' *' if is_best else '')
#             print(message)
# #             visualizer.print_reid_results(message)
            
#             #=========source test=========
#             top1, mAP = evaluator.evaluate(source_test_loader, source_dataset.query, source_dataset.gallery, second_stage=False, rerank_topk=100, dataset=opt.dataset)
# #             mAP = evaluator.evaluate(val_loader, dataset.val, dataset.val, top1=False)
            
#             writer.add_scalar('src_rank1', top1, epoch)
#             writer.add_scalar('src_mAP', mAP, epoch)

#         print('End of epoch %d / %d \t Time Taken: %d sec' %
#               (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
#         model.update_learning_rate()

    # Final test
    if opt.stage!=1:
        print('Test with best model:')
        checkpoint = load_checkpoint(osp.join(opt.checkpoints, opt.name, '%s_net_%s.pth' % ('best', 'E')))
        model.net_E.load_state_dict(checkpoint)
        top1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, rerank_topk=100, second_stage=False, dataset=opt.dataset)
        message = '\n Test with best model:  mAP: {:5.1%}  top1: {:5.1%}\n'.format(mAP, top1)
        print(message)
#         visualizer.print_reid_results(message)

if __name__ == '__main__':
    main()
