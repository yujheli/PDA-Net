import os, sys
import os.path as osp
import time

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.autograd import Variable

from reid import datasets
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomPairSampler
from reid.utils.data import transforms as T
from reid.utils.serialization import load_checkpoint
from reid.evaluators import CascadeEvaluator
from pdanet.options import Options
from pdanet.model import PDANetModel
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter 

def get_cross_data(opt):
    height = opt.height
    width = opt.width
    batch_size = opt.batch_size
    workers = opt.workers
    pose_aug = opt.pose_aug
    
    source_root = osp.join(opt.dataroot, opt.source)
    target_root = osp.join(opt.dataroot, opt.target)
    
    source_dataset = datasets.create(opt.source, source_root)
    target_dataset = datasets.create(opt.target, target_root)

    #==================================== Define your own target and source dataloaders here ==================================================================
    
    #source_train_loader
    #Output format: dictionary: 'pid', 'posemap', 'origin', 'target'
    
    #source_test_loader
    #Output format: (imgs, fnames, pids, _)
    
    #target_train_loader
    #Output format: dictionary: 'posemap', 'origin', 'target'
    
    #target_test_loader
    #Output format: (imgs, fnames, pids, _)
    
    
    
    #==========================================================================================================================================================
    

    return source_dataset, source_train_loader, source_test_loader, target_dataset, target_train_loader, target_test_loader


def main():
    opt = Options().parse()
    source_dataset, source_train_loader, source_test_loader, target_dataset, target_train_loader, target_test_loader = get_cross_data(opt)

    model = PDANetModel(opt)
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
        
    total_steps = 0
    best_mAP = 0
    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        model.reset_model_status()
        
        
        target_iter = enumerate(target_train_loader)

        for i, source_data in enumerate(source_train_loader):


            """ Load Target Data at the same time"""
            try:
                _, target_data = next(target_iter)
            except:
                target_iter = enumerate(target_train_loader)
                _, target_data = next(target_iter)
                
            iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            
            model.set_inputs(target_data,source_data)
            model.optimize_cross_parameters()
            
            if total_steps % 10000 == 0:
                top1, mAP = evaluator.evaluate(target_test_loader, target_dataset.query, target_dataset.gallery, second_stage=False, rerank_topk=100, dataset=opt.dataset)
                writer.add_scalar('tgt_rank1', top1, int(total_steps/10000))
                writer.add_scalar('tgt_mAP', mAP, int(total_steps/10000))
            
            #Display visual results
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                
                #TB visualization
                visual_data = model.get_tf_cross_visuals()
                for key, value in visual_data.items():
                    writer.add_image(key, make_grid(value, nrow=16), total_steps)
                                
            #Plot curves
            if total_steps % opt.print_freq == 0:
#                 errors = model.get_current_errors()
                errors = model.get_current_cross_errors()
                #TB scalar
                for key, value in errors.items():
                    writer.add_scalar(key, value, total_steps)
                
                t = (time.time() - iter_start_time) / opt.batch_size

        if epoch % opt.save_step == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save(epoch)

        if epoch % opt.eval_step == 0 and opt.stage!=1:
            
            top1, mAP = evaluator.evaluate(target_test_loader, target_dataset.query, target_dataset.gallery, second_stage=False, rerank_topk=100, dataset=opt.dataset)
            
            writer.add_scalar('tar_rank1', top1, epoch)
            writer.add_scalar('tar_mAP', mAP, epoch)
            
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            if is_best:
                model.save('best')
            message = '\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.format(epoch, mAP, best_mAP, ' *' if is_best else '')
            print(message)
            
            #=========source test=========
            top1, mAP = evaluator.evaluate(source_test_loader, source_dataset.query, source_dataset.gallery, second_stage=False, rerank_topk=100, dataset=opt.dataset)
            
            writer.add_scalar('src_rank1', top1, epoch)
            writer.add_scalar('src_mAP', mAP, epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    # Final test
    if opt.stage!=1:
        print('Test with best model:')
        checkpoint = load_checkpoint(osp.join(opt.checkpoints, opt.name, '%s_net_%s.pth' % ('best', 'E')))
        model.net_E.load_state_dict(checkpoint)
        top1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, rerank_topk=100, second_stage=False, dataset=opt.dataset)
        message = '\n Test with best model:  mAP: {:5.1%}  top1: {:5.1%}\n'.format(mAP, top1)
        print(message)

if __name__ == '__main__':
    main()
