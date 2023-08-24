from __future__ import print_function

import argparse
import logging
import os
import time
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import loss
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed
from loader import custom_dataloader
from loss import feature_loss_function, kd_loss_function
from loss.dml_loss import DMLLoss
# from loss.pskd_loss import Custom_CrossEntropy_PSKD
from models.network import get_network
from torch.autograd import Variable
from utils.AverageMeter import AverageMeter
from utils.color import Colorer
from utils.dir_maker import DirectroyMaker
from utils.etc import (adjust_learning_rate, check_args, get_learning_rate,
                       is_main_process, paser_config_save, progress_bar,
                       save_on_master, set_logging_defaults, setup_seed, parse_args)
from utils.metric import metric_ece_aurc_eaurc, accuracy
from tensorboardX import SummaryWriter
import sys

## for debugging ## // Please comment these chuck of code when run the code
# import debugpy
# debugpy.listen(5678)
# print("wait for debugger")
# debugpy.wait_for_client()
# print("attach")
##

C = Colorer.instance()

def main():
    logger = logging.getLogger('main')
    args = parse_args()
    logger.info(C.green("[!] Start the PS-KD."))
    logger.info(C.green("[!] Created by LG CNS AI Research(LAIR), modified by CS570 Team 4"))
    
    logger.info('arguments:', vars(args))
    
    dir_maker = DirectroyMaker(root=args.experiments_dir, save_model=True, save_log=True, save_config=True)
    model_log_config_dir = dir_maker.experiments_dir_maker(args, sys.argv)
    model_dir = model_log_config_dir[0]
    log_dir = model_log_config_dir[1]
    config_dir = model_log_config_dir[2]
    tensorboard_dir = model_log_config_dir[3]
    writer = SummaryWriter(tensorboard_dir)
    
    paser_config_save(args, config_dir)
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, model_dir, log_dir, args, writer))
        print(C.green("[!] Multi/Single Node, Multi-GPU All multiprocessing_distributed Training Done."))
        print(C.underline(C.red2('[Info] Save Model dir:')), C.red2(model_dir))
        print(C.underline(C.red2('[Info] Log dir:')), C.red2(log_dir))
        print(C.underline(C.red2('[Info] Config dir:')), C.red2(config_dir))      
        print(C.underline(C.red2('[Info] Tensorboard dir:')), C.red2(tensorboard_dir))        
    else:
        print(C.green("[!] Multi/Single Node, Single-GPU per node, multiprocessing_distributed Training Done."))
        main_worker(0, ngpus_per_node, model_dir, log_dir, args, writer)
        print(C.green("[!] All Single GPU Training Done"))
        print(C.underline(C.red2('[Info] Save Model dir:')), C.red2(model_dir))
        print(C.underline(C.red2('[Info] Log dir:')), C.red2(log_dir))
        print(C.underline(C.red2('[Info] Config dir:')), C.red2(config_dir))
        print(C.underline(C.red2('[Info] Tensorboard dir:')), C.red2(tensorboard_dir)) 
    
    writer.close()

def main_worker(gpu, ngpus_per_node, model_dir, log_dir, args, writer):
    logger = logging.getLogger('main')
    
    best_acc = 0
    best_epoch = 0
    net = get_network(args)
    args.ngpus_per_node = ngpus_per_node
    args.gpu = gpu
    if args.gpu is not None:
        logger.info(C.underline(C.yellow("[Info] Using GPU: {} for training".format(args.gpu))))
    
    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    logger.info(C.green("[!] [Rank {}] Distributed Init Setting Done.".format(args.rank)))
    
    if not torch.cuda.is_available():
        logger.info(C.red2("[Warnning] Using CPU, this will be slow."))
        
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            logger.info(C.green("[!] [Rank {}] Distributed DataParallel Setting Start".format(args.rank)))
            
            torch.cuda.set_device(args.gpu)
            net.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
            args.batch_size = int(args.batch_size / args.ngpus_per_node)
            print(C.underline(C.yellow("[Info] [Rank {}] Workers: {}".format(args.rank, args.workers))))
            print(C.underline(C.yellow("[Info] [Rank {}] Batch_size: {}".format(args.rank, args.batch_size))))
            
            net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[args.gpu])
            print(C.green("[!] [Rank {}] Distributed DataParallel Setting End".format(args.rank)))
            
        else:
            net.cuda()
            net = torch.nn.parallel.DistributedDataParallel(net)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        net = torch.nn.DataParallel(net).cuda()
        
    set_logging_defaults(log_dir, args)
    train_loader, valid_loader, train_sampler = custom_dataloader.dataloader(args)
    
    """ loss objectives - start"""
    # w/ and w/out progressive loss 
    criterion_CE = nn.CrossEntropyLoss().cuda(args.gpu)
    # criterion_CE = Custom_CrossEntropy_PSKD().cuda(args.gpu) # custom CE for progressive self-knowledge distillation
    logsoftmax = nn.LogSoftmax(dim=1).cuda()
    
    # feature-level divergence loss
    criterion_BYOT = nn.KLDivLoss(reduction='batchmean').cuda(args.gpu)
    
    # feature-level, response-level deep mutual learning
    criterion_DML = DMLLoss().cuda(args.gpu)
    """ loss objectives - end"""
    
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    if args.PSKD:
        all_predictions = torch.zeros(len(train_loader.dataset), len(train_loader.dataset.classes), dtype=torch.float32)
        logger.info(C.underline(C.yellow("[Info] all_predictions matrix shape {}".format(all_predictions.shape))))
    else:
        all_predictions = None

    if args.resume:
        if args.gpu is None:
            print(f'loading model from: {args.resume}')
            checkpoint = torch.load(args.resume)
        else:
            if args.distributed:
                # Map model to be loaded to specified single gpu.
                dist.barrier()
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        
        args.start_epoch = checkpoint['epoch'] + 1 
        alpha_t = checkpoint['alpha_t']
        best_acc = checkpoint['best_acc']
        if 'prev_predictions' in checkpoint.keys() and checkpoint['prev_predictions'] is not None:
            all_predictions = checkpoint['prev_predictions'].cpu()
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(C.green("[!] [Rank {}] Model loaded".format(args.rank)))

        del checkpoint
    
    from datetime import datetime
    import time
    t1 = datetime.now()
    logger.info(f'Start time: {t1.time()}')
    
    if args.val_only:
        print(C.green(f'validating only...'))
        epoch = 0
        acc = val(
                  criterion_CE,
                  criterion_BYOT,
                  criterion_DML,                  
                  net,
                  epoch,
                  valid_loader,
                  args,
                  writer)

        #---------------------------------------------------
        #  Save_dict for saving model
        #---------------------------------------------------
        save_dict = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc' : best_acc,
                    'accuracy' : acc,
                    'alpha_t' : alpha_t,
                    'prev_predictions': all_predictions
                    }

        
        t_best = datetime.now()
        difference = t_best-t1
        total_time_up_to_now = divmod(difference.total_seconds(), 60)
        logger.info(f'End time until now: {total_time_up_to_now[0]:5.1f} min {total_time_up_to_now[1]:5.1f} sec')
        if acc > best_acc:
            logger.info(f'total time till best acc:{total_time_up_to_now[0]:5.1f} min {total_time_up_to_now[1]:5.1f} sec') # (min, seconds)
            best_acc = acc
            best_epoch = epoch
            logger.info(f'[best acc updated] best val_top1_acc.: {best_acc}')
            save_on_master(save_dict,os.path.join(model_dir, 'checkpoint_best.pth'))
            if is_main_process():
                logger.info(C.green("[!] Save best checkpoint."))

        if args.saveckp_freq and (epoch) % args.saveckp_freq == 0:
            save_on_master(save_dict,os.path.join(model_dir, f'checkpoint_{epoch:03}.pth'))
            if is_main_process():
                logger.info(C.green("[!] Save checkpoint."))

    else:        
        for epoch in range(args.start_epoch, args.end_epoch):
            
            adjust_learning_rate(optimizer, epoch, args)
            if args.distributed:
                train_sampler.set_epoch(epoch)
        
            if args.PSKD:
                alpha_t = args.alpha_T * ((epoch + 1) / args.end_epoch)
                alpha_t = max(0, alpha_t)
            else:
                alpha_t = -1 
                
            all_predictions = train(
                                    all_predictions,
                                    criterion_CE,
                                    logsoftmax,
                                    criterion_BYOT, # not used
                                    criterion_DML,
                                    optimizer,
                                    net,
                                    epoch,
                                    alpha_t,
                                    train_loader,
                                    args, 
                                    writer)


            if args.distributed:
                dist.barrier()
                    
                
            #---------------------------------------------------
            #  Validation
            #---------------------------------------------------
            acc = val(
                    criterion_CE,
                    criterion_BYOT,
                    criterion_DML,                  
                    net,
                    epoch,
                    valid_loader,
                    args,
                    writer)

            #---------------------------------------------------
            #  Save_dict for saving model
            #---------------------------------------------------
            save_dict = {
                        'net': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_acc' : best_acc,
                        'accuracy' : acc,
                        'alpha_t' : alpha_t,
                        'prev_predictions': all_predictions
                        }

            
            t_best = datetime.now()
            difference = t_best-t1
            total_time_up_to_now = divmod(difference.total_seconds(), 60)
            logger.info(f'End time until now: {total_time_up_to_now[0]:5.1f} min {total_time_up_to_now[1]:5.1f} sec')
            if acc > best_acc:
                logger.info(f'total time till best acc: {total_time_up_to_now[0]:5.1f} min {total_time_up_to_now[1]:5.1f} sec') # (min, seconds)
                best_acc = acc
                best_epoch = epoch
                logger.info(f'[best acc updated] best val_top1_acc.: {best_acc}')
                save_on_master(save_dict,os.path.join(model_dir, 'checkpoint_best.pth'))
                if is_main_process():
                    logger.info(C.green("[!] Save best checkpoint."))

            if args.saveckp_freq and (epoch) % args.saveckp_freq == 0:
                save_on_master(save_dict,os.path.join(model_dir, f'checkpoint_{epoch:03}.pth'))
                if is_main_process():
                    logger.info(C.green("[!] Save checkpoint."))


    # logger.info(f'total time till best acc: {total_time_up_to_now} at epoch [{best_epoch}]') # (min, seconds)
    t2 = datetime.now()
    difference = t2-t1
    total_time_up_to_now = divmod(difference.total_seconds(), 60)
    logger.info(f'total time: {total_time_up_to_now}')
    
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
        logger.info(C.green("[!] [Rank {}] Distroy Distributed process".format(args.rank)))



#-------------------------------
# Train 
#------------------------------- 
def train(all_predictions,
          criterion_CE,
          logsoftmax,
          criterion_BYOT,
          criterion_DML,
          optimizer,
          net,
          epoch,
          alpha_t,
          train_loader,
          args,
          writer):
    
    logger = logging.getLogger('train')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ce_losses = AverageMeter()
    top1 = AverageMeter()
    
    # loss log
    byot_feature_losses_1 = AverageMeter()
    byot_feature_losses_2 = AverageMeter()
    byot_feature_losses_3 = AverageMeter()
    
    byot_losses1_kd = AverageMeter()
    byot_losses2_kd = AverageMeter()
    byot_losses3_kd = AverageMeter()    
    byot_middle1_ce_losses = AverageMeter()
    byot_middle2_ce_losses = AverageMeter()
    byot_middle3_ce_losses = AverageMeter()
    byot_losses = AverageMeter()
    
    dml_losses = AverageMeter()
    
    # accuracy log
    middle1_top1 = AverageMeter()
    middle2_top1 = AverageMeter()
    middle3_top1 = AverageMeter()
    
    total_losses = AverageMeter()
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()
    
    correct = 0
    total = 0


    targets_list = []
    confidences = []
    
    net.train()
    current_LR = get_learning_rate(optimizer)[0]
    end = 0
    
    corrects = [0 for _ in range(args.num_resnet_blocks+1)]
    predicted = [0 for _ in range(args.num_resnet_blocks+1)]
    for batch_idx, (inputs, targets, input_indices) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        if args.gpu is not None:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True).type(torch.LongTensor).to(inputs.device)
        
        # print(inputs.shape, targets.shape)
        
        #-----------------------------------
        # Self-KD or none
        #-----------------------------------
        total_loss = 0.
        ce_loss = 0.
        byot_loss = 0.
        dml_loss = 0.
        
        # BYOT ResNet
        if args.classifier_type in ['ResNetBeMyOwnTeacher18', 'ResNetBeMyOwnTeacher50', 'resnet34', 'resnet101', 'resnet152', 'wideresnet50', 'wideresnet101', 'resnext50_32x4d', 'resnext101_32x8d'] :
            outputs, middle_output1, middle_output2, middle_output3, final_fea, middle1_fea, middle2_fea, middle3_fea = net(inputs)
        else:
            outputs = net(inputs)
        
        # output_seq = [outputs, middle_output1, middle_output2, middle_output3]
        if args.num_resnet_blocks == 3:
            output_seq = [outputs, middle_output1, middle_output2]
        elif args.num_resnet_blocks == 2:
            output_seq = [outputs, middle_output1]
        elif args.num_resnet_blocks == 1:
            output_seq = [outputs]
        else:
            output_seq = [outputs, middle_output1, middle_output2, middle_output3]
        
        softmax_predictions = F.softmax(outputs, dim=1)
        softmax_predictions = softmax_predictions.cpu().detach().numpy()
        for values_ in softmax_predictions:
            confidences.append(values_.tolist())
        
        # loss source 1: student cross entropy loss
        if args.PSKD:
            targets_numpy = targets.cpu().detach().numpy()
            identity_matrix = torch.eye(len(train_loader.dataset.classes)) 
            targets_one_hot = identity_matrix[targets_numpy]
            
            if epoch == 0:
                all_predictions[input_indices] = targets_one_hot

            # create new soft-targets
            soft_targets = ((1 - alpha_t) * targets_one_hot) + (alpha_t * all_predictions[input_indices])
            soft_targets = soft_targets.cuda()
            gt_target_labels = soft_targets
            
            log_probs = logsoftmax(outputs)
            ce_loss = (- gt_target_labels * log_probs).mean(0).sum()
            
        else:
            # train a single model without pregressive self-knowledge distillation            
            gt_target_labels = targets.to(torch.long)
            soft_targets = targets
            
            ce_loss = criterion_CE(outputs, gt_target_labels)
            
        ce_losses.update(ce_loss.item(), inputs.size(0))
        
        softmax_output = F.softmax(outputs.data, dim=1) 
        if args.distributed:
            gathered_prediction = [torch.ones_like(softmax_output) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_prediction, softmax_output)
            gathered_prediction = torch.cat(gathered_prediction, dim=0)

            gathered_indices = [torch.ones_like(input_indices.cuda()) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_indices, input_indices.cuda())
            gathered_indices = torch.cat(gathered_indices, dim=0)

        if args.BYOT:
            # BYOT loss 1: feature hint loss
           
            feature_loss_1 = feature_loss_function(middle1_fea, final_fea.detach()) if((middle1_fea != None) and (args.BYOT_from_k_block <= 1)) else 0.0
            feature_loss_2 = feature_loss_function(middle2_fea, final_fea.detach()) if((middle2_fea != None) and (args.BYOT_from_k_block <= 2)) else 0.0
            feature_loss_3 = feature_loss_function(middle3_fea, final_fea.detach()) if((middle3_fea != None) and (args.BYOT_from_k_block <= 3)) else 0.0
            byot_feature_losses_1.update(feature_loss_1, inputs.size(0)) if (args.BYOT_from_k_block <= 1) else 0.0
            byot_feature_losses_2.update(feature_loss_2, inputs.size(0)) if (args.BYOT_from_k_block <= 2) else 0.0
            byot_feature_losses_3.update(feature_loss_3, inputs.size(0)) if (args.BYOT_from_k_block <= 3) else 0.0
            
            # BYOT loss 2: multi mid-layer KD loss
            temp4 = outputs / args.temperature
            temp4 = torch.softmax(temp4, dim=1)
            loss1by4 = kd_loss_function(middle_output1, temp4.detach(), args.temperature) * (args.temperature**2) if((middle1_fea != None) and (args.BYOT_from_k_block <= 1)) else 0.0
            loss2by4 = kd_loss_function(middle_output2, temp4.detach(), args.temperature) * (args.temperature**2) if((middle2_fea != None) and (args.BYOT_from_k_block <= 2)) else 0.0
            loss3by4 = kd_loss_function(middle_output3, temp4.detach(), args.temperature) * (args.temperature**2) if((middle3_fea != None) and (args.BYOT_from_k_block <= 3)) else 0.0
            byot_losses1_kd.update(loss1by4, inputs.size(0)) if (args.BYOT_from_k_block <= 1) else 0.0
            byot_losses2_kd.update(loss2by4, inputs.size(0)) if (args.BYOT_from_k_block <= 2) else 0.0
            byot_losses3_kd.update(loss3by4, inputs.size(0)) if (args.BYOT_from_k_block <= 3) else 0.0
            
            # BYOT loss 3: CE on mid=layer outputs
            middle1_loss = criterion_CE(middle_output1, gt_target_labels) if((middle1_fea != None) and (args.BYOT_from_k_block <= 1)) else 0.0
            middle2_loss = criterion_CE(middle_output2, gt_target_labels) if((middle2_fea != None) and (args.BYOT_from_k_block <= 2)) else 0.0
            middle3_loss = criterion_CE(middle_output3, gt_target_labels) if((middle3_fea != None) and (args.BYOT_from_k_block <= 3)) else 0.0
            # log
            byot_middle1_ce_losses.update(middle1_loss.item(), inputs.size(0)) if((middle1_fea != None) and (args.BYOT_from_k_block <= 1)) else 0.0
            byot_middle2_ce_losses.update(middle2_loss.item(), inputs.size(0)) if((middle2_fea != None) and (args.BYOT_from_k_block <= 2)) else 0.0
            byot_middle3_ce_losses.update(middle3_loss.item(), inputs.size(0)) if((middle3_fea != None) and (args.BYOT_from_k_block <= 3)) else 0.0
            
            byot_loss = args.beta * (feature_loss_1 + feature_loss_2 + feature_loss_3) + \
                        args.alpha * (loss1by4 + loss2by4 + loss3by4) + \
                        (1 - args.alpha) * (ce_loss + middle1_loss + middle2_loss + middle3_loss)
            byot_losses.update(byot_loss, inputs.size(0))

        # ==============================
        # loss source: DML (our addition)
        # ==============================
        if args.DML:
            if args.DML_on_output:
                # applies DML between outputs and the outputs
                DML_targets = outputs
                # features = [outputs, outputs, outputs]
                # features = features[:(args.num_resnet_blocks-1)]
                # # feat1_DML =  [out for out in [outputs] * (args.num_resnet_blocks-1) if (out != None)]
                # feat1_DML = [out for out in features if (out != None)]
            else:
                # applies DML between middle outputs and the outputs
                DML_targets = final_fea
            features = [middle1_fea, middle2_fea, middle3_fea]
            features = features[:(args.num_resnet_blocks-1)]
            
            feat1_DML = [out for out in features if (out != None)]
            feat2_DML = [DML_targets,] * len(feat1_DML)
            
            dml_loss = criterion_DML(feat1_DML, feat2_DML)            
            dml_losses.update(dml_loss.item(), inputs.size(0))
        
        # aggregate losses
        total_loss = ce_loss
        if args.DML: total_loss += dml_loss
        if args.BYOT: total_loss += byot_loss
            
        total_losses.update(total_loss.item(), inputs.size(0))
        
        err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))
        train_top1.update(err1.item(), inputs.size(0))
        train_top5.update(err5.item(), inputs.size(0))
        
        # evaluate accuracy
        prec1 = accuracy(outputs.data, targets, topk=(1,))
        top1.update(prec1[0], inputs.size(0))

        middle1_prec1 = accuracy(middle_output1.data, targets, topk=(1,)) if(middle_output1 != None) else -1.0
        middle2_prec1 = accuracy(middle_output2.data, targets, topk=(1,)) if(middle_output2 != None) else -1.0
        middle3_prec1 = accuracy(middle_output3.data, targets, topk=(1,)) if(middle_output3 != None) else -1.0
        
        middle1_top1.update(middle1_prec1[0], inputs.size(0)) if(middle_output1 != None) else -1.0
        middle2_top1.update(middle2_prec1[0], inputs.size(0)) if(middle_output2 != None) else -1.0
        middle3_top1.update(middle3_prec1[0], inputs.size(0)) if(middle_output3 != None) else -1.0

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        
        # compute correct samples
        ensemble = sum(output_seq[:-1])/len(output_seq)
        ensemble.detach_()
        output_seq.append(ensemble)
        for i in range(len(output_seq)):
            _, predicted[i] = torch.max(output_seq[i].data, 1)
            corrects[i] += float(predicted[i].eq(targets.data).cpu().sum())
        corrects = [int(item) for item in corrects]
        
        if args.PSKD:
            if args.distributed:
                for jdx in range(len(gathered_prediction)):
                    all_predictions[gathered_indices[jdx]] = gathered_prediction[jdx].detach()
            else:
                all_predictions[input_indices] = softmax_output.cpu().detach()
        
        # progress_bar(epoch, batch_idx, len(train_loader), args, 'batch time: {:.3f} | data time: {:.3f} | lr: {:.1e} | alpha_t: {:.3f} | loss: {:.3f} | top1_acc: {:.3f} | top5_acc: {:.3f} | correct/total({}/{})'.format(
            # batch_time.val, data_time.val, current_LR, alpha_t, total_losses.avg, train_top1.avg, train_top5.avg, correct, total))
        
        
        
        correct = corrects[0]
    if args.distributed:
        dist.barrier()
    
        
        ece,aurc,eaurc = metric_ece_aurc_eaurc(confidences,
                                               targets_list,
                                               bin_size=0.1)
        
        if args.num_resnet_blocks == 4:
            progress_bar(epoch, batch_idx, len(train_loader), args, 'batch time: {:.3f} | data time: {:.3f} | lr: {:.1e} | alpha_t: {:.3f} | loss: {:.3f} | top1_acc: {:.3f} | top5_acc: {:.3f} | (4/4) {:.2f}% | (3/4) {:.2f}% | (2/4) {:.2f}% | (1/4) {:.2f}% | Ensemble: {:.2f}% [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}]'.format(
            batch_time.val, data_time.val, current_LR, alpha_t, total_losses.avg, train_top1.avg, train_top5.avg, corrects[0]/total*100, corrects[1]/total*100, corrects[2]/total*100, corrects[3]/total*100, corrects[4]/total*100, ece, aurc, eaurc))
        elif args.num_resnet_blocks == 3:
            progress_bar(epoch, batch_idx, len(train_loader), args, 'batch time: {:.3f} | data time: {:.3f} | lr: {:.1e} | alpha_t: {:.3f} | loss: {:.3f} | top1_acc: {:.3f} | top5_acc: {:.3f} | (3/3) {:.2f}% | (2/3) {:.2f}% | (1/3) {:.2f}% | Ensemble: {:.2f}% [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}]'.format(
            batch_time.val, data_time.val, current_LR, alpha_t, total_losses.avg, train_top1.avg, train_top5.avg, corrects[0]/total*100, corrects[1]/total*100, corrects[2]/total*100, corrects[3]/total*100, ece, aurc, eaurc))
        if args.num_resnet_blocks == 2:
            progress_bar(epoch, batch_idx, len(train_loader), args, 'batch time: {:.3f} | data time: {:.3f} | lr: {:.1e} | alpha_t: {:.3f} | loss: {:.3f} | top1_acc: {:.3f} | top5_acc: {:.3f} | (2/2) {:.2f}% | (1/2) {:.2f}% | Ensemble: {:.2f}% [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}]'.format(
            batch_time.val, data_time.val, current_LR, alpha_t, total_losses.avg, train_top1.avg, train_top5.avg, corrects[0]/total*100, corrects[1]/total*100, corrects[2]/total*100, ece, aurc, eaurc))
        if args.num_resnet_blocks == 1:
            progress_bar(epoch, batch_idx, len(train_loader), args, 'batch time: {:.3f} | data time: {:.3f} | lr: {:.1e} | alpha_t: {:.3f} | loss: {:.3f} | top1_acc: {:.3f} | top5_acc: {:.3f} | (1/1) {:.2f}% | Ensemble: {:.2f}% [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}]'.format(
            batch_time.val, data_time.val, current_LR, alpha_t, total_losses.avg, train_top1.avg, train_top5.avg, corrects[0]/total*100, ece, aurc, eaurc))
            
    # write to tensorboard
    writer.add_scalar('train/byot_loss', byot_losses.avg, epoch)
    writer.add_scalar('train/dml_loss', dml_losses.avg, epoch)
    writer.add_scalar('train/ce_loss', ce_losses.avg, epoch)
    writer.add_scalar('train/total_loss', total_losses.avg, epoch)    
    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/data_time',data_time.avg, epoch)
    writer.add_scalar('train/acc', correct/total, epoch)
    
    logger.info('[Rank {}] [Epoch {}] [lr {:.1e}] [alpht_t {:.3f}] [train_loss {:.3f}] [train_top1_acc {:.3f}] [train_top5_acc {:.3f}] [correct/total {}/{}]'.format(
        args.rank,
        epoch,
        current_LR,
        alpha_t,
        total_losses.avg,
        train_top1.avg,
        train_top5.avg,
        correct,
        total))
    
    return all_predictions


#-------------------------------          
# Validation
#------------------------------- 
def val(criterion_CE,
        criterion_KL,
        criterion_DML,
        net,
        epoch,
        val_loader,
        args,
        writer):

    logger = logging.getLogger('val')
    
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    val_losses = AverageMeter()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()

    targets_list = []
    confidences = []

    net.eval()
    correct = 0
    total = 0
    corrects = [0 for _ in range(args.num_resnet_blocks+1)]
    predicted = [0 for _ in range(args.num_resnet_blocks+1)]
    with torch.no_grad():
        end = 0
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):              
            data_time.update(time.time() - end)
            
            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True).type(torch.LongTensor)
                
            #for ECE, AURC, EAURC
            targets_numpy = targets.cpu().numpy()
            targets_list.extend(targets_numpy.tolist())
                
            # model output
            if args.classifier_type in ['ResNetBeMyOwnTeacher18', 'ResNetBeMyOwnTeacher50', 'resnet34', 'resnet101', 'resnet152', 'wideresnet50', 'wideresnet101', 'resnext50_32x4d', 'resnext101_32x8d'] :
                outputs, middle_output1, middle_output2, middle_output3, final_fea, middle1_fea, middle2_fea, middle3_fea = net(inputs)
            else:
                outputs = net(inputs)
            
            if args.num_resnet_blocks == 3:
                output_seq = [outputs, middle_output1, middle_output2]
            elif args.num_resnet_blocks == 2:
                output_seq = [outputs, middle_output1]
            elif args.num_resnet_blocks == 1:
                output_seq = [outputs]
            else:
                output_seq = [outputs, middle_output1, middle_output2, middle_output3]
                
            # for ECE, AURC, EAURC
            softmax_predictions = F.softmax(outputs, dim=1)
            softmax_predictions = softmax_predictions.cpu().numpy()
            for values_ in softmax_predictions:
                confidences.append(values_.tolist())
                
            batch_time.update(time.time() - end)
            end = time.time()
            
            # _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            
            # compute correct samples
            targets = targets.to(outputs.device)
            # output_seq = [outputs, middle_output1, middle_output2, middle_output3]
                    
            ensemble = sum(output_seq[:-1])/len(output_seq)
            ensemble.detach_()
            output_seq.append(ensemble)
            for i in range(len(output_seq)):
                _, predicted[i] = torch.max(output_seq[i].data, 1)
                corrects[i] += float(predicted[i].eq(targets.data).cpu().sum())
            corrects = [int(item) for item in corrects]
            
            
            # get loss
            loss = criterion_CE(outputs, targets)
            val_losses.update(loss.item(), inputs.size(0))
            
            
            # save evaluation
            #Top1, Top5 Err
            err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))
            val_top1.update(err1.item(), inputs.size(0))
            val_top5.update(err5.item(), inputs.size(0))        

            
            
            correct = corrects[0]

    if args.distributed:
        dist.barrier()
            
    if is_main_process():
        ece,aurc,eaurc = metric_ece_aurc_eaurc(confidences,
                                               targets_list,
                                               bin_size=0.1)

        
        
        if args.num_resnet_blocks == 4:
            logger.info('[Epoch {}] val_loss: {:.3f} | val_top1_acc: {:.3f} | val_top5_acc: {:.3f} | (4/4) {:.2f}% | (3/4) {:.2f}% | (2/4) {:.2f}% | (1/4) {:.2f}% | Ensemble: {:.2f}% [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}]'.format(epoch, val_losses.avg, val_top1.avg, val_top5.avg, corrects[0]/total*100, corrects[1]/total*100, corrects[2]/total*100, corrects[3]/total*100, corrects[4]/total*100, ece, aurc, eaurc))
            
        elif args.num_resnet_blocks == 3:
            logger.info('[Epoch {}] val_loss: {:.3f} | val_top1_acc: {:.3f} | val_top5_acc: {:.3f} | (3/3) {:.2f}% | (2/3) {:.2f}% | (1/3) {:.2f}% | Ensemble: {:.2f}% [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}]'.format(epoch, val_losses.avg, val_top1.avg, val_top5.avg, corrects[0]/total*100, corrects[1]/total*100, corrects[2]/total*100, corrects[3]/total*100, ece, aurc, eaurc))
            
            
        if args.num_resnet_blocks == 2:
            logger.info('[Epoch {}] val_loss: {:.3f} | val_top1_acc: {:.3f} | val_top5_acc: {:.3f} | (2/2) {:.2f}% | (1/2) {:.2f}% | Ensemble: {:.2f}% [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}]'.format(epoch, val_losses.avg, val_top1.avg, val_top5.avg, corrects[0]/total*100, corrects[1]/total*100, corrects[2]/total*100, ece, aurc, eaurc))
            
        if args.num_resnet_blocks == 1:
            logger.info('[Epoch {}] val_loss: {:.3f} | val_top1_acc: {:.3f} | val_top5_acc: {:.3f} | (1/) {:.2f}% | Ensemble: {:.2f}% [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}]'.format(epoch, val_losses.avg, val_top1.avg, val_top5.avg, corrects[0]/total*100, corrects[1]/total*100, ece, aurc, eaurc))
        
        # progressbar
        if args.num_resnet_blocks == 4:
            progress_bar(epoch, batch_idx, len(val_loader), args, 'val_loss: {:.3f} | val_top1_acc: {:.3f} | val_top5_acc: {:.3f} | (4/4) {:.2f}% | (3/4) {:.2f}% | (2/4) {:.2f}% | (1/4) {:.2f}% | Ensemble: {:.2f}% [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}]'.format(val_losses.avg, val_top1.avg, val_top5.avg, corrects[0]/total*100, corrects[1]/total*100, corrects[2]/total*100, corrects[3]/total*100, corrects[4]/total*100, ece, aurc, eaurc))
            
        elif args.num_resnet_blocks == 3:
            progress_bar(epoch, batch_idx, len(val_loader), args, 'val_loss: {:.3f} | val_top1_acc: {:.3f} | val_top5_acc: {:.3f} | (3/3) {:.2f}% | (2/3) {:.2f}% | (1/4) {:.2f}% | Ensemble: {:.2f}% [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}]'.format(val_losses.avg, val_top1.avg, val_top5.avg, corrects[0]/total*100, corrects[1]/total*100, corrects[2]/total*100, corrects[3]/total*100, ece, aurc, eaurc))
            
        if args.num_resnet_blocks == 2:
            progress_bar(epoch, batch_idx, len(val_loader), args, 'val_loss: {:.3f} | val_top1_acc: {:.3f} | val_top5_acc: {:.3f} | (2/2) {:.2f}% | (1/2) {:.2f}% | Ensemble: {:.2f}% [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}]'.format(val_losses.avg, val_top1.avg, val_top5.avg, corrects[0]/total*100, corrects[1]/total*100, corrects[2]/total*100, ece, aurc, eaurc))
            
        if args.num_resnet_blocks == 1:
            progress_bar(epoch, batch_idx, len(val_loader), args, 'val_loss: {:.3f} | val_top1_acc: {:.3f} | val_top5_acc: {:.3f} | (1/1) {:.2f}% | Ensemble: {:.2f}% [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}]'.format(val_losses.avg, val_top1.avg, val_top5.avg, corrects[0]/total*100, corrects[1]/total*100, ece, aurc, eaurc))
                
        writer.add_scalar('val/val_loss', val_losses.avg, epoch)
        writer.add_scalar('val/val_top1', val_top1.avg, epoch)
        writer.add_scalar('val/val_top5', val_top5.avg, epoch)
        writer.add_scalar('val/ece', ece, epoch)
        writer.add_scalar('val/aurc', aurc, epoch)
        writer.add_scalar('val/eaurc', eaurc, epoch)
        writer.add_scalar('val/acc', correct/total, epoch)
        writer.add_scalar('val/batch_time', batch_time.avg, epoch)
        writer.add_scalar('val/data_time',data_time.avg, epoch )
        


    return val_top1.avg

if __name__ == '__main__':
    RANDOM_SEED = 4
    import utils
    setup_seed(RANDOM_SEED)
    main()