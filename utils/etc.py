'''Some helper functions for PyTorch, including:
    - progress_bar: progress bar mimic xlua.progress.
'''

import copy
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from utils.color import Colorer, ColorerContext

C = Colorer.instance()

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 15.
last_time = time.time()
begin_time = last_time

def check_args(args):
    # --epoch
    try:
        assert args.end_epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

#----------------------------------------------------
#  Adjust_learning_rate & get_learning_rate  
#----------------------------------------------------
def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr

    for milestone in args.lr_decay_schedule:
        lr *= args.lr_decay_rate if epoch >= milestone else 1.
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def progress_bar(epoch, current, total, args, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    
    C.yellow("Epoch: [{}]".format(epoch))
    sys.stdout.write(C.cyan2("Epoch: [{}/{}]".format(epoch+1, (args.end_epoch))))
    sys.stdout.write(C.cyan2(' ['))
    for i in range(cur_len):
        sys.stdout.write(C.cyan2('='))
    sys.stdout.write(C.cyan2('>'))
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(C.cyan2(']'))

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(C.cyan2(msg))
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(C.cyan2(' %d/%d ' % (current+1, total)))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

        
def paser_config_save(args, PATH):
    temp = args
    with open(PATH+'/'+'config.json', 'w') as f:
        json.dump(temp.__dict__, f, indent=2)    
    del temp

        
def set_logging_defaults(logdir, args):
    # set basic configuration for logging
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'experiment.log')),
                                  logging.StreamHandler(os.sys.stdout)])
    # log cmdline argumetns
    logger = logging.getLogger('main')
    if is_main_process():
        logger.info(args)
        
def setup_seed(seed=0):    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


import argparse

from utils.color import Colorer
from utils.etc import check_args

C = Colorer.instance()

def parse_args():
    parser = argparse.ArgumentParser(description='Progressive Self-Knowledge Distillation : PS-KD')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--lr_decay_schedule', default=[150, 225], nargs='*', type=int, help='when to drop lr')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--end_epoch', default=3, type=int, help='number of training epoch to run')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size (default: 128), this is the total'
                                                                    'batch size of all GPUs on the current node when '
                                                                    'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--experiments_dir', type=str, default='experiments',help='Directory name to save the model, log, config')
    
    ## architecture setting
    parser.add_argument('--classifier_type', type=str, default='ResNetBeMyOwnTeacher18', choices=['ResNetBeMyOwnTeacher18', 'ResNetBeMyOwnTeacher50','resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'wideresnet50', 'wideresnet101', 'resnext50_32x4d', 'resnext101_32x8d'], help='Select classifier')
    parser.add_argument('--num_resnet_blocks', default=4, type=int, help='specifiy number of resnet blocks')
    ###
    
    parser.add_argument('--data_path', type=str, default='./datasets', help='download dataset path')
    parser.add_argument('--data_type', type=str, default='cifar100', help='type of dataset')
    parser.add_argument('--alpha_T',default=0.8 ,type=float, help='alpha_T')
    parser.add_argument('--saveckp_freq', default=299, type=int, help='Save checkpoint every x epochs. Last model saving set to 299')
    parser.add_argument('--rank', default=-1, type=int,help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,help='number of distributed processes')
    parser.add_argument('--dist_backend', default='nccl', type=str,help='distributed backend')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:8080', type=str,help='url used to set up distributed training')
    parser.add_argument('--workers', default=40, type=int, help='number of workers for dataloader')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--resume', type=str, default=None, help='load model path')

    """Our addition: BYOT + feature DML"""
    parser.add_argument('--PSKD', action='store_true', default=False, help='PSKD')
    parser.add_argument('--BYOT', action='store_true', default=False,  help='use be your own teacher')
    parser.add_argument('--DML', action='store_true', default=False,  help='use DML')

    # BYOT arguments
    parser.add_argument('--temperature', default=3, type=int,help='temperature to smooth the logits')
    parser.add_argument('--alpha', default=0.1, type=float,help='weight of kd loss')
    parser.add_argument('--beta', default=1e-6, type=float,help='weight of feature loss')
    parser.add_argument('--BYOT_from_k_block', default=1, type=int, help='Perform BYOT from Kth resnet block')

    parser.add_argument('--debug', default=False, action='store_true',help='log differently if debug')
    parser.add_argument('--val_only', default=False, action='store_true',help='log differently if debug')
    
    parser.add_argument('--DML_on_output', action='store_true', help='flag for DML on outputs; otherwise on features')
    args = parser.parse_args()
    
    if args.BYOT:
        print(C.green("Use be your own teacher method to improve accuracy!".format(args.rank)))
    if args.DML:
        print(C.green("Use DML method to improve accuracy!".format(args.rank)))
        
    print(vars(args))
    return check_args(args)