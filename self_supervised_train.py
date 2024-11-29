import argparse
import os
import random
import time

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.make_dataset import *
from logger import Logger
from model.unet import *
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def val(args, epoch, feature_extractor, constructor, criterion, val_loader, logger=None):
    feature_extractor.eval()  
    constructor.eval() 

    num_progress, next_print = 0, args.print_freq
    train_loss = 0.0
    num_loader = len(val_loader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        latent, size = feature_extractor(inputs)
        output = constructor(latent, size)

        loss = criterion(output, targets)
        train_loss += loss.item()

        num_progress += len(inputs)
        logger.add_history('total', {'loss': loss.item()})
        logger.add_history('batch', {'loss': loss.item()})

        if num_progress >= next_print:
            if logger is not None:
                logger(history_key='batch', epoch=epoch, batch=num_progress, time=time.strftime('%Y.%m.%d.%H:%M:%S'))
            next_print += args.print_freq

    if logger is not None:
        logger('*Validation {}'.format(epoch), history_key='total', time=time.strftime('%Y.%m.%d.%H:%M:%S'))

    return train_loss / num_loader

def train(args, epoch, feature_extractor, constructor, criterion, optimizer, train_loader, logger=None):
    feature_extractor.train()  
    constructor.train() 

    num_progress, next_print = 0, args.print_freq
    train_loss = 0.0
    num_loader = len(train_loader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        latent, feature = feature_extractor(inputs)
        output = constructor(latent, feature)

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        num_progress += len(inputs)
        logger.add_history('total', {'loss': loss.item()})
        logger.add_history('batch', {'loss': loss.item()})

        if num_progress >= next_print:
            if logger is not None:
                logger(history_key='batch', epoch=epoch, batch=num_progress, time=time.strftime('%Y.%m.%d.%H:%M:%S'))
            next_print += args.print_freq

    if logger is not None:
        logger(history_key='total', epoch=epoch, lr=round(optimizer.param_groups[0]['lr'], 12))

    return train_loss / num_loader

def run(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=args, name=args.run_name)

    # Random Seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    feature_ext = feature_extractor(in_channel=3)
    construct = reconstructor(out_channel=3)

    criterion = MSELoss()
    optimizer = torch.optim.Adam(list(feature_ext.parameters()) + list(construct.parameters()), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # CUDA
    feature_ext.to(device)
    construct.to(device)

    # Dataset
    train_loader, val_loader = make_target_loader(batch_size=args.batch_size, noise_factor=0.25)

    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=args.epochs, dataset_size=len(train_loader.dataset), float_round=5)
    logger.set_sort(['loss', 'lr', 'time'])
    logger(str(args))

    # Run
    save_dir = os.path.join(args.result, 'checkpoints')
    for epoch in range(args.epochs):
        # Train
        train_loss = train(args, epoch, feature_ext, construct, criterion, optimizer, train_loader, logger)

        # Validation
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val_loss = val(args, epoch, feature_ext, construct, criterion, val_loader, logger)
            os.makedirs(save_dir, exist_ok=True)

            model_state_dict = feature_ext.module.state_dict() if torch.cuda.device_count() > 1 else feature_ext.state_dict()
            if epoch % 2 == 1:
                torch.save(model_state_dict, os.path.join(save_dir, '{}.pth'.format(epoch)))

        # Scheduler Step
        scheduler.step()
        if args.use_wandb and epoch % args.val_freq == 0:
            wandb.log({'training_loss': train_loss, 'lr': optimizer.param_groups[0]['lr'],
                       'validation_loss': val_loss,})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Model Arguments
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')  # 가중치 로드
    # Data Arguments
    parser.add_argument('--data', default='./Data/Qupath2/patch', help='path to dataset')  # 데이터셋 경로
    parser.add_argument('--mask_dir', default='./Data/Qupath2/mask', help='path to mask dir')  # 마스크 디렉토리 경로
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    # Training Arguments
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run') 
    parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size')  
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate',
                        dest='lr')  
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    # Validation and Debugging Arguments
    parser.add_argument('--val_freq', default=1, type=int, help='validation frequency')
    parser.add_argument('--print_freq', default=1000, type=int, help='print frequency')
    parser.add_argument('--print_confusion_mat', default=False,
                        action='store_true')  # 혼동 행렬 출력
    parser.add_argument('--result', default='unet_SSL', type=str, help='path to results')
    parser.add_argument('--tag', default=None, type=str)
    # WandB Arguments
    parser.add_argument('--use_wandb', action='store_true', help='whether to use WandB for logging')
    parser.add_argument('--wandb_project', default='unet_SSL', type=str, help='WandB project name')
    parser.add_argument('--run_name', default=None, type=str, help='WandB run name')
    args = parser.parse_args()

    # Paths setting
    args.data = os.path.expanduser(args.data)
    args.result = os.path.expanduser(args.result)
    args.result = os.path.join(args.result, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    if args.tag is not None:
        args.result = '{}_{}'.format(args.result, args.tag)
    os.makedirs(args.result, exist_ok=True)

    run(args)
