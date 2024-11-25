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
from model.autoencoder import *
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts




def val(args, epoch, encoder, decoder, criterion, val_loader, logger=None):

    encoder.eval()  
    decoder.eval() 

    num_progress, next_print = 0, args.print_freq


    train_loss = 0.0

    num_loader = len(val_loader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)


        latent, size = encoder(inputs)
        output = decoder(latent, size)

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

def train(args, epoch, encoder, decoder, criterion, optimizer, train_loader, logger=None):

    encoder.train()  
    decoder.train() 

    num_progress, next_print = 0, args.print_freq

    train_loss = 0.0

    num_loader = len(train_loader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        latent, size = encoder(inputs)
        output = decoder(latent, size)
        # print(f'input : {inputs.shape}, output : {output.shape}, target : {targets.shape}')
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
    # Random Seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    model_encoder = encoder(in_channel=3)
    model_decoder = decoder(out_channel=3)
    if args.resume is not None:  # resume
        model_encoder.load_state_dict(torch.load(args.resume)) # load encoder
        model_decoder.load_state_dict(torch.load(args.resume)) # load decoder

    criterion = MSELoss()
    optimizer = torch.optim.Adam(list(model_encoder.parameters()) + list(model_decoder.parameters()), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # CUDA
    model_encoder.to(device)
    model_decoder.to(device)


    # Dataset
    train_loader, val_loader = make_target_loader(batch_size=args.batch_size, patch_flag=True)
    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=args.epochs, dataset_size=len(train_loader.dataset), float_round=5)
    logger.set_sort(['loss', 'lr', 'time'])
    logger(str(args))

    # Run
    save_dir = os.path.join(args.result, 'checkpoints')
    for epoch in range(args.epochs):
        # Train
        train(args, epoch, model_encoder, model_decoder, criterion, optimizer, train_loader, logger)

        # Validation
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val(args, epoch, model_encoder, model_decoder, criterion, val_loader,logger)
            os.makedirs(save_dir, exist_ok=True)

            model_state_dict = model_encoder.module.state_dict() if torch.cuda.device_count() > 1 else model_encoder.state_dict()
            if epoch % 2 == 1:
                torch.save(model_state_dict, os.path.join(save_dir, '{}.pth'.format(epoch)))

        # Scheduler Step
        scheduler.step()



if __name__ == '__main__':
    # print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

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
    parser.add_argument('--result', default='noise_patch_ssl', type=str, help='path to results')
    parser.add_argument('--tag', default=None, type=str)
    args = parser.parse_args()


    # Paths setting
    args.data = os.path.expanduser(args.data)
    args.result = os.path.expanduser(args.result)
    args.result = os.path.join(args.result, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    if args.tag is not None:
        args.result = '{}_{}'.format(args.result, args.tag)
    os.makedirs(args.result, exist_ok=True)

    run(args)