import argparse
import os
import random
import time

import pandas as pd
import torch
from tqdm import tqdm

from model.deeplab3plus import *
from loss.diceBCEloss import DiceBCELoss
from dataset.make_dataset import *
from logger import Logger
from model.autoencoder import *
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def dice(y_true, y_pred):
    num = y_true.size(0)
    eps = 1e-7

    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)

    intersection = (y_true_flat * y_pred_flat).sum(1)

    score = (2 * intersection) / (y_true_flat.sum(1) + y_pred_flat.sum(1) + eps)
    score = score.sum() / num
    return score


@torch.no_grad()
def val(args, epoch, enc, dec, criterion, val_loader, logger=None):
    enc.eval()
    dec.eval()

    # Metrics
    val_loss = 0.0
    val_dice = 0.0

    num_loader = len(val_loader)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    for i, (inputs, masks) in enumerate(val_loader):
        inputs, masks = inputs.to(device), masks.to(device)

        # Forward Pass
        latent, size = enc(inputs)  # Feature extraction
        output = dec(latent, size)  # Segmentation prediction

        # Calculate segmentation loss
        loss = criterion(output, masks)

        # Calculate metrics
        dice_score = dice(masks, output).item()

        # Accumulate losses and metrics
        val_loss += loss.item()
        val_dice += dice_score

        # 로그 히스토리 저장
        logger.add_history('total', {'loss': loss.item(), 'Dice score': dice_score})

    # 평균 값 계산
    avg_loss = val_loss / num_loader
    avg_dice = val_dice / num_loader

        # 히스토리 출력
    if logger is not None:
        logger('*Validation {}'.format(epoch), history_key='total', time=time.strftime('%Y.%m.%d.%H:%M:%S'))

    return avg_loss, avg_dice



def train(args, epoch, enc, dec, criterion, 
          optimizer, train_loader, logger=None):

    enc.train()
    dec.train()

    # Metrics
    train_loss = 0.0
    train_dice = 0.0

    num_progress, next_print = 0, args.print_freq

    num_loader = len(train_loader)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    for i, (inputs, masks) in enumerate(train_loader):
        # Load Source and Target Data
        inputs, masks = inputs.to(device), masks.to(device)

        # Zero Gradients
        optimizer.zero_grad()

        ### Step 1: Train Segmentation Model (Source Domain Only)
        latent, size = enc(inputs)  # Feature extraction
        output = dec(latent, size)  # Segmentation prediction

        # Calculate segmentation loss
        segmentation_loss = criterion(output, masks)
        segmentation_loss.backward()
        optimizer.step()

        dice_score = dice(masks, output).item()

        # Accumulate losses and metrics
        train_loss += segmentation_loss.item()
        train_dice += dice_score

        # 로그 히스토리 저장
        num_progress += len(inputs)
        logger.add_history('total', {'loss': segmentation_loss.item(), 'Dice score': dice_score})
        logger.add_history('batch', {'loss': segmentation_loss.item(), 'Dice score': dice_score})

        # 일정 주기마다 로그 히스토리 출력
        if num_progress >= next_print:
            if logger is not None:
                logger(history_key='batch', epoch=epoch, batch=num_progress, time=time.strftime('%Y.%m.%d.%H:%M:%S'))
            next_print += args.print_freq

    # 전체 로그 히스토리
    if logger is not None:
        logger(history_key='total', epoch=epoch, 
           lr_f=round(optimizer.param_groups[0]['lr'], 12)
        )

    
    return (train_loss / num_loader, train_dice / num_loader)



def run(args):

    # Random Seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 정의
    enc = encoder(in_channel=3)
    path = "/Users/Han/Desktop/capstone/JaPyGuri-AI/pre_trained_parameter/noise_ssl/noise_ssl_101.pth"
    enc.load_state_dict(torch.load(path, map_location=torch.device(device)))
    dec = decoder(out_channel=1)

    # 모델을 GPU로 보내기
    enc = enc.to(device)
    dec = dec.to(device)

    # 로스 설정
    criterion = DiceBCELoss()
    # [변경] Optimizer 옵티마이저  설정
    optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)




    # Dataset
    train_loader, val_loader = make_source_loader(batch_size=args.batch_size)

    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=args.epochs, dataset_size=len(train_loader.dataset), float_round=5)
    logger.set_sort(['loss', 'accuracy', 'lr', 'time'])
    logger(str(args))

    # Run
    save_dir = os.path.join(args.result, 'checkpoints')
    for epoch in range(args.epochs):
        # Train
        train(args, epoch, enc, dec, criterion, optimizer, train_loader, logger)

        # Validation
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val(args, epoch, enc, dec, criterion, val_loader, logger)

            os.makedirs(save_dir, exist_ok=True)

            if epoch % 2 == 1:
                model_state_dict_e = enc.module.state_dict() 
                torch.save(model_state_dict_e, os.path.join(save_dir, 'f_{}.pth'.format(epoch)))
                model_state_dict_d = dec.module.state_dict() 
                torch.save(model_state_dict_d, os.path.join(save_dir, 'c_{}.pth'.format(epoch)))

        # Scheduler Step
        scheduler.step()



if __name__ == '__main__':
    # Arguments 설정
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Model Arguments
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')  # [변경] 데이터의 클래스 종류의 수
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')  # 가중치 로드 (이어서)
    # Data Arguments
    parser.add_argument('--data', default='./Data/Qupath2/patch', help='path to dataset')  # [변경] 이미지 패치 저장 경로
    parser.add_argument('--mask_dir', default='./Data/Qupath2/mask', help='path to mask dir')  # [변경] 패치 마스크 저장 경로
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    # Training Arguments
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')  # [변경]훈련 반복 수
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')  # [변경]배치 사이즈
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate',
                        dest='lr')  # [변경] 초기 Learning rate
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    # Validation and Debugging Arguments
    parser.add_argument('--val_freq', default=1, type=int, help='validation frequency')
    parser.add_argument('--print_freq', default=1000, type=int, help='print frequency')
    parser.add_argument('--print_confusion_mat', default=False,
                        action='store_true')  # [변경] Validation이 끝날 때 Confusion Matrix 출력 여부.
    parser.add_argument('--result', default='noise_SSL_finetune', type=str, help='path to results')
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