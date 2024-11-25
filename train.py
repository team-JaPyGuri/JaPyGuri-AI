import argparse
import os
import random
import time

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.deeplab3plus import *
from loss.diceBCEloss import DiceBCELoss
from dataset.make_dataset import make_loader
from logger import Logger
from model.unet import unet

def iou(y_true, y_pred):
    num = y_true.size(0)
    eps = 1e-7

    y_true_flat = y_true.view(num, -1)  # flatten 과 유사하게 [batch, whole pixel] 형태로 바꾸는 역할
    y_pred_flat = y_pred.view(num, -1)

    intersection = (y_true_flat * y_pred_flat).sum(1)  # 교집합
    union = ((y_true_flat + y_pred_flat) > 0.0).float().sum(1)  # 합집합

    score = (intersection) / (union + eps)
    score = score.sum() / num
    return score


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
def val(args, epoch, feature_extractor, constructer, criterion, val_loader, logger=None):
    feature_extractor.eval()
    constructer.eval()

    # Metrics
    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0

    num_loader = len(val_loader)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    for i, (inputs, masks) in enumerate(val_loader):
        inputs, masks = inputs.to(device), masks.to(device)

        # Forward Pass
        latent = feature_extractor(inputs)  # Feature extraction
        output = constructer(latent, inputs)  # Segmentation prediction

        # Calculate segmentation loss
        loss = criterion(output, masks)

        # Calculate metrics
        iou_score = iou(masks, output).item()
        dice_score = dice(masks, output).item()

        # Accumulate losses and metrics
        val_loss += loss.item()
        val_iou += iou_score
        val_dice += dice_score

        # 로그 히스토리 저장
        logger.add_history('total', {'loss': loss.item(), 'IoU score': iou_score,
                                         'Dice score': dice_score})

    # 평균 값 계산
    avg_loss = val_loss / num_loader
    avg_iou = val_iou / num_loader
    avg_dice = val_dice / num_loader

        # 히스토리 출력
    if logger is not None:
        logger('*Validation {}'.format(epoch), history_key='total', time=time.strftime('%Y.%m.%d.%H:%M:%S'))

    return avg_loss, avg_iou, avg_dice



def train(args, epoch, feature_extractor, constructer, criterion, 
          optimizer_f, optimizer_c, train_loader_source, logger=None):

    feature_extractor.train()
    constructer.train()

    # Metrics
    train_loss = 0.0
    train_iou = 0.0
    train_dice = 0.0

    num_progress, next_print = 0, args.print_freq

    num_loader = len(train_loader_source)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    source_iter = iter(train_loader_source)

    for i in range(num_loader):
        # Load Source and Target Data
        inputs, masks = next(source_iter)
        inputs, masks = inputs.to(device), masks.to(device)

        # Zero Gradients
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()

        ### Step 1: Train Segmentation Model (Source Domain Only)
        latent = feature_extractor(inputs)  # Feature extraction
        output = constructer(latent, inputs)  # Segmentation prediction

        # Calculate segmentation loss
        segmentation_loss = criterion(output, masks)
        segmentation_loss.backward()
        optimizer_f.step()
        optimizer_c.step()

        iou_score = iou(masks, output).item()
        dice_score = dice(masks, output).item()

        # Accumulate losses and metrics
        train_loss += segmentation_loss.item()
        train_iou += iou_score
        train_dice += dice_score

        # 로그 히스토리 저장
        num_progress += len(inputs)
        logger.add_history('total', {'loss': segmentation_loss.item(), 'IoU score': iou_score,
                                     'Dice score': dice_score})
        logger.add_history('batch', {'loss': segmentation_loss.item(), 'IoU score': iou_score,
                                     'Dice score': dice_score})

        # 일정 주기마다 로그 히스토리 출력
        if num_progress >= next_print:
            if logger is not None:
                logger(history_key='batch', epoch=epoch, batch=num_progress, time=time.strftime('%Y.%m.%d.%H:%M:%S'))
            next_print += args.print_freq

    # 전체 로그 히스토리
    if logger is not None:
        logger(history_key='total', epoch=epoch, 
           lr_f=round(optimizer_f.param_groups[0]['lr'], 12), 
           lr_c=round(optimizer_c.param_groups[0]['lr'], 12)
        )

    
    return (train_loss / num_loader, train_iou / num_loader, train_dice / num_loader)



def run(args):
    # Random Seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # 모델 정의
    model_f = DeepLabv3_plus_extractor(nInputChannels=3, os=16, pretrained=True)
    model_c = constructer(n_classes=args.num_classes)

    # CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델을 GPU로 보내기
    if torch.cuda.is_available():
        # 여러 GPU가 있다면 DataParallel을 사용
        if torch.cuda.device_count() > 1:
            model_f = torch.nn.DataParallel(model_f).to(device)
            model_c = torch.nn.DataParallel(model_c).to(device)
        else:
            model_f = model_f.to(device)
            model_c = model_c.to(device)
    else:
        # CUDA가 없으면 CPU로 모델을 로드
        model_f = model_f.to(device)
        model_c = model_c.to(device)

    # 로스 설정
    criterion = DiceBCELoss()
    # [변경] Optimizer 옵티마이저  설정
    optimizer_f = torch.optim.Adam(model_f.parameters(), lr=args.lr)
    optimizer_c = torch.optim.Adam(model_c.parameters(), lr=args.lr)

    # [변경] 스케줄러 설정
    scheduler_f = torch.optim.lr_scheduler.StepLR(optimizer_f, step_size=40, gamma=0.5)
    scheduler_c = torch.optim.lr_scheduler.StepLR(optimizer_c, step_size=40, gamma=0.5)


    # Dataset
    train_loader, val_loader = make_loader(batch_size=args.batch_size)

    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=args.epochs, dataset_size=len(train_loader.dataset), float_round=5)
    logger.set_sort(['loss', 'accuracy', 'lr', 'time'])
    logger(str(args))

    # Run
    save_dir = os.path.join(args.result, 'checkpoints')
    for epoch in range(args.epochs):
        # Train
        train(args, epoch, model_f, model_c, criterion, optimizer_f, optimizer_c, train_loader, logger)

        # Validation
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val(args, epoch, model_f, model_c, criterion, val_loader, logger)

            os.makedirs(save_dir, exist_ok=True)

            model_state_dict_f = model_f.module.state_dict() if torch.cuda.device_count() > 1 else model_f.state_dict()
            torch.save(model_state_dict_f, os.path.join(save_dir, 'f_{}.pth'.format(epoch)))
            model_state_dict_c = model_c.module.state_dict() if torch.cuda.device_count() > 1 else model_c.state_dict()
            torch.save(model_state_dict_c, os.path.join(save_dir, 'c_{}.pth'.format(epoch)))

        # Scheduler Step
        scheduler_f.step()
        scheduler_c.step()



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
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')  # [변경]훈련 반복 수
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')  # [변경]배치 사이즈
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate',
                        dest='lr')  # [변경] 초기 Learning rate
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    # Validation and Debugging Arguments
    parser.add_argument('--val_freq', default=1, type=int, help='validation frequency')
    parser.add_argument('--print_freq', default=1000, type=int, help='print frequency')
    parser.add_argument('--print_confusion_mat', default=False,
                        action='store_true')  # [변경] Validation이 끝날 때 Confusion Matrix 출력 여부.
    parser.add_argument('--result', default='results_segmentor', type=str, help='path to results')
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