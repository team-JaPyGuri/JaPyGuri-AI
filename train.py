import argparse
import os
import random
import time

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.deeplab3plus import DeepLabv3_plus
from loss.diceBCEloss import DiceBCELoss
from dataset.make_dataset import make_loader
from logger import Logger

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


def val(epoch, model, criterion, val_loader, logger=None):
    """
    epoch: 현재 epoch
    model: 학습할 모델 객체
    criterion: 손실 함수
    val_loader: 평가 데이터를 담은 데이터 로더
    logger: 로그를 저장할 객체
    """

    model.eval()  # 모델을 평가 모드로

    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0

    num_loader = len(val_loader)

    # Confusion matrix 초기화
    confusion_mat = [[0 for _ in range(args.num_classes)] for _ in range(args.num_classes)]

    # 평가 데이터를 가져오기 위한 반복문
    with torch.no_grad():  # Disable gradient calculation
        for i, (inputs, targets) in tqdm(enumerate(val_loader), leave=False, desc='Validation {}'.format(epoch),
                                         total=len(val_loader)):
            # CUDA 사용 가능 시 GPU 사용
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 모델 출력
            output = model(inputs)
            loss = criterion(output, targets)

            # Accuracy 계산
            threshold = 0.3
            pred = output
            pred[pred > threshold] = 1
            pred[pred <= threshold] = 0

            iou_score = iou(targets, pred).item()
            dice_score = dice(targets, pred).item()

            val_loss += loss.item()
            val_iou += iou_score
            val_dice += dice_score

            """
            pred_flat = pred.flatten()
            true_flat = targets.flatten()
            # Confusion matrix 저장
            for t, p in zip(true_flat, pred_flat):
                confusion_mat[int(t.item())][int(p.item())] += 1
            """

            # 로그 히스토리 저장
            logger.add_history('total', {'loss': loss.item(), 'IoU score': iou_score,
                                         'Dice score': dice_score})

    # 히스토리 출력
    if logger is not None:
        logger('*Validation {}'.format(epoch), history_key='total', time=time.strftime('%Y.%m.%d.%H:%M:%S'))
    # Confusion matrix 출력
    if args.print_confusion_mat:
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(pd.DataFrame(confusion_mat))

    return val_loss / num_loader, val_iou / num_loader, val_dice / num_loader


def train(args, epoch, model, criterion, optimizer, train_loader, logger=None):
    """
    args: 학습 설정을 담고 있는 객체
    epoch: 현재 epoch
    model: 학습할 모델 객체
    criterion: 손실 함수
    optimizer: 최적화 알고리즘
    train_loader: 학습 데이터를 담은 데이터 로더
    logger: 로그를 저장할 객체
    """

    model.train()  # 모델을 학습 모드로

    # For print progress
    num_progress, next_print = 0, args.print_freq

    # Confusion matrix 초기화
    confusion_mat = [[0 for _ in range(args.num_classes)] for _ in range(args.num_classes)]

    train_loss = 0.0
    train_iou = 0.0
    train_dice = 0.0

    num_loader = len(train_loader)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 학습 데이터를 가져오기 위한 반복문
    for i, (inputs, targets) in enumerate(train_loader):
        # CUDA 사용 가능 시 GPU 사용
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 경사 초기화
        optimizer.zero_grad()
        # 모델 출력 값 계산
        output = model(inputs)

        # print(output.shape)
        # print("output : ",torch.max(output), torch.min(output))
        # print(output)

        # print(targets.shape)
        # print("targets : ",torch.max(targets), torch.min(targets))
        # print(targets)

        # 손실 계산
        loss = criterion(output, targets)
        # 경사 계산
        loss.backward()
        # 가중치 업데이트
        optimizer.step()

        threshold = 0.3
        pred = output
        pred[pred > threshold] = 1
        pred[pred <= threshold] = 0

        iou_score = iou(targets, pred).item()
        dice_score = dice(targets, pred).item()
        # print(iou_score, dice_score)

        train_loss += loss.item()
        train_iou += iou_score
        train_dice += dice_score

        """
        pred_flat = pred.flatten()
        true_flat = targets.flatten()
        # Confusion matrix 저장
        for t, p in zip(true_flat, pred_flat):
            confusion_mat[int(t.item())][int(p.item())] += 1
        """

        # 로그 히스토리 저장
        num_progress += len(inputs)
        logger.add_history('total', {'loss': loss.item(), 'IoU score': iou_score,
                                     'Dice score': dice_score})
        logger.add_history('batch', {'loss': loss.item(), 'IoU score': iou_score,
                                     'Dice score': dice_score})

        # 일정 주기마다 로그 히스토리 출력
        if num_progress >= next_print:
            if logger is not None:
                logger(history_key='batch', epoch=epoch, batch=num_progress, time=time.strftime('%Y.%m.%d.%H:%M:%S'))
            next_print += args.print_freq

    # 전체 로그 히스토리 및 Confusion matrix 출력
    if logger is not None:
        logger(history_key='total', epoch=epoch, lr=round(optimizer.param_groups[0]['lr'], 12))
    if args.print_confusion_mat:
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(pd.DataFrame(confusion_mat))

    return train_loss / num_loader, train_iou / num_loader, train_dice / num_loader

def run(args):
    # Random Seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # [변경] Model 설정
    model = DeepLabv3_plus(nInputChannels=3, n_classes=args.num_classes, os=16, pretrained=False, _print=False)
    if args.resume is not None:  # resume
        model.load_state_dict(torch.load(args.resume))

    # [변경] Criterion (Loss Function, 손실 함수)  설정
    criterion = DiceBCELoss()

    # [변경] Optimizer 옵티마이저  설정
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # [변경] 스케줄러 설정
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # CUDA
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

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
        train(args, epoch, model, criterion, optimizer, train_loader, logger=logger)

        # Validation
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val(epoch, model, criterion, val_loader, logger=logger)
            os.makedirs(save_dir, exist_ok=True)

            model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
            torch.save(model_state_dict, os.path.join(save_dir, '{}.pth'.format(epoch)))

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
    parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size')  # [변경]배치 사이즈
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