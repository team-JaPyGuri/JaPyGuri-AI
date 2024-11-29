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


def train(args, epoch, source_feature_extractor, target_feature_extractor, 
          domain_discriminator, criterion, optimizer_t, optimizer_d, 
          train_loader_source, train_loader_target, logger=None):

    # Set mode for models
    source_feature_extractor.eval()  # Source feature extractor is frozen
    target_feature_extractor.train()
    domain_discriminator.train()

    # Metrics
    train_adversarial_loss = 0.0
    train_target_loss = 0.0
    train_acc = 0.0
    train_discriminator_acc = 0.0

    source_len = len(train_loader_source)
    target_len = len(train_loader_target)
    num_loader = target_len + source_len
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    source_iter = iter(train_loader_source)
    target_iter = iter(train_loader_target)

    for i in range(num_loader):
        # Load Source and Target Data
        source_inputs, _ = next(source_iter)  # Source images and masks
        target_inputs, _ = next(target_iter)  # Target images (no labels)
        
        source_inputs = source_inputs.to(device)
        target_inputs = target_inputs.to(device)

        # ======== Step 1: Train Discriminator ========
        optimizer_d.zero_grad()

        # Extract features
        with torch.no_grad():  # Source features are frozen
            source_features = source_feature_extractor(source_inputs)
        target_features = target_feature_extractor(target_inputs)

        # Domain labels
        source_labels = torch.ones(source_features.size(0), 1).to(device)  # Label 1 for source
        target_labels = torch.zeros(target_features.size(0), 1).to(device)  # Label 0 for target

        # Discriminator predictions
        source_preds = domain_discriminator(source_features)
        target_preds = domain_discriminator(target_features)

        # Compute domain loss
        domain_loss_source = criterion(source_preds, source_labels)
        domain_loss_target = criterion(target_preds, target_labels)
        adversarial_loss = domain_loss_source + domain_loss_target

        # Backprop and optimize discriminator
        adversarial_loss.backward()
        optimizer_d.step()

        # ======== Step 2: Train Target Feature Extractor ========
        optimizer_t.zero_grad()

        # Extract target features
        target_features = target_feature_extractor(target_inputs)
        target_preds = domain_discriminator(target_features)

        # Reverse domain labels to "confuse" discriminator
        reversed_labels = torch.ones(target_preds.size(0), 1).to(device)

        # Compute loss for target extractor
        target_loss = criterion(target_preds, reversed_labels)

        # Backprop and optimize target feature extractor
        target_loss.backward()
        optimizer_t.step()

        # ======== Metrics ========
        # Calculate accuracy for discriminator
        all_preds = torch.cat([source_preds, target_preds], dim=0)
        all_labels = torch.cat([source_labels, target_labels], dim=0)

        # Discriminator accuracy (accuracy for the whole batch)
        discriminator_acc = ((all_preds > 0.5).float() == all_labels).sum().item() / all_labels.size(0)

        # Calculate accuracy for the target extractor
        acc = ((target_preds > 0.5).float() == target_labels).sum().item() / target_labels.size(0)

        # Accumulate losses and metrics
        train_adversarial_loss += adversarial_loss.item()
        train_target_loss += target_loss.item()
        train_acc += acc
        train_discriminator_acc += discriminator_acc  # Add discriminator accuracy to the total

        # ======== Logging ========
        if logger is not None:
            logger.add_history('batch', {
                'adversarial_loss': adversarial_loss.item(),
                'target_loss': target_loss.item(),
                'accuracy': acc,
                'discriminator_accuracy': discriminator_acc  # Log discriminator accuracy
            })

        if i % args.print_freq == 0 and logger is not None:
            logger(history_key='batch', epoch=epoch, batch=i, time=time.strftime('%Y.%m.%d.%H:%M:%S'))

    # ======== Final Logging ========
    if logger is not None:
        logger(history_key='total', epoch=epoch, 
               lr_t=round(optimizer_t.param_groups[0]['lr'], 12), 
               lr_d=round(optimizer_d.param_groups[0]['lr'], 12))

    # Normalize metrics
    avg_adversarial_loss = train_adversarial_loss / num_loader
    avg_target_loss = train_target_loss / target_len
    avg_acc = train_acc / target_len
    avg_discriminator_acc = train_discriminator_acc / num_loader  

    return avg_adversarial_loss, avg_target_loss, avg_acc, avg_discriminator_acc


def val(args, epoch, source_feature_extractor, target_feature_extractor, 
             domain_discriminator, criterion, val_loader_source, val_loader_target, logger=None):
    # Set mode for models
    source_feature_extractor.eval()  # Set source feature extractor to evaluation mode
    target_feature_extractor.eval()  # Set target feature extractor to evaluation mode
    domain_discriminator.eval()      # Set domain discriminator to evaluation mode

    # Metrics
    val_adversarial_loss = 0.0
    val_target_loss = 0.0
    val_acc = 0.0
    val_discriminator_acc = 0.0

    source_len = len(val_loader_source)
    target_len = len(val_loader_target)
    num_loader = target_len + source_len
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    source_iter = iter(val_loader_source)
    target_iter = iter(val_loader_target)

    with torch.no_grad():  # Disable gradients during validation
        for i in range(num_loader):
            # Load Source and Target Data
            source_inputs, _ = next(source_iter)  # Source images and masks
            target_inputs, _ = next(target_iter)  # Target images (no labels)
            
            source_inputs = source_inputs.to(device)
            target_inputs = target_inputs.to(device)

            # ======== Step 1: Forward pass through Discriminator ========
            # Extract features
            source_features = source_feature_extractor(source_inputs)
            target_features = target_feature_extractor(target_inputs)

            # Domain labels
            source_labels = torch.ones(source_features.size(0), 1).to(device)  # Label 1 for source
            target_labels = torch.zeros(target_features.size(0), 1).to(device)  # Label 0 for target

            # Discriminator predictions
            source_preds = domain_discriminator(source_features)
            target_preds = domain_discriminator(target_features)

            # Compute domain loss
            domain_loss_source = criterion(source_preds, source_labels)
            domain_loss_target = criterion(target_preds, target_labels)
            adversarial_loss = domain_loss_source + domain_loss_target

            # ======== Step 2: Forward pass through Target Feature Extractor ========
            target_preds = domain_discriminator(target_features)

            # Reverse domain labels to "confuse" discriminator
            reversed_labels = torch.ones(target_preds.size(0), 1).to(device)

            # Compute loss for target extractor
            target_loss = criterion(target_preds, reversed_labels)

            # ======== Metrics ========
            # Calculate accuracy for discriminator
            all_preds = torch.cat([source_preds, target_preds], dim=0)
            all_labels = torch.cat([source_labels, target_labels], dim=0)

            # Discriminator accuracy (accuracy for the whole batch)
            discriminator_acc = ((all_preds > 0.5).float() == all_labels).sum().item() / all_labels.size(0)

            # Calculate accuracy for the target extractor
            acc = ((target_preds > 0.5).float() == reversed_labels).sum().item() / reversed_labels.size(0)

            # Accumulate losses and metrics
            val_adversarial_loss += adversarial_loss.item()
            val_target_loss += target_loss.item()
            val_acc += acc
            val_discriminator_acc += discriminator_acc  # Add discriminator accuracy to the total

            # 로그 히스토리 저장
            logger.add_history('total', {'adversarial_loss': adversarial_loss.item(), 'discriminator acc': discriminator_acc,
                                            'target loss': target_loss.item(), 'target acc': acc})


    # 히스토리 출력
    if logger is not None:
        logger('*Validation {}'.format(epoch), history_key='total', time=time.strftime('%Y.%m.%d.%H:%M:%S'))

    # Normalize metrics
    avg_adversarial_loss = val_adversarial_loss / num_loader
    avg_target_loss = val_target_loss / target_len
    avg_acc = val_acc / target_len
    avg_discriminator_acc = val_discriminator_acc / num_loader  # Average discriminator accuracy

    return avg_adversarial_loss, avg_target_loss, avg_acc, avg_discriminator_acc




def run(args):
    # Random Seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    model_s = DeepLabv3_plus_extractor(nInputChannels=3, os=16, pretrained=False)
    model_t = DeepLabv3_plus_extractor(nInputChannels=3, os=16, pretrained=False)
    model_d = discriminator(304)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model_s = torch.nn.DataParallel(model_s).to(device)
            model_t = torch.nn.DataParallel(model_t).to(device)
            model_d = torch.nn.DataParallel(model_d).to(device)
        else:
            model_s = model_s.to(device)
            model_t = model_t.to(device)
            model_d = model_d.to(device)
    else:
        model_s = model_s.to(device)
        model_t = model_t.to(device)
        model_d = model_d.to(device)

    criterion = nn.BCELoss()

    # [변경] Optimizer 옵티마이저  설정
    optimizer_t = torch.optim.Adam(model_t.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=args.lr)

    # [변경] 스케줄러 설정
    scheduler_t = torch.optim.lr_scheduler.StepLR(optimizer_t, step_size=40, gamma=0.5)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=40, gamma=0.5)

    

    # Dataset
    source_train_loader, source_val_loader = make_loader(batch_size=args.batch_size)
    target_train_loader, target_val_loader = make_loader(batch_size=args.batch_size) ### target 데이터 불러오는 것 구현해야함
    

    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=args.epochs, dataset_size=len(source_train_loader.dataset)+len(target_train_loader.dataset), float_round=5)
    logger.set_sort(['loss', 'accuracy', 'lr', 'time'])
    logger(str(args))

    # Run
    save_dir = os.path.join(args.result, 'checkpoints')
    for epoch in range(args.epochs):
        # Train
        train(args, epoch, model_s, model_t, model_d, criterion, optimizer_t, optimizer_d, 
          source_train_loader, target_train_loader, logger=None)

        # Validation
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val(args, epoch, model_s, model_t, 
             model_d, criterion, source_val_loader, target_val_loader, logger=None)

            os.makedirs(save_dir, exist_ok=True)

            model_state_dict_t = model_t.module.state_dict() if torch.cuda.device_count() > 1 else model_t.state_dict()
            torch.save(model_state_dict_t, os.path.join(save_dir, 'f_{}.pth'.format(epoch)))
            model_state_dict_d = model_d.module.state_dict() if torch.cuda.device_count() > 1 else model_d.state_dict()
            torch.save(model_state_dict_d, os.path.join(save_dir, 'c_{}.pth'.format(epoch)))

        # Scheduler Step
        scheduler_t.step()
        scheduler_d.step()



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