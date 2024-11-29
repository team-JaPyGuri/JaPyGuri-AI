import argparse
import os
import random
import time

import torch
import wandb

from dataset.make_dataset import *
from logger import Logger
from model.autoencoder import *
from torch.nn import BCELoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from loss.diceBCEloss import *

def dice(y_true, y_pred):
    num = y_true.size(0)
    eps = 1e-7

    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)

    intersection = (y_true_flat * y_pred_flat).sum(1)

    score = (2 * intersection) / (y_true_flat.sum(1) + y_pred_flat.sum(1) + eps)
    score = score.sum() / num
    return score


def val(args, epoch, encoder, decoder, criterion, val_loader, logger=None):
    encoder.eval()  
    decoder.eval() 

    val_loss = 0.0
    val_dice = 0.0

    num_loader = len(val_loader)
    device = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")

    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            latent, size = encoder(inputs)
            output = decoder(latent, size)

            loss = criterion(output, targets).item()
            val_loss += loss
            
            score = dice(output, targets).item()
            val_dice += score

            if logger:
                logger.add_history('total', {'loss': loss, 'Dice score': score})

    if logger:
        logger('*Validation {}'.format(epoch), history_key='total', time=time.strftime('%Y.%m.%d.%H:%M:%S'))

    return val_loss / num_loader, val_dice / num_loader


def train(args, epoch, encoder, decoder, criterion, optimizer, train_loader, logger=None):
    encoder.train()  
    decoder.train() 

    train_loss = 0.0
    train_dice = 0.0

    num_loader = len(train_loader)
    device = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")

    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        latent, size = encoder(inputs)
        output = decoder(latent, size)

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        score = dice(output, targets).item()
        train_dice += score

        if logger:
            logger.add_history('total', {'loss': loss.item(), 'Dice score': score})
            logger.add_history('batch', {'loss': loss.item(), 'Dice score': score})

    if logger:
        logger(history_key='total', epoch=epoch, lr=round(optimizer.param_groups[0]['lr'], 12))

    return train_loss / num_loader, train_dice / num_loader


def run(args):

    # Random Seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")
    
    base_path = "/mnt/hgh/JaPyGuri-AI/pretrained_parameter/"
    if args.patch:
        path = f"{base_path}noise_patch_ssl/noise_patch_{args.load_ssl_param}.pth"
    else:
        path = f"{base_path}noise_ssl/noise_ssl_{args.load_ssl_param}.pth"


    model_encoder = encoder(in_channel=3)
    # model_encoder.load_state_dict(torch.load(path))
    # print("Complete loading parameter")
    
    model_decoder = decoder(out_channel=1)

    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(list(model_encoder.parameters()) + list(model_decoder.parameters()), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # CUDA
    model_encoder.to(device)
    model_decoder.to(device)

    # Dataset
    train_loader, val_loader = make_source_loader(batch_size=args.batch_size)
    
    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=args.epochs, dataset_size=len(train_loader.dataset), float_round=5)
    logger.set_sort(['loss', 'lr', 'time'])
    logger(str(args))
    
    # Initialize wandb
    wandb.init(project="nail_segmentation", name=f"experiment_{time.strftime('%Y%m%d%H%M%S')}", config=args)
    wandb.watch(model_encoder)
    wandb.watch(model_decoder)

    # Run
    save_dir = os.path.join(args.result, 'checkpoints')
    for epoch in range(args.epochs):
        # Train
        train_loss, train_dice = train(args, epoch, model_encoder, model_decoder, criterion, optimizer, train_loader, logger)
        
        # Validation
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val_loss, val_dice = val(args, epoch, model_encoder, model_decoder, criterion, val_loader, logger)
            os.makedirs(save_dir, exist_ok=True)

            if epoch % 2 == 1:
                # Save models
                torch.save(model_encoder.state_dict(), os.path.join(save_dir, f'e_{epoch}.pth'))
                torch.save(model_decoder.state_dict(), os.path.join(save_dir, f'd_{epoch}.pth'))
        
        # Log to wandb
        wandb.log({
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        # Scheduler Step
        scheduler.step()

    wandb.finish()


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
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')  
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate',
                        dest='lr')  
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    # Validation and Debugging Arguments
    parser.add_argument('--val_freq', default=1, type=int, help='validation frequency')
    parser.add_argument('--print_freq', default=1000, type=int, help='print frequency')
    parser.add_argument('--print_confusion_mat', default=False,
                        action='store_true')  # 혼동 행렬 출력
    parser.add_argument('--result', default='downstream_segmentor', type=str, help='path to results')
    parser.add_argument('--tag', default=None, type=str)
    
    parser.add_argument("--patch", type=bool, default=True , help="Specify if patch mode is True or False")
    parser.add_argument("--load_ssl_param", type=int, default=249,  choices=[51, 101, 151, 199,249,299,349,399,449,499],
                        help="Select the SSL parameter to load (choices: 51, 101, 151, 199)")
    parser.add_argument("--cuda_num", type=int, default=0, help="Specify the CUDA device number.")
    
    
    args = parser.parse_args()

    # Paths setting
    args.data = os.path.expanduser(args.data)
    args.result = os.path.expanduser(args.result)
    args.result = os.path.join(args.result, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    if args.tag is not None:
        args.result = '{}_{}'.format(args.result, args.tag)
    os.makedirs(args.result, exist_ok=True)

    run(args)
