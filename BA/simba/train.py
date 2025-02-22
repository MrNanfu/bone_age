# -*- coding: utf-8 -*-

"""
Bone Age Assessment SIMBA train routine.
"""

# Standard lib imports
import os
import csv
import time
import argparse
import warnings
import pandas as pd
import os.path as osp

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # 添加 TensorBoard

# Other imports
from tqdm import tqdm

# Local imports
from models.simba import SIMBA
from data.data_loader import BoneageDataset as Dataset
from utils import AverageMeter
from utils import metric_average


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

# Model settings
parser.add_argument('--relative-age', default=False, action='store_true',
                help='Train model with relative age')
parser.add_argument('--chronological-age', default=False, action='store_true',
                help='Train model with chronological age multiplier')
parser.add_argument('--gender-multiplier', default=False, action='store_true',
                help='Train model with gender multiplier')
parser.add_argument('--use-gut-microbiome', default=False, action='store_true',
                help='Train model with gut microbiome')
parser.add_argument('--use-pe-performance', default=False, action='store_true',
                help='Train model with pe performance')

# Dataloading-related settings
parser.add_argument('--cropped', default=False, action='store_true',
                help='Train model with cropped images according to bbox')
parser.add_argument('--dataset', default='RSNA', type=str,choices=['RSNA','RHPE', 'KG'],
                help='Dataset to perform training')
parser.add_argument('--data-train', default='data/train/images', type=str,
                help='path to train data folder')
parser.add_argument('--heatmaps-train', default='data/train/heatmaps', type=str,
                help='path to train heatmaps data folder')
parser.add_argument('--ann-path-train', default='train.csv', type=str,
                help='path to BAA annotations file')
parser.add_argument('--rois-path-train', default='train.json',
                type=str, help='path to ROIs annotations in coco format')
parser.add_argument('--data-val', default='data/val/images', type=str,
                help='path to val data folder')
parser.add_argument('--heatmaps-val', default='data/val/heatmaps', type=str,
                help='path to train heatmaps data folder')
parser.add_argument('--ann-path-val', default='val.csv', type=str,
                help='path to BAA annotations file')
parser.add_argument('--rois-path-val', default='val.json',
                type=str, help='path to ROIs annotations in coco format')

parser.add_argument('--trainval', default=False, action='store_true',
                help='Train model with train and validation sets')
parser.add_argument('--save-folder', default='TRAIN/new_test/',
                help='location to save checkpoint models')
parser.add_argument('--snapshot', default='boneage_bonet_weights.pth',
                help='path to weight snapshot file')
parser.add_argument('--optim-snapshot', type=str,
                default='boneage_bonet_optim.pth',
                help='path to optimizer state snapshot')
parser.add_argument('--eval-first', default=False, action='store_true',
                help='evaluate model weights before training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                help='number of data loading workers (default: 4)')

# Training procedure settings
parser.add_argument('--batch-size', default=1, type=int,
                help='Batch size for training')
parser.add_argument('--epochs', type=int, default=20,
                help='upper epoch limit')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                help='initial learning rate')
parser.add_argument('--patience', default=2, type=int,
                help='patience epochs for LR decreasing')
parser.add_argument('--start-epoch', type=int, default=1,
                help='epoch number to resume')
parser.add_argument('--seed', type=int, default=1111,
                help='random seed')
parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                help='report interval')
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()
args.rank = 0
args_dict = vars(args)

print('Argument list to program')
print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                 for arg in args_dict]))
print('\n\n')

torch.manual_seed(args.seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Create the network architecture and load the best model
net = SIMBA(
    chronological_age=args.chronological_age,
    gender_multiplier=args.gender_multiplier,
    use_gut_microbiome=args.use_gut_microbiome,
    use_pe_performance=args.use_pe_performance
)

if args.rank == 0:
    print('---> Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))

model_to_load = args.snapshot
if osp.exists(model_to_load) and args.rank == 0:
    print('Loading state dict from: {0}'.format(model_to_load))
    snapshot_dict = torch.load(model_to_load, map_location=lambda storage, loc: storage) 
    weights = net.state_dict()
    new_snapshot_dict = snapshot_dict.copy()
    for key in snapshot_dict:
        if key not in weights.keys():
            new_key = 'inception_v3.' + key
            if new_key in snapshot_dict and weights[new_key].shape == snapshot_dict[key].shape:
                new_snapshot_dict[new_key] = snapshot_dict[key]
            new_snapshot_dict.pop(key)
        elif weights[key].shape != snapshot_dict[key].shape:
            new_snapshot_dict.pop(key)

    weights.update(new_snapshot_dict)
    net.load_state_dict(weights)

# TensorBoard 初始化
writer = SummaryWriter(log_dir=args.save_folder + '/tensorboard_logs')

net = net.to(device)

# Criterion
criterion = nn.L1Loss()

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=args.lr)
annealing = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.8, patience=args.patience, cooldown=5,
    min_lr=0.00001, eps=0.0001, verbose=True)
optim_to_load = args.optim_snapshot
if osp.exists(optim_to_load) and args.rank == 0:
    print('loading optim snapshot from {}'.format(optim_to_load))
    optimizer.load_state_dict(torch.load(optim_to_load, map_location=lambda storage,
                                             loc: storage))
    
# Dataloaders
train_transform = transforms.Compose([transforms.Resize((500, 500)),
                               transforms.RandomAffine(
                                   20, translate=(0.2, 0.2),
                                   scale=(1, 1.2)
                                ),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor()]
                            )

val_transform = transforms.Compose([transforms.Resize((500, 500)),
                               transforms.ToTensor()]
                            )

if args.trainval:
    train_dataset = Dataset([args.data_train,args.data_val],
                                [args.heatmaps_train, args.heatmaps_val],
                                [args.ann_path_train,args.ann_path_val],
                                [args.rois_path_train,args.rois_path_val],
                                img_transform=train_transform, crop=args.cropped,
                                dataset=args.dataset
                            )
else:
    train_dataset = Dataset([args.data_train], [args.heatmaps_train],
                                [args.ann_path_train],[args.rois_path_train],
                                img_transform=train_transform, crop=args.cropped,
                                dataset=args.dataset
                            )

val_dataset = Dataset([args.data_val], [args.heatmaps_val],
                        [args.ann_path_val], [args.rois_path_val],
                        img_transform=val_transform, crop=args.cropped,
                        dataset=args.dataset
                    )

train_sampler = None
val_sampler = None

train_loader = DataLoader(train_dataset,
                            shuffle=(train_sampler is None),
                            sampler=train_sampler,
                            batch_size=args.batch_size,
                            num_workers=args.workers
                        )

val_loader = DataLoader(val_dataset,
                            shuffle=(val_sampler is None),
                            sampler=val_sampler,
                            batch_size=1,
                            num_workers=args.workers
                        )

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

from tqdm import tqdm  # 确保 tqdm 被正确导入

def train(epoch, relative_age=True):
    net.train()
    total_loss = AverageMeter()
    epoch_loss_stats = AverageMeter()
    time_stats = AverageMeter()
    loss = 0
    optimizer.zero_grad()
    for (batch_idx, (imgs, bone_ages, genders, chronological_ages, _, gut, pe)) in enumerate(train_loader):
        imgs = imgs.to(device)
        bone_ages = bone_ages.to(device)
        genders = genders.to(device)
        gut = gut.to(device)
        pe = pe.to(device)
        chronological_ages = chronological_ages.to(device)
        if relative_age:
            relative_ages = chronological_ages.squeeze(1) - bone_ages

        start_time = time.time()
        outputs = net(imgs, genders, chronological_ages, gut, pe)
        if relative_age:
            loss = criterion(outputs.squeeze(), relative_ages)
        else:
            loss = criterion(outputs.squeeze(), bone_ages)
        loss.backward()
        optimizer.step()

        loss = metric_average(loss.item(), 'loss')

        time_stats.update(time.time() - start_time, 1)
        total_loss.update(loss, 1)
        
        # 记录 TensorBoard
        writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + batch_idx)
        
        epoch_loss_stats.update(loss, 1)
        optimizer.zero_grad()

        if (batch_idx % args.log_interval == 0) and args.rank == 0:
            elapsed_time = time_stats.avg
            print(' [{:5d}] ({:5d}/{:5d}) | ms/batch {:.4f} |'
                  ' loss {:.6f} | avg loss {:.6f} | lr {:.7f}'.format(
                      epoch, batch_idx, len(train_loader),
                      elapsed_time * 1000, total_loss.avg,
                      epoch_loss_stats.avg,
                      optimizer.param_groups[0]['lr']))
            total_loss.reset()

    epoch_total_loss = epoch_loss_stats.avg
    args.resume_iter = 0

    if args.rank == 0:
        filename = 'boneage_bonet_snapshot.pth'
        filename = osp.join(args.save_folder, filename)
        torch.save(net.state_dict(), filename)

        optim_filename = 'boneage_bonet_optim.pth'
        optim_filename = osp.join(args.save_folder, optim_filename)
        torch.save(optimizer.state_dict(), optim_filename)

    return epoch_total_loss

def evaluate(relative_age=True):
    net.eval()
    epoch_total_loss = AverageMeter()
    absolute_loss_meter = AverageMeter()
    for (batch_idx, (imgs, bone_ages, genders, chronological_ages, _, gut, pe)) in enumerate(val_loader):
        imgs = imgs.to(device)
        bone_ages = bone_ages.to(device)
        genders = genders.to(device)
        gut = gut.to(device)
        pe = pe.to(device)
        chronological_ages = chronological_ages.to(device)
        if relative_age:
            relative_ages = chronological_ages.squeeze(1) - bone_ages

        with torch.no_grad():
            outputs = net(imgs, genders, chronological_ages, gut, pe)
        if relative_age:
            loss = criterion(outputs.squeeze(), relative_ages)
            predicted_bone_age = chronological_ages.squeeze(1) - outputs.squeeze()
        else:
            loss = criterion(outputs.squeeze(), bone_ages)
            predicted_bone_age = outputs.squeeze()
        absolute_loss = criterion(predicted_bone_age, bone_ages)
        absolute_loss = metric_average(absolute_loss.item(), 'absolute_loss')
        loss = metric_average(loss.item(), 'loss')   
        epoch_total_loss.update(loss, 1)
        absolute_loss_meter.update(absolute_loss, 1)
        
        # 记录 TensorBoard
        writer.add_scalar('Loss_relative/val', loss, batch_idx)
        writer.add_scalar('Loss_absolute/val', absolute_loss, batch_idx)

    epoch_total_loss = epoch_total_loss.avg
    absolute_loss_meter = absolute_loss_meter.avg

    if args.rank == 0:
        print('Relative val loss: {:.5f}'.format(epoch_total_loss))
        print('Absolute val loss: {:.5f}'.format(absolute_loss_meter))

    return epoch_total_loss
        
def main():
    if args.rank == 0:
        print('Train begins...')
    best_val_loss = None
    # Find best model in validation
    if osp.exists(osp.join(args.save_folder, 'train.csv')):
        with open(osp.join(args.save_folder, 'train.csv')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            val_list = []
            for row in csv_reader:
                val_list.append(float(row[2]))
            best_val_loss = min(val_list)
    if args.eval_first:
        val_loss = evaluate(relative_age=args.relative_age)
    try:
        out_file = open(os.path.join(args.save_folder, 'train.csv'), 'a+')
        
        for epoch in range(args.start_epoch, args.epochs + 1):
            if args.rank == 0:
                epoch_start_time = time.time()
            train_loss = train(epoch, relative_age=args.relative_age)
            annealing.step(train_loss)
            val_loss = evaluate(relative_age=args.relative_age)
            if args.rank == 0:
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s '
                      '| epoch loss {:.6f} |'.format(
                          epoch, time.time() - epoch_start_time, train_loss))
                print('-' * 89)
                out_file.write('{}, {}, {}\n'.format(epoch, train_loss, val_loss))
                out_file.flush()

                if best_val_loss is None or val_loss < best_val_loss and args.rank == 0:
                    best_val_loss = val_loss
                    filename = osp.join(args.save_folder, 'boneage_bonet_weights.pth')
                    torch.save(net.state_dict(), filename)
        out_file.close()
        writer.close()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

if __name__ == '__main__':
    main()
