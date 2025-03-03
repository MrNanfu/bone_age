# -*- coding: utf-8 -*-

"""
Bone Age Assessment SIMBA test routine.
"""

# Standard lib imports
import os
import csv
import glob
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
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Local imports
from models.simba import SIMBA
from data.data_loader import BoneageDataset as Dataset
from utils import AverageMeter
from utils import metric_average

# Other imports
from tqdm import tqdm

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
parser.add_argument('--use-correlation', default=False, action='store_true',
                help='Train model with correlation features')
parser.add_argument('--use-image', default=False, action='store_true',
                help='Train model with image')

parser.add_argument('--feature-extractor', default='resnet', type=str,
                help='imaage feature extraction')

# Dataloading-related settings
parser.add_argument('--cropped', default=False, action='store_true',
                help='Test model with cropped images according to bbox')
parser.add_argument('--dataset', default='RSNA', type=str,choices=['RSNA','RHPE', 'KG'],
                help='Dataset to perform test')

parser.add_argument('--data-test', default='data/test/images', type=str,
                help='path to test data folder')
parser.add_argument('--heatmaps-test', default='data/test/heatmaps', type=str,
                help='path to test heatmaps data folder')
parser.add_argument('--ann-path-test', default='test.csv', type=str,
                help='path to BAA annotations file')
parser.add_argument('--rois-path-test', default='test.json',
                type=str, help='path to ROIs annotations in coco format')

parser.add_argument('--save-folder', default='TRAIN/new_test/',
                help='location to save checkpoint models')
parser.add_argument('--snapshot', default='boneage_bonet_weights.pth',
                help='path to weight snapshot file')
parser.add_argument('--save-file', default='test.csv',
                help='path to predictions file')


parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                help='number of data loading workers (default: 4)')

# Training procedure settings
parser.add_argument('--batch-size', default=1, type=int,
                help='Batch size for training')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--gpu', type=str, default='3')

parser.add_argument('--inference-only', default=False, action='store_true',
                help='Only generate test predictions (use it when you dont have groundtruth)')


args = parser.parse_args()
args.rank = 0

args_dict = vars(args)
print('Argument list to program')
print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                 for arg in args_dict]))
print('\n\n')

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if not os.path.exists(os.path.join(args.save_folder, 'inference')):
    os.makedirs(os.path.join(args.save_folder, 'inference'))


# Create the network architecture and load the best model
net = SIMBA(
    chronological_age=args.chronological_age,
    gender_multiplier=args.gender_multiplier,
    use_gut_microbiome=args.use_gut_microbiome,
    use_pe_performance=args.use_pe_performance,
    use_correlation=args.use_correlation,
    use_image=args.use_image,
    feature_extractor=args.feature_extractor
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
            new_key= 'inception_v3.' + key
            new_snapshot_dict[new_key] = snapshot_dict[key]
            new_snapshot_dict.pop(key)

    net.load_state_dict(new_snapshot_dict)

net = net.to(device)

# Criterion
criterion = nn.L1Loss()


if args.feature_extractor == 'vit':
    # Dataloader
    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor()]
                            )
else:
    # Dataloader
    test_transform = transforms.Compose([transforms.Resize((500, 500)),
                                transforms.ToTensor()]
                            )

test_dataset = Dataset([args.data_test], [args.heatmaps_test],
                        [args.ann_path_test], [args.rois_path_test],
                        img_transform=test_transform, crop=args.cropped,
                        dataset=args.dataset,inference=args.inference_only
                    )

# Data samplers
test_sampler = None

test_loader = DataLoader(test_dataset,
                            shuffle=False, 
                            sampler=test_sampler,
                            batch_size=1,
                            num_workers=args.workers
                        )

def main():
    print('Inference begins...')
    
    # 读取 ID 信息
    carpograms = pd.read_csv(os.path.join('Paths', args.ann_path_test))
    ids = carpograms['ID']
    p_dict = dict.fromkeys(ids)

    # 运行测试并获取预测和真值
    p_dict = test(args, net, test_loader, test_sampler, criterion, p_dict, 
                  relative_age=args.relative_age, inference=args.inference_only)

    # 处理成 DataFrame
    data_list = [{"ID": k, "True Age": v[0], "Predicted Age": v[1]} for k, v in p_dict.items()]
    df = pd.DataFrame(data_list)

    # 保存 CSV
    df.to_csv(os.path.join(args.save_folder, 'inference', args.save_file), index=False)
    print(f"Results saved to {os.path.join(args.save_folder, 'inference', args.save_file)}")


def test(args, net, loader, sampler, criterion, p_dict, relative_age=True, inference=False):
    net.eval()
    for child in net.children():
        if type(child) == nn.BatchNorm2d:
            child.track_running_stats = False
    epoch_loss = AverageMeter()
    epoch_mse = AverageMeter()
    true_values = []
    pred_values = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
            inputs, bone_ages, gender, chronological_age, p_id, gut, pe, cor = batch
            inputs, gender, chronological_age, gut, pe, cor= Variable(inputs).cuda(), Variable(gender).cuda(), Variable(chronological_age).cuda(), Variable(gut).cuda(), Variable(pe).cuda(), Variable(cor).cuda()
            bone_ages = Variable(bone_ages).cuda()
            outputs = net(inputs, gender, chronological_age,  gut, pe, cor)
            if not inference:
                if relative_age:
                    predicted_bone_ages = chronological_age.squeeze(1) - outputs.squeeze()
                else:
                    predicted_bone_ages = outputs.squeeze()  # 如果输出是骨龄，直接使用

                loss = criterion(predicted_bone_ages, bone_ages)
                epoch_loss.update(loss)
                mse = torch.mean((predicted_bone_ages - bone_ages) ** 2)  # MSE 计算
                epoch_mse.update(mse)

                true_values.append(bone_ages.item())  # 记录真实值
                pred_values.append(predicted_bone_ages.item())  # 记录预测值

            # 存储 ID、真实骨龄、预测骨龄
            p_dict[p_id.item()] = (bone_ages.item(), outputs.item())
    if not inference:
        loss = metric_average(epoch_loss.avg, 'loss')
        mse = metric_average(epoch_mse.avg, 'mse')

        # 计算 R²
        true_values_tensor = torch.tensor(true_values)
        pred_values_tensor = torch.tensor(pred_values)
        ss_total = torch.sum((true_values_tensor - true_values_tensor.mean()) ** 2)
        ss_residual = torch.sum((true_values_tensor - pred_values_tensor) ** 2)
        r2 = 1 - ss_residual / ss_total

        if args.rank == 0:
            print('Test loss (MAE): {}'.format(loss))
            print('Test MSE: {}'.format(mse))
            print('Test R²: {:.4f}'.format(r2.item()))
    return p_dict


if __name__ == '__main__':
    main()