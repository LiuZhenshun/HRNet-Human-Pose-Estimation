# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import cv2
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import tools._init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.vis import save_batch_image_with_joints

import dataset
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


args = parse_args()
update_config(cfg, args)

# Data loading code
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
    cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
    transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
)

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
for i, (input, target, target_weight, meta) in enumerate(train_loader):
    save_batch_image_with_joints(input, meta['joints'], meta['joints_vis'], "/media/hkuit164/Backup/ThermalProject/HRNet-Human-Pose-Estimation/output/ThermalCocoFormat/pose_hrnet/w32_256x192_adam_lr1e-3_rgbChannel/train_0_hm_gt1.jpg")

for i in range(len(train_dataset)):
    input, target, target_weight, meta = train_dataset[i]

    image = input.numpy().transpose(1,2,0).copy()
    joints = meta["joints"]
    joints_vis = meta["joints_vis"]
    target_heatmap = target.numpy()
    for index in range(17):
        # cv2.line(image, (int(annos[i]['keypoints'][(point[0]-1)*3]), int(annos[i]['keypoints'][(point[0]-1)*3+1])),
        #          (int(annos[i]['keypoints'][(point[1]-1)*3+1]), int(annos[i]['keypoints'][(point[1]-1)*3+1])), (255, 0, 0), 3)
        if joints_vis[index][0]:
            point = (int(joints[index,0]), int(joints[index,1]))
            image = cv2.circle(image, point, 1, (255, 0, 0), 3)
    cv2.imshow('demon', image)
    heatmap0 = np.concatenate((target_heatmap[0],target_heatmap[1],target_heatmap[2],target_heatmap[3],target_heatmap[4], target_heatmap[5]),axis=1)
    heatmap1 = np.concatenate((target_heatmap[6],target_heatmap[7],target_heatmap[8],target_heatmap[9],target_heatmap[10], target_heatmap[11]),axis=1)
    heatmap2 = np.concatenate((target_heatmap[12],target_heatmap[13],target_heatmap[14],target_heatmap[15],target_heatmap[16], target_heatmap[0]),axis=1)
    heatmap = np.concatenate((heatmap0,heatmap1,heatmap2),axis=0)
    cv2.imshow('heatmap',heatmap)
    cv2.waitKey(0)

