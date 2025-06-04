import argparse
import logging
import glob
import os
import pprint

import torch
import numpy as np
from torch import nn
import SimpleITK as sitk
from PIL import Image
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import Flare2Dataset
from model import unet
from utils.classes import CLASSES
from utils.ohem import ProbOhemCrossEntropy2d
from utils.util import count_params, AverageMeter, intersectionAndUnion, init_log, intersectionAndUnion_3d
from utils.dist_helper import setup_distributed

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = argparse.ArgumentParser(description='Fully-Supervised Training in Semantic Segmentation')
parser.add_argument('--config', type=str, default='./configs/fetal25.yaml')
parser.add_argument('--save_path', type=str, default='./log')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

def convert_image(img):
    # Open the image and convert to grayscale
    width, height = img.size

    # Create a new image in RGB mode
    new_img = Image.new("RGB", (width, height))
    pixels = img.load()
    new_pixels = new_img.load()

    for x in range(width):
        for y in range(height):
            intensity = pixels[x, y]
            if intensity == 0:
                new_pixels[x, y] = (255, 255, 255)    # Red
            elif intensity == 1:
                new_pixels[x, y] = (0, 0, 0)      # Black
            elif intensity == 2:
                new_pixels[x, y] = (125, 125, 125)    # Green
            else:
                new_pixels[x, y] = (intensity, intensity, intensity)

    return new_img


def evaluate(model, loader, mode, cfg, multiplier=None, epoch=None, is_best=False):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    pred_sum_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:
            
            img = img.cuda()
                
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, cfg['nclass'], h, w).cuda()      # 3 is number of classes
                
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: row + grid, col: col + grid])
                        final[:, :, row: row + grid, col: col + grid] += pred.softmax(dim=1)
                        if col == w - grid:
                            break
                        col = min(col + int(grid * 2 / 3), w - grid)
                    if row == h - grid:
                        break
                    row = min(row + int(grid * 2 / 3), h - grid)
                    
                pred = final
            
            else:
                assert mode == 'original'
                
                if multiplier is not None:
                    ori_h, ori_w = img.shape[-2:]
                    if multiplier == 512:
                        new_h, new_w = 512, 512
                    else:
                        new_h, new_w = int(ori_h / multiplier + 0.5) * multiplier, int(ori_w / multiplier + 0.5) * multiplier
                    img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)
                
                pred = model(img)
            
                if multiplier is not None:
                    pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)
                
            pred = pred.argmax(dim=1)

            # pred_tensor = pred
            # pred_npy = pred_tensor.cpu().numpy()
            # mask_npy = mask.cpu().numpy()
            # pred_npy = np.transpose(pred_npy, (1, 2, 0))
            # mask_npy = np.transpose(mask_npy, (1, 2, 0))
           

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            
            pred_sum = union + intersection - target

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()
            reduced_pred_sum = torch.from_numpy(pred_sum).cuda()
            
            # 并行验证
            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)
            dist.all_reduce(reduced_pred_sum)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            target_meter.update(reduced_target.cpu().numpy())
            pred_sum_meter.update(reduced_pred_sum.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) 
    mIOU = np.mean(iou_class)

    dice_numerator = 2 * intersection_meter.sum
    dice_denominator = pred_sum_meter.sum + target_meter.sum
    dice_class = dice_numerator / (dice_denominator + 1e-10) 
    mDICE = np.mean(dice_class)

    return mIOU, iou_class, mDICE, dice_class


def evaluate_for_train(model, loader, mode, cfg, multiplier=None, epoch=None, is_best=False):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    pred_sum_meter = AverageMeter()

    with torch.no_grad():
        for img, mask in loader:
            
            img = img.cuda()
                
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, cfg['nclass'], h, w).cuda()      # 3 is number of classes
                
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: row + grid, col: col + grid])
                        final[:, :, row: row + grid, col: col + grid] += pred.softmax(dim=1)
                        if col == w - grid:
                            break
                        col = min(col + int(grid * 2 / 3), w - grid)
                    if row == h - grid:
                        break
                    row = min(row + int(grid * 2 / 3), h - grid)
                    
                pred = final
            
            else:
                assert mode == 'original'
                
                if multiplier is not None:
                    ori_h, ori_w = img.shape[-2:]
                    if multiplier == 512:
                        new_h, new_w = 512, 512
                    else:
                        new_h, new_w = int(ori_h / multiplier + 0.5) * multiplier, int(ori_w / multiplier + 0.5) * multiplier
                    img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)
                
                pred = model(img)
            
                if multiplier is not None:
                    pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)
                
            pred = pred.argmax(dim=1)

            # pred_tensor = pred
            # pred_npy = pred_tensor.cpu().numpy()
            # mask_npy = mask.cpu().numpy()
            # pred_npy = np.transpose(pred_npy, (1, 2, 0))
            # mask_npy = np.transpose(mask_npy, (1, 2, 0))
           

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            
            pred_sum = union + intersection - target

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()
            reduced_pred_sum = torch.from_numpy(pred_sum).cuda()
            
            # 并行验证
            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)
            dist.all_reduce(reduced_pred_sum)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            target_meter.update(reduced_target.cpu().numpy())
            pred_sum_meter.update(reduced_pred_sum.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) 
    mIOU = np.mean(iou_class)

    dice_numerator = 2 * intersection_meter.sum
    dice_denominator = pred_sum_meter.sum + target_meter.sum
    dice_class = dice_numerator / (dice_denominator + 1e-10) 
    mDICE = np.mean(dice_class)

    return mIOU, iou_class, mDICE, dice_class


def evaluate_3d(model, loader, mode, cfg, multiplier=None, writer=None, epoch=None, is_best=False):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    pred_sum_meter = AverageMeter()
    HD95 = AverageMeter()
    hd95 = 95.0

    with torch.no_grad():
        for img, mask, id in loader:
            
            img = img.cuda()
                
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 3, h, w).cuda()      # 3 is number of classes
                
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: row + grid, col: col + grid])
                        final[:, :, row: row + grid, col: col + grid] += pred.softmax(dim=1)
                        if col == w - grid:
                            break
                        col = min(col + int(grid * 2 / 3), w - grid)
                    if row == h - grid:
                        break
                    row = min(row + int(grid * 2 / 3), h - grid)
                    
                pred = final
            
            else:
                assert mode == 'original'
                
                if multiplier is not None:
                    ori_h, ori_w, ori_l = img.shape[-3:]
                    # if multiplier == 512:
                    #     new_h, new_w = 512, 512
                    # else:
                    #     new_h, new_w = int(ori_h / multiplier + 0.5) * multiplier, int(ori_w / multiplier + 0.5) * multiplier
                    new_h, new_w, new_l = 64, 128, 128
                    img = F.interpolate(img, (new_h, new_w, new_l), mode='trilinear', align_corners=True)
                    # img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)
                
                pred = model(img)
            
                if multiplier is not None:
                    pred = F.interpolate(pred, (ori_h, ori_w, ori_l), mode='trilinear', align_corners=True)
                
            pred = pred.argmax(dim=1)
            # pred_tensor = pred
            # pred_npy = pred_tensor.cpu().numpy()

            # pred_npy_new = np.squeeze(pred_npy)
            # prd_img = Image.fromarray(pred_npy_new.astype(np.uint8))
            # save_path = f'./predictions/{id[0].split(" ")}.png'  # Save with the image ID as filename
            # os.makedirs('./predictions', exist_ok=True)          # Create directory if it doesn't exist
            # pred_rgb = convert_image(prd_img)
            # if is_best:
            #     pred_rgb.save(save_path)
            
            # pred = pred.argmax(dim=1)
            # pred_tensor = pred
            # pred_npy = pred_tensor.cpu().numpy()
            # mask_npy = mask.cpu().numpy()
            # pred_npy = np.transpose(pred_npy, (1, 2, 0))
            # mask_npy = np.transpose(mask_npy, (1, 2, 0))
            # pred_sitk = sitk.GetImageFromArray(pred_npy)
            # mask_sitk = sitk.GetImageFromArray(mask_npy)

            # cal HD95
            # hd = dict()
            # # 计算upper指标
            # pred_data_upper = sitk.GetArrayFromImage(pred_sitk)
            # pred_data_upper[pred_data_upper == 2] = 0
            # pred_upper = sitk.GetImageFromArray(pred_data_upper)

            # label_data_upper = sitk.GetArrayFromImage(mask_sitk)
            # label_data_upper[label_data_upper == 2] = 0
            # label_upper = sitk.GetImageFromArray(label_data_upper)
            # hd['hd_upper'] = float(cal_hd(pred_upper, label_upper))
            # # 计算lower指标
            # pred_data_lower = sitk.GetArrayFromImage(pred_sitk)
            # pred_data_lower[pred_data_lower == 1] = 0
            # pred_data_lower[pred_data_lower == 2] = 1
            # pred_lower = sitk.GetImageFromArray(pred_data_lower)

            # label_data_lower = sitk.GetArrayFromImage(mask_sitk)
            # label_data_lower[label_data_lower == 1] = 0
            # label_data_lower[label_data_lower == 2] = 1
            # label_lower = sitk.GetImageFromArray(label_data_lower)
            # hd['hd_lower'] = float(cal_hd(pred_lower, label_lower))
            # # 计算总体指标
            # pred_data_all = sitk.GetArrayFromImage(pred_sitk)
            # pred_data_all[pred_data_all == 2] = 1
            # pred_all = sitk.GetImageFromArray(pred_data_all)

            # label_data_all = sitk.GetArrayFromImage(mask_sitk)
            # label_data_all[label_data_all == 2] = 1
            # label_all = sitk.GetImageFromArray(label_data_all)
            # hd['hd_all'] = float(cal_hd(pred_all, label_all))
            # result = (hd['hd_upper'] + hd['hd_lower'] + hd['hd_all']) / 3.0
            # HD95.update(result)

            intersection, union, target = \
                intersectionAndUnion_3d(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            
            pred_sum = union + intersection - target

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()
            reduced_pred_sum = torch.from_numpy(pred_sum).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target) 
            dist.all_reduce(reduced_pred_sum)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            target_meter.update(reduced_target.cpu().numpy())
            pred_sum_meter.update(reduced_pred_sum.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) 
    mIOU = np.mean(iou_class)

    dice_numerator = 2 * intersection_meter.sum
    dice_denominator = pred_sum_meter.sum + target_meter.sum
    dice_class = dice_numerator / (dice_denominator + 1e-10) 
    mDICE = np.mean(dice_class)
    # hd95 = HD95.avg

    return mIOU, iou_class, mDICE, dice_class


def evaluate_3d_4train(model, loader, mode, cfg, multiplier=None, writer=None, epoch=None, is_best=False):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    pred_sum_meter = AverageMeter()
    HD95 = AverageMeter()
    hd95 = 95.0

    with torch.no_grad():
        for img, mask in loader:
            
            img = img.cuda()
                
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 14, h, w).cuda()      # 3 is number of classes
                
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: row + grid, col: col + grid])
                        final[:, :, row: row + grid, col: col + grid] += pred.softmax(dim=1)
                        if col == w - grid:
                            break
                        col = min(col + int(grid * 2 / 3), w - grid)
                    if row == h - grid:
                        break
                    row = min(row + int(grid * 2 / 3), h - grid)
                    
                pred = final
            
            else:
                assert mode == 'original'
                
                if multiplier is not None:
                    ori_h, ori_w, ori_l = img.shape[-3:]
                    # if multiplier == 512:
                    #     new_h, new_w = 512, 512
                    # else:
                    #     new_h, new_w = int(ori_h / multiplier + 0.5) * multiplier, int(ori_w / multiplier + 0.5) * multiplier
                    new_h, new_w, new_l = 64, 128, 128
                    img = F.interpolate(img, (new_h, new_w, new_l), mode='trilinear', align_corners=True)
                    # img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)
                
                pred = model(img)
            
                if multiplier is not None:
                    pred = F.interpolate(pred, (ori_h, ori_w, ori_l), mode='trilinear', align_corners=True)
                
            pred = pred.argmax(dim=1)
            # pred_tensor = pred
            # pred_npy = pred_tensor.cpu().numpy()

            # pred_npy_new = np.squeeze(pred_npy)
            # prd_img = Image.fromarray(pred_npy_new.astype(np.uint8))
            # save_path = f'./predictions/{id[0].split(" ")}.png'  # Save with the image ID as filename
            # os.makedirs('./predictions', exist_ok=True)          # Create directory if it doesn't exist
            # pred_rgb = convert_image(prd_img)
            # if is_best:
            #     pred_rgb.save(save_path)
            
            # pred = pred.argmax(dim=1)
            # pred_tensor = pred
            # pred_npy = pred_tensor.cpu().numpy()
            # mask_npy = mask.cpu().numpy()
            # pred_npy = np.transpose(pred_npy, (1, 2, 0))
            # mask_npy = np.transpose(mask_npy, (1, 2, 0))
            # pred_sitk = sitk.GetImageFromArray(pred_npy)
            # mask_sitk = sitk.GetImageFromArray(mask_npy)

            # cal HD95
            # hd = dict()
            # # 计算upper指标
            # pred_data_upper = sitk.GetArrayFromImage(pred_sitk)
            # pred_data_upper[pred_data_upper == 2] = 0
            # pred_upper = sitk.GetImageFromArray(pred_data_upper)

            # label_data_upper = sitk.GetArrayFromImage(mask_sitk)
            # label_data_upper[label_data_upper == 2] = 0
            # label_upper = sitk.GetImageFromArray(label_data_upper)
            # hd['hd_upper'] = float(cal_hd(pred_upper, label_upper))
            # # 计算lower指标
            # pred_data_lower = sitk.GetArrayFromImage(pred_sitk)
            # pred_data_lower[pred_data_lower == 1] = 0
            # pred_data_lower[pred_data_lower == 2] = 1
            # pred_lower = sitk.GetImageFromArray(pred_data_lower)

            # label_data_lower = sitk.GetArrayFromImage(mask_sitk)
            # label_data_lower[label_data_lower == 1] = 0
            # label_data_lower[label_data_lower == 2] = 1
            # label_lower = sitk.GetImageFromArray(label_data_lower)
            # hd['hd_lower'] = float(cal_hd(pred_lower, label_lower))
            # # 计算总体指标
            # pred_data_all = sitk.GetArrayFromImage(pred_sitk)
            # pred_data_all[pred_data_all == 2] = 1
            # pred_all = sitk.GetImageFromArray(pred_data_all)

            # label_data_all = sitk.GetArrayFromImage(mask_sitk)
            # label_data_all[label_data_all == 2] = 1
            # label_all = sitk.GetImageFromArray(label_data_all)
            # hd['hd_all'] = float(cal_hd(pred_all, label_all))
            # result = (hd['hd_upper'] + hd['hd_lower'] + hd['hd_all']) / 3.0
            # HD95.update(result)

            intersection, union, target = \
                intersectionAndUnion_3d(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            
            pred_sum = union + intersection - target

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()
            reduced_pred_sum = torch.from_numpy(pred_sum).cuda()

            # dist.all_reduce(reduced_intersection)
            # dist.all_reduce(reduced_union)
            # dist.all_reduce(reduced_target) 
            # dist.all_reduce(reduced_pred_sum)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            target_meter.update(reduced_target.cpu().numpy())
            pred_sum_meter.update(reduced_pred_sum.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) 
    mIOU = np.mean(iou_class)

    dice_numerator = 2 * intersection_meter.sum
    dice_denominator = pred_sum_meter.sum + target_meter.sum
    dice_class = dice_numerator / (dice_denominator + 1e-10) 
    mDICE = np.mean(dice_class)
    # hd95 = HD95.avg

    return mIOU, iou_class, mDICE, dice_class

def cal_hd(a, b):
    a = sitk.Cast(sitk.RescaleIntensity(a), sitk.sitkUInt8)
    b = sitk.Cast(sitk.RescaleIntensity(b), sitk.sitkUInt8)
    filter1 = sitk.HausdorffDistanceImageFilter()
    filter1.Execute(a, b)
    hd = filter1.GetHausdorffDistance()
    return hd


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    save_path = os.path.join(args.save_path, 'epoch{}BS{}spv'.format(cfg['epochs'], cfg['batch_size']))
    cp_path = os.path.join(args.checkpoint_path, 'Epoch{}BS{}spv'.format(cfg['epochs'], cfg['batch_size']))
    os.makedirs(cp_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)


    cfg['batch_size'] *= 2
    
    
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    
    writer = SummaryWriter(save_path)
    
    os.makedirs(save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = unet(3, 14)
    
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict, strict=False)
    
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    
    n_upsampled = {
        'pascal': 3000, 
        'cityscapes': 3000, 
        'ade20k': 6000, 
        'coco': 30000,
        'fetal': 3000
    }
    trainset = Flare2Dataset(
        imgpath=cfg['labeled'], mode='train_l', labelpath=cfg['label_train'], size=cfg['crop_size'], id_path=cfg['labeled_id_path'], nsample=n_upsampled[cfg['dataset']]
    )
    valset = Flare2Dataset(
        imgpath=cfg['val'], mode='val', labelpath=cfg['label_val']
    )
    
    trainloader = DataLoader(
        trainset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True
    )
    
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False
    )
    
    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    # previous_best = 0.0
    pre_best_iou = 0.0
    pre_best_dice= 0.0
    best_epoch = 0
    epoch = -1
    
    # if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
    #     checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     epoch = checkpoint['epoch']
    #     previous_best = checkpoint['previous_best']
        
    #     logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.7f}, Previous best: {:.4f}'.format(
            epoch, optimizer.param_groups[0]['lr'], pre_best_dice))

        model.train()
        total_loss = AverageMeter()

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            pred = model(img)

            loss = criterion(pred, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_x', loss.item(), iters)
            
            if (i % (len(trainloader) // 8) == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))
        
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class, mDICE, dice_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
        
        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'IoU: {:.3f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        for (cls_idx, dice) in enumerate(dice_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'Dice: {:.3f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice))
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.3f}\nMDice: {:.3f}'.format(eval_mode, mIoU, mDICE))
        
        writer.add_scalar('eval/mIoU', mIoU, epoch)
        for i, iou in enumerate(iou_class):
            writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
        writer.add_scalar('eval/mDice', mDICE, epoch)
        for i, dice in enumerate(dice_class):
            writer.add_scalar('eval/%s_Dice' % (CLASSES[cfg['dataset']][i]), dice, epoch)
        
        is_best = mDICE >= pre_best_dice
        # previous_best = max(mIoU, previous_best)
        pre_best_iou = max(mIoU, pre_best_iou)
        pre_best_dice = max(mDICE, pre_best_dice)
        
        if mDICE == pre_best_dice:
            best_epoch = epoch

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_epoch': best_epoch,
            'previous_best': pre_best_dice,
        }
        if is_best:
            torch.save(checkpoint, os.path.join(cp_path, 'best_epoch_{}_bs{}_dice{}_spv.pth'.format(epoch, cfg['batch_size'], mDICE)))


if __name__ == '__main__':
    main()
