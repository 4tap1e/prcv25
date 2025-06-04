import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.classes import CLASSES
from ssl_repo import ramps, losses
from ssl_repo.brats2019 import (flare22, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from ssl_repo.utils import test_all_case, evaluate_3d
from model.unet_3d import unet_3D_cps

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,default='./27_FLARE2022', help='Name of Experiment')
parser.add_argument('--exp', type=str,default='', help='Your experiment name')
parser.add_argument('--model', type=str, default='unet_3D', help='model_name')
parser.add_argument('--dataset', type=str, default='flare22_ssl', help='dataset_name')
parser.add_argument('--num_classes', type=int, default=14, help='number of class')
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.1,help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[64, 128, 128],help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=42,help='labeled data')
parser.add_argument('--data_num', type=int, default=420, help='all data')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,default=200.0, help='consistency_rampup')
args = parser.parse_args()

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def train(args, snapshot_path):
    start_time = time.time()
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = args.num_classes

    # net1 = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes).cuda()
    # net2 = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes).cuda()
    net1 = unet_3D_cps(in_chns=1, class_num=num_classes).cuda()
    net2 = unet_3D_cps(in_chns=1, class_num=num_classes).cuda()
    model1 = kaiming_normal_init_weight(net1)
    model2 = xavier_normal_init_weight(net2)
    model1.train()
    model2.train()
    db_train = flare22(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.data_num))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    # optimizer1 = AdamW( 
    #     params=model1.parameters(),
    #     lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01)
    # optimizer2 = AdamW( 
    #     params=model2.parameters(),
    #     lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01)
    best_performance1 = 0.2
    best_performance2 = 0.2
    best_performance_1, best_performance_2 = 0.7, 0.7
    iter_num = 0
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/tensorboardlog')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1   # max_iterations
    epoch = -1
    best_epoch1, best_epoch2 = 0, 0
    is_best1, is_best2 = False, False
    if os.path.exists(os.path.join(snapshot_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(snapshot_path, 'latest.pth'), weights_only=False)
        model1.load_state_dict(checkpoint['model1'])
        model2.load_state_dict(checkpoint['model2'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        epoch = checkpoint['epoch']
        best_performance1= checkpoint['previous_best1']
        best_performance2 = checkpoint['previous_best2']
        best_epoch1 = checkpoint['best_epoch1']
        best_epoch2 = checkpoint['best_epoch2']
        iter_num = checkpoint['iter_num']
        logging.info('***** Load from checkpoint at epoch: {}, iter_num: {} *****'.format(epoch, iter_num))
        
    for epoch in range(epoch + 1, max_epoch):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            # complmentray dropout for cps
            features1 = model1.encoder(volume_batch)
            bs = features1[-1].shape[0]
            dropout_prob = 0.5
            num_kept = int(bs * (1 - dropout_prob))
            binomial = torch.distributions.binomial.Binomial(probs=0.5)
            kept_indexes = torch.randperm(bs)[:num_kept]
            dropoutmask1, dropoutmask2 = [], []
            for j in range(0, len(features1)):
                d =  features1[j].shape[1]
                dropout_mask1 = binomial.sample((bs, d)).cuda() * 2.0
                dropout_mask2 = 2.0 - dropout_mask1
                dropout_mask1[kept_indexes, :] = 1.0
                dropout_mask2[kept_indexes, :] = 1.0
                dropoutmask1.append(dropout_mask1)
                dropoutmask2.append(dropout_mask2)
            outputs1 = model1(volume_batch, comp_drop=dropoutmask1)
            # outputs1 = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch, comp_drop=dropoutmask2)
            # outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs],
                                   label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs],
                                   label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1 = ce_loss(outputs1[args.labeled_bs:], pseudo_outputs2)
            pseudo_supervision2 = ce_loss(outputs2[args.labeled_bs:], pseudo_outputs1)

            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            loss = model1_loss + model2_loss
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group1 in optimizer1.param_groups:
                param_group1['lr'] = lr_
            for param_group2 in optimizer2.param_groups:
                param_group2['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            if iter_num % len(trainloader) == 0:
                logging.info('iteration %d : model1 loss : %f model2 loss : %f consistency_weight: %f' % (iter_num, model1_loss.item(), model2_loss.item(), consistency_weight))
            if iter_num >= 0.6 * args.max_iterations and iter_num % len(trainloader) == 0:
                model1.eval()
                avg_metric1 = test_all_case(
                    model1, args.root_path, test_list="val.txt", num_classes=num_classes, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64)
                if avg_metric1[:, 0].mean() > best_performance_1:
                    best_performance_1 = avg_metric1[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter{}dice{}.pth'.format(
                                                      iter_num, round(best_performance_1, 4)))
                    torch.save(model1.state_dict(), save_mode_path)

                writer.add_scalar('test_model_1/model1_dice_score',
                                  avg_metric1[:, 0].mean(), iter_num)
                for cls_idx1 in range(args.num_classes - 1):
                    logging.info(f'*****>>>>>>test_model_1/model1_dice_{CLASSES[args.dataset][cls_idx1]}{cls_idx1}: {avg_metric1[cls_idx1, 0]}')
                    writer.add_scalar(f'test_model_1/model1_dice{CLASSES[args.dataset][cls_idx1]}', avg_metric1[cls_idx1, 0], iter_num)
                logging.info(
                    '*****iteration %d/%d : model1_mdice: %.5f model1_hd95 : %.5f best_performance_1: %.5f' % (
                        iter_num, args.max_iterations, avg_metric1[:, 0].mean(), avg_metric1[:, 1].mean(), best_performance_1))
                model1.train()

                model2.eval()
                avg_metric2 = test_all_case(
                    model2, args.root_path, test_list="val.txt", num_classes=num_classes, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64)
                if avg_metric2[:, 0].mean() > best_performance_2:
                    best_performance_2 = avg_metric2[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter{}dice{}.pth'.format(
                                                      iter_num, round(best_performance_2, 4)))
                    torch.save(model2.state_dict(), save_mode_path)

                writer.add_scalar('test_model_2/model1_dice_score',
                                  avg_metric2[:, 0].mean(), iter_num)
                for cls_idx2 in range(args.num_classes - 1):
                    logging.info(f'*****>>>>>>test_model_2/model2_dice_{CLASSES[args.dataset][cls_idx2]}{cls_idx2}: {avg_metric2[cls_idx2, 0]}')
                    writer.add_scalar(f'test_model_2/model2_dice{CLASSES[args.dataset][cls_idx2]}', avg_metric2[cls_idx2, 0], iter_num)
                logging.info(
                    '*****iteration %d/%d : model2_mdice : %.5f model2_hd95 : %.5f best_performance_2: %.5f' % (
                        iter_num, args.max_iterations, avg_metric2[:, 0].mean(), avg_metric2[:, 1].mean(), best_performance_2))
                model2.train()

            if iter_num >= max_iterations:
                break
        
        checkpoint = {
            'model1': model1.state_dict(),'model2': model2.state_dict(),
            'optimizer1': optimizer1.state_dict(),'optimizer2': optimizer2.state_dict(),
            'epoch': epoch,
            'previous_best1': best_performance1,'previous_best2': best_performance2,
            'best_epoch1': best_epoch1,'best_epoch2': best_epoch2,
            'iter_num': iter_num}
        torch.save(checkpoint, os.path.join(snapshot_path, 'latest.pth'))
        if iter_num >= max_iterations:
            break
    writer.close()
    end_time = time.time()
    total_time = end_time - start_time  
    logging.info(f'*****Model time *****:{round(total_time, 8)}')

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = f"./{args.exp}_labeled{args.labeled_num}_{args.dataset}_{args.model}_{args.base_lr}"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
