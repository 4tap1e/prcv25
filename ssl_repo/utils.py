import os
import torch, math, h5py
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import measure
from medpy import metric
from tqdm import tqdm
import torch.nn.functional as F
import scipy.ndimage as nd
import torch.distributed as dist

from utils.util import AverageMeter, intersectionAndUnion_3d

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def get_cityscapes_labels():
    return np.array([
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    # for key, val in param.items():
    #     log_file.write(key + ':' + str(val) + '\n')
    log_file.write(str(param))
    log_file.close()

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def get_iou(pred, gt, n_classes=21):
    total_iou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou

    return total_iou

def get_dice(pred, gt):
    total_dice = 0.0
    pred = pred.long()
    gt = gt.long()
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]
        dice = 2.0*torch.sum(pred_tmp*gt_tmp).item()/(1.0+torch.sum(pred_tmp**2)+torch.sum(gt_tmp**2)).item()
        print(dice)
        total_dice += dice

    return total_dice

def get_mc_dice(pred, gt, num=2):
    # num is the total number of classes, include the background
    total_dice = np.zeros(num-1)
    pred = pred.long()
    gt = gt.long()
    for i in range(len(pred)):
        for j in range(1, num):
            pred_tmp = (pred[i]==j)
            gt_tmp = (gt[i]==j)
            dice = 2.0*torch.sum(pred_tmp*gt_tmp).item()/(1.0+torch.sum(pred_tmp**2)+torch.sum(gt_tmp**2)).item()
            total_dice[j-1] +=dice
    return total_dice

def post_processing(prediction):
    prediction = nd.binary_fill_holes(prediction)
    label_cc, num_cc = measure.label(prediction,return_num=True)
    total_cc = np.sum(prediction)
    measure.regionprops(label_cc)
    for cc in range(1,num_cc+1):
        single_cc = (label_cc==cc)
        single_vol = np.sum(single_cc)
        if single_vol/total_cc<0.2:
            prediction[single_cc]=0

    return prediction


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/(np.expand_dims(cnt, axis=0) + 1e-7)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)


def test_all_case(net, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, num_gpus=1):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/{}/2022.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 2))
    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        for i in range(1, num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)
    print("Validation end")
    total_metric = torch.tensor(total_metric)
    dist.all_reduce(total_metric)
    total_metric = total_metric / num_gpus
    return total_metric / len(image_list)


def cal_hd(a, b):
    a, b = sitk.Cast(sitk.RescaleIntensity(a), sitk.sitkUInt8), sitk.Cast(sitk.RescaleIntensity(b), sitk.sitkUInt8)
    filter1 = sitk.HausdorffDistanceImageFilter()
    filter1.Execute(a, b)
    hd = filter1.GetHausdorffDistance()
    return hd


def evaluate_3d(model, mode, args):
    model.eval()
    base_dir = args.root_path
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    pred_sum_meter = AverageMeter()
    HD95 = AverageMeter()
    hd95 = 95.0
    
    with open(base_dir + '/{}.txt'.format(mode), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/{}/2022.h5".format(item.replace('\n', '').split(",")[0]) for item in image_list]
    with torch.no_grad():
        for image_path in tqdm(image_list):
            h5f = h5py.File(image_path, 'r')
            img = h5f['image'][:]
            mask = h5f['label'][:]
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask)
            img = img.cuda()
            ori_h, ori_w, ori_l = img.shape[-3:]
            new_h, new_w, new_l = 64, 128, 128
            img = img.unsqueeze(0).unsqueeze(0)
            img = F.interpolate(img, (new_h, new_w, new_l), mode='trilinear', align_corners=True)   
            pred = model(img)
            pred = F.interpolate(pred, (ori_h, ori_w, ori_l), mode='trilinear', align_corners=True)
            pred = pred.argmax(dim=1)
            pred = pred.squeeze(0)
            
            # # cal HD95
            # pred_npy = pred.cpu().numpy()
            # mask_npy = mask.cpu().numpy()
            # pred_npy = np.transpose(pred_npy, (1, 2, 0))
            # mask_npy = np.transpose(mask_npy, (1, 2, 0))
            # pred_sitk = sitk.GetImageFromArray(pred_npy)
            # mask_sitk = sitk.GetImageFromArray(mask_npy)
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
                intersectionAndUnion_3d(pred.cpu().numpy(), mask.numpy(), args.num_classes, 255)
            
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

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-7) 
    mIOU = np.mean(iou_class)
    dice_numerator = 2 * intersection_meter.sum
    dice_denominator = pred_sum_meter.sum + target_meter.sum
    dice_class = dice_numerator / (dice_denominator + 1e-7) 
    mDICE = np.mean(dice_class)
    hd95 = HD95.avg

    return mDICE, dice_class, hd95


def calculate_metrics(pred_folder, label_folder, args):
    pred_files = [os.path.join(pred_folder, file) for file in os.listdir(pred_folder)]
    label_files = [os.path.join(label_folder, file) for file in os.listdir(label_folder)]

    accuracy_lists = []
    jaccard_scores = []
    num_samples = args.test_num 
    class_accuracy = [0.0] * (args.num_classes - 1)
    dice_lists = {str(i): [] for i in range(args.num_classes - 1)} # 不考虑背景类别

    for pred_path, label_path in zip(pred_files, label_files):
        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        dice_scores = dice_coefficient(pred, label, args.num_classes)
        accuracy_lists.append(dice_scores)
        # for class_id, dice_score in enumerate(dice_scores, start=1):
        for class_id, dice_score in enumerate(dice_scores):
            # if class_id == 1:
            #     print(f">>>>1=>>>>")
            # if class_id == 4:
            #     print(f">>>>4====>")
            # if class_id == 14:
            #     print(f">>>>14====>---")
            # if class_id == 15:
            #     print(f">>>>15====>---!!!!!!!!")
            # print(f"Class{CLASSES[args.dataset][class_id+1]}_{class_id} Dice: {dice_score:.4f}")
            dice_lists[str(class_id)].append(round(dice_score, 4))


        jaccard_scores.append(jaccard_coefficient(pred, label, slice))
    # for i in range(args.num_classes-1):
    #     print(f"Class{CLASSES[args.dataset][i+1]} mDice: {sum(dice_lists[str(i)]) / len(dice_lists[str(i)]):.4f}")
    
    for i in range(4):
        accuracy_lists[-(i+1)][8] = 0
        
    for accuracy_list in accuracy_lists:
        for class_id in range(args.num_classes-1):
            class_accuracy[class_id] += accuracy_list[class_id]

    # 计算每个类别的平均精度
    average_accuracy = [acc / num_samples for acc in class_accuracy]    

    average_accuracy[8] = average_accuracy[8] * args.test_num / 10    # test中有四个案例没有8这个器官，对于flare22数据集而言

    avg_jaccard = np.mean(jaccard_scores)


    # 打印每个类别的平均精度
    # for class_id, avg_acc in enumerate(average_accuracy):
    #     print(f"Class {CLASSES[args.dataset][class_id+1]} Average Accuracy: {avg_acc*100:.2f}")

    # print(f'******************{args.dataset} Average Accuracy: {np.mean(average_accuracy)}')
    # print(f'******************{args.dataset} Average jaccard: {avg_jaccard}')
    return average_accuracy, avg_jaccard
        

def dice_coefficient(prediction, target, class_num = 14):
    dice_coefficient = []

    for i in range(class_num - 1):
        dice_cls = metric.binary.dc(prediction == (i + 1), target == (i + 1))
        dice_coefficient.append(dice_cls)

    return dice_coefficient

def jaccard_coefficient(prediction, target, slice,class_num = 14 ):

    jaccard_coefficient = []
    a = np.unique(target)
    for i in range(class_num - 1):
        try:
            dice_cls = metric.binary.jc(prediction == (i + 1), target == (i + 1))
            jaccard_coefficient.append(dice_cls)
        except ZeroDivisionError:
            pass

    return sum(jaccard_coefficient)/len(jaccard_coefficient)

         
def calculate_metrics_amos(pred_folder, label_folder, args):
    pred_files = [os.path.join(pred_folder, file) for file in os.listdir(pred_folder)]
    label_files = [os.path.join(label_folder, file) for file in os.listdir(label_folder)]
    pred_files = sorted(pred_files)
    label_files = sorted(label_files)

    accuracy_lists = []
    jaccard_scores = []
    num_samples = args.test_num 
    class_accuracy = [0.0] * (args.num_classes - 1)

    dice_lists = {str(i): [] for i in range(args.num_classes - 1)} # 不考虑背景类别

    for pred_path, label_path in zip(pred_files, label_files):
        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        
        dice_scores = dice_coefficient(pred, label, args.num_classes)
        accuracy_lists.append(dice_scores)
        for class_id, dice_score in enumerate(dice_scores):
            # print(f"Class{CLASSES[args.dataset][class_id+1]}_{class_id} Dice: {dice_score:.4f}")
            dice_lists[str(class_id)].append(round(dice_score, 4))

        jaccard_scores.append(jaccard_coefficient(pred, label, slice))
    # for i in range(args.num_classes - 1):
    #     print(f"Class{CLASSES[args.dataset][i+1]} mDice: {sum(dice_lists[str(i)]) / len(dice_lists[str(i)]):.4f}")
    
    # 除去背景后剩下的里面的第4个器官缺失，共计6个数据
    accuracy_lists[6][3] = 0  
    accuracy_lists[15][3] = 0 
    accuracy_lists[26][3] = 0    
    accuracy_lists[31][3] = 0 
    accuracy_lists[45][3] = 0
    accuracy_lists[59][3] = 0  

    # 除去背景后剩下的里面的第1个器官缺失
    accuracy_lists[13][0] = 0
    
    # 除去背景后剩下的里面的第14个器官缺失
    accuracy_lists[29][13] = 0 

    # 除去背景后剩下的里面的第15个器官缺失
    accuracy_lists[29][14] = 0 

    for accuracy_list in accuracy_lists:
        for class_id in range(args.num_classes-1):
            class_accuracy[class_id] += accuracy_list[class_id]

    # 计算每个类别的平均精度
    average_accuracy = [acc / num_samples for acc in class_accuracy]    
    
    # test中对应的案例中没有这些器官
    average_accuracy[3] = average_accuracy[3] * 60 / 54
    average_accuracy[0] = average_accuracy[0] * 60 / 59   # class 1 have 59 cases
    average_accuracy[13] = average_accuracy[13] * 60 / 59
    average_accuracy[14] = average_accuracy[14] * 60 / 59

    avg_jaccard = np.mean(jaccard_scores)

    # 打印每个类别的平均精度
    # for class_id, avg_acc in enumerate(average_accuracy):
    #     print(f"Class {CLASSES[args.dataset][class_id+1]} Average Accuracy: {avg_acc*100:.2f}")

    # print(f'******************{args.dataset} Average Accuracy: {np.mean(average_accuracy)}')
    # print(f'******************{args.dataset} Average jaccard: {avg_jaccard}')
    return average_accuracy, avg_jaccard