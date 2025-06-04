import SimpleITK as sitk
import torch, math, os
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from medpy import metric

from ssl_repo.utils import calculate_metrics, calculate_metrics_amos

def getFiles(targetdir):
    ls = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue
        ls.append(fname)
    return ls


def nii_val_case(rank, model, output, args, num_gpus, pbar=None):
    print("Validation begin")
    total_metric = np.zeros((args.num_classes-1, 2))
    imgdir = args.nii_img_path
    maskdir = args.nii_mask_path
    patch_size = args.val_patch_size
    stride_xy = args.val_xy
    stride_z = args.val_z
    for pdx, fname in enumerate(sorted(getFiles(imgdir))):
        sitk_img = sitk.ReadImage(os.path.join(imgdir, fname))  # img
        image = sitk.GetArrayFromImage(sitk_img)  # zyx

        sitk_mask = sitk.ReadImage(os.path.join(maskdir, fname))
        label = sitk.GetArrayFromImage(sitk_mask)
        
        w, h, d = image.shape
        add_pad = False
        if w < patch_size[0]:
            w_pad = patch_size[0] - w
            add_pad = True
        else:
            w_pad = 0
        if h < patch_size[1]:
            h_pad = patch_size[1] - h
            add_pad = True
        else:
            h_pad = 0
        if d < patch_size[2]:
            d_pad = patch_size[2] - d
            add_pad = True
        else:
            d_pad = 0
        wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
        hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
        dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
        if add_pad:
            image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                        constant_values=0)
        ww, hh, dd = image.shape
        
        sx = math.ceil((ww - patch_size[0]) / stride_z) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_xy) + 1
        score_map = np.zeros((args.num_classes,) + image.shape).astype(np.float32)
        cnt = np.zeros(image.shape).astype(np.float32)
        
        for x in range(0, sx):
            xs = min(stride_z * x, ww - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_xy * z, dd - patch_size[2])
                    test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                    test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(
                        np.float32)  # 添加一个维度 ex:(1,512,512)到(1,1,512,512)
                    test_patch = torch.from_numpy(test_patch).cuda()

                    y1 = model(test_patch)
                    y = F.softmax(y1, dim=1)

                    y = y.cpu().data.numpy()
                    y = y[0, :, :, :, :] 
                    score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                    cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
        score_map = score_map / (np.expand_dims(cnt, axis=0) + 1e-7)
        label_map = np.argmax(score_map, axis=0)
        if add_pad:
            label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
            score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
            
        prediction, score_map = label_map, score_map
        prediction = prediction.astype(np.uint8)
        
        saveprediction = sitk.GetImageFromArray(prediction)
        saveprediction.SetSpacing(sitk_img.GetSpacing())
        saveprediction.SetOrigin(sitk_img.GetOrigin())
        saveprediction.SetDirection(sitk_img.GetDirection())
        sitk.WriteImage(saveprediction, output + fname.split('_')[0] + "_"
                            + fname.split('_')[1] + ".nii.gz")
        if rank == 0:
            if pbar:
                pbar.update(1)
    if args.dataset.split('_')[0] == 'amos':
        average_accuracy, avg_jaccard = calculate_metrics_amos(output, maskdir, args)
    elif args.dataset.split('_')[0] == 'flare22':
        average_accuracy, avg_jaccard = calculate_metrics(output, maskdir, args)
    else:
        print("dataset not supported")
    
    for i in range(len(average_accuracy)):
        total_metric[i, 0] = average_accuracy[i]
        total_metric[i, 1] = avg_jaccard

    total_metric = torch.tensor(total_metric).to(rank)
    dist.all_reduce(total_metric)
    total_metric = total_metric / num_gpus
    return total_metric


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


# def calculate_metrics_amos_global(pred_folder, label_folder, args):
#     pred_files = [os.path.join(pred_folder, file) for file in os.listdir(pred_folder)]
#     label_files = [os.path.join(label_folder, file) for file in os.listdir(label_folder)]
#     pred_files = sorted(pred_files)
#     label_files = sorted(label_files)

#     # 这里的 accuracy_lists 实际上是 per-image Dice scores
#     accuracy_lists = [] # 存储每张图的各个类别的Dice scores
#     jaccard_scores_overall_per_image = [] # 存储每张图的整体 Jaccard score

#     num_total_validation_images = len(pred_files) # 这是总的验证图像数量

#     for pred_path, label_path in zip(pred_files, label_files):
#         pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
#         label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        
#         dice_scores_per_class = dice_coefficient(pred, label, args.num_classes)
#         accuracy_lists.append(dice_scores_per_class) # 存储这张图所有前景类别的Dice
        
#         # 你的 jaccard_coefficient 函数需要接受单张图的预测和标签
#         # 这里假设它返回的是单张图的整体Jaccard，而不是按类别
#         jaccard_scores_overall_per_image.append(jaccard_coefficient(pred, label, None)) # slice_arg 应该被移除或置为 None

#     # --- 这里是你的硬编码修正逻辑 ---
#     # 这部分逻辑需要非常小心，它依赖于固定的文件顺序和已知的数据问题。
#     # 确保这些索引 (6, 15, 26, 31, 45, 59 等) 对应的是 `pred_files` 和 `label_files` 中
#     # `sorted()` 后的文件索引。
#     # 如果文件列表的顺序发生变化，这些修正会出错。
#     # 强烈建议将这种数据修正逻辑放到数据加载或预处理阶段，而不是指标计算阶段。
    
#     # 除去背景后剩下的里面的第4个器官缺失，共计6个数据 (索引从0开始)
#     if num_total_validation_images > 6: # 确保索引存在
#         accuracy_lists[6][3] = 0  
#         accuracy_lists[15][3] = 0 
#         accuracy_lists[26][3] = 0    
#         accuracy_lists[31][3] = 0 
#         accuracy_lists[45][3] = 0
#         accuracy_lists[59][3] = 0 

#     # 除去背景后剩下的里面的第1个器官缺失
#     if num_total_validation_images > 13:
#         accuracy_lists[13][0] = 0
    
#     # 除去背景后剩下的里面的第14个器官缺失
#     if num_total_validation_images > 29:
#         accuracy_lists[29][13] = 0 

#     # 除去背景后剩下的里面的第15个器官缺失
#     if num_total_validation_images > 29:
#         accuracy_lists[29][14] = 0 
#     # --- 硬编码修正结束 ---

#     # 计算每个类别的平均 Dice (你称为 accuracy_lists)
#     # accuracy_lists 是一个列表的列表，每个子列表是单张图的Dice scores
#     # 我们需要按类别聚合
#     class_wise_dice_sum = np.zeros(args.num_classes - 1, dtype=np.float32)
#     for dice_scores_per_image in accuracy_lists:
#         class_wise_dice_sum += np.array(dice_scores_per_image)

#     average_accuracy_per_class = class_wise_dice_sum / num_total_validation_images # 这里是每个类别的平均Dice

#     # 你原始的 average_accuracy 修正逻辑
#     # 同样需要检查这些索引是否正确且稳定
#     if num_total_validation_images > 0: # 避免除以零
#         average_accuracy_per_class[3] = average_accuracy_per_class[3] * 60 / 54
#         average_accuracy_per_class[0] = average_accuracy_per_class[0] * 60 / 59  # class 1 have 59 cases
#         average_accuracy_per_class[13] = average_accuracy_per_class[13] * 60 / 59
#         average_accuracy_per_class[14] = average_accuracy_per_class[14] * 60 / 59
    
#     avg_jaccard_overall = np.mean(jaccard_scores_overall_per_image)

#     return average_accuracy_per_class, avg_jaccard_overall # 返回每个类别的平均Dice，和整体平均Jaccard


# def nii_val_case_ddp(rank, model, output_dir, args, num_gpus, pbar=None):
#     print(f"Rank {rank}: Validation begin (each GPU processes all files)")
#     model.eval() # Set model to evaluation mode

#     imgdir = args.nii_img_path
#     maskdir = args.nii_mask_path
#     patch_size = args.val_patch_size
#     stride_xy = args.val_xy
#     stride_z = args.val_z

#     # 1. Get ALL NIfTI filenames for EVERY GPU to process
#     all_fnames = getFiles(imgdir)
    
#     # === No DistributedSampler here! Each GPU will iterate through `all_fnames` ===

#     # 2. Initialize local accumulation metrics for this GPU
#     # These will be NumPy arrays initially, then potentially converted for logging
#     # Note: args.num_classes - 1 for foreground classes
#     local_sum_jaccard = np.zeros(args.num_classes - 1, dtype=np.float32)
#     local_sum_accuracy = np.zeros(args.num_classes - 1, dtype=np.float32)
#     local_num_images_processed = 0 # Python int for count

#     # Loop through ALL files for THIS GPU
#     with torch.no_grad(): # Disable gradient computation for validation
#         # Only rank 0 needs the tqdm progress bar for all files
#         file_iterator = all_fnames
#         for fname in file_iterator: # Each GPU processes ALL files
#             # Load NIfTI image and label
#             sitk_img = sitk.ReadImage(os.path.join(imgdir, fname))
#             image = sitk.GetArrayFromImage(sitk_img).astype(np.float32) # zyx order
#             sitk_mask = sitk.ReadImage(os.path.join(maskdir, fname))
#             label = sitk.GetArrayFromImage(sitk_mask).astype(np.uint8) # Ensure label is uint8

#             # Padding logic (remains the same)
#             w, h, d = image.shape
#             add_pad = False
#             w_pad = max(0, patch_size[0] - w)
#             h_pad = max(0, patch_size[1] - h)
#             d_pad = max(0, patch_size[2] - d)
#             if w_pad > 0 or h_pad > 0 or d_pad > 0:
#                 add_pad = True
            
#             wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
#             hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
#             dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
            
#             if add_pad:
#                 image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
#                                 constant_values=0)
            
#             ww, hh, dd = image.shape
            
#             sx = math.ceil((ww - patch_size[0]) / stride_z) + 1
#             sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
#             sz = math.ceil((dd - patch_size[2]) / stride_xy) + 1
            
#             score_map = np.zeros((args.num_classes,) + image.shape).astype(np.float32)
#             cnt = np.zeros(image.shape).astype(np.float32)
            
#             # Internal sliding window inference
#             for x in range(0, sx):
#                 xs = min(stride_z * x, ww - patch_size[0])
#                 for y in range(0, sy):
#                     ys = min(stride_xy * y, hh - patch_size[1])
#                     for z in range(0, sz):
#                         zs = min(stride_xy * z, dd - patch_size[2])
#                         test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
#                         test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0) # (1, 1, D, H, W)
#                         test_patch = torch.from_numpy(test_patch).to(rank) # Ensure patch is on current GPU

#                         y1 = model(test_patch)
#                         y = F.softmax(y1, dim=1) # softmax across class dimension

#                         # Move result from GPU to CPU and convert to NumPy
#                         y_np = y.cpu().data.numpy()[0, :, :, :, :] # Remove batch dimension

#                         score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += y_np
#                         cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1
            
#             # Compute final prediction map
#             score_map = score_map / (np.expand_dims(cnt, axis=0) + 1e-7)
#             label_map = np.argmax(score_map, axis=0).astype(np.uint8) # Convert to uint8
            
#             # Remove padding (if added)
#             if add_pad:
#                 label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
#                 # score_map is usually not needed after prediction, so we omit slicing it back.
#                 # score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
            
#             # Save prediction result (each GPU saves its own copy of all files if needed)
#             # IMPORTANT: If you save all files from each GPU, you'll have duplicate outputs.
#             # Usually, you only save from rank 0, or each rank saves its *assigned* files.
#             # If you *must* have each GPU save everything, ensure your disk can handle it.
#             # For this scenario, I recommend only rank 0 saves to avoid duplicates.
#             # if rank == 0:
#             #     base_fname = "_".join(fname.split('_')[:2]) + ".nii.gz"
#             #     save_path = os.path.join(output_dir, base_fname)
#             #     saveprediction = sitk.GetImageFromArray(label_map)
#             #     saveprediction.SetSpacing(sitk_img.GetSpacing())
#             #     saveprediction.SetOrigin(sitk_img.GetOrigin())
#             #     saveprediction.SetDirection(sitk_img.GetDirection())
#             #     sitk.WriteImage(saveprediction, save_path)


#             # === Accumulate local metrics for this GPU (for the entire dataset) ===
#             current_accuracy, current_jaccard = calculate_metrics_amos_global(label_map, label, args)
            
#             local_sum_jaccard += current_jaccard
#             local_sum_accuracy += current_accuracy
#             local_num_images_processed += 1
            
#             if rank == 0 and pbar:
#                 pbar.update(1) # Update progress bar for rank 0

#     # === No `dist.all_reduce` is needed for metrics, as each GPU calculated for the full set ===
#     # === Only rank 0 reports the final results ===
#     if rank == 0:
#         print(f"\n--- Global Validation Metrics (Processed {local_num_images_processed} images) ---")
#         # Ensure division by the total number of images processed by THIS GPU
#         global_average_jaccard = local_sum_jaccard / local_num_images_processed
#         global_average_accuracy = local_sum_accuracy / local_num_images_processed

#         for i in range(args.num_classes - 1): # Iterate over foreground classes
#             print(f"  Class {i+1} - Avg Accuracy: {global_average_accuracy[i]:.4f}, Avg Jaccard: {global_average_jaccard[i]:.4f}")
        
#         # Convert final NumPy metrics to Tensor for consistent return type
#         final_metric_tensor = torch.tensor(np.stack([global_average_accuracy, global_average_jaccard], axis=1), device=rank)
#         return final_metric_tensor # Return the final aggregated result from rank 0

#     else: # For ranks other than 0, return placeholder or None
#         return None # Or torch.zeros(...) if you need a tensor returned by all ranks
