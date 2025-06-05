import numpy as np
import random
import torch
import cv2
from PIL import Image
from scipy.ndimage import zoom, rotate
from torchvision import transforms
from torch.nn.functional import interpolate


def flip_3d(img, mask):
    """img, mask : 3d numpy array or tensor
    """
    p = np.random.randint(3)
    if p == 0:
        img_f, mask_f = img[::-1, :, :], mask[::-1, :, :]
    elif p == 1:
        img_f, mask_f = img[:, ::-1, :], mask[:, ::-1, :]
    elif p == 2:
        img_f, mask_f = img[:, :, ::-1], mask[:, :, ::-1]
    
    return img_f, mask_f


# def crop_3d(image, mask, mode, ignore_value=255):
#     """对三维图像进行随机裁剪image, mask (numpy.ndarray), size (int): 裁剪后的边长, ignore_value (int): 填充区域的掩码值.
#     """
#     # r, w, h = image.shape
#     if not isinstance(image, torch.Tensor) or not isinstance(image, torch.Tensor):
#         img = torch.tensor(image, dtype=torch.float32)
#         mask = torch.tensor(mask, dtype=torch.float32)
#     u_size = (100, 330, 330)
#     l_size = (119, 270, 316)
#     if mode == 'train_l':
#         size = l_size
#         cropped_img = interpolate(img.unsqueeze(0).unsqueeze(0), size=l_size, mode='trilinear', align_corners=False).squeeze()
#     elif mode == 'train_u':
#         size = u_size
#         cropped_img = interpolate(img.unsqueeze(0).unsqueeze(0), size=u_size, mode='trilinear', align_corners=False).squeeze()
#     # cropped_img = interpolate(img.unsqueeze(0).unsqueeze(0), size=(size,size,size), mode='trilinear', align_corners=False).squeeze()
#     cropped_image = np.array(cropped_img)

#     c_mask = interpolate(mask.unsqueeze(0).unsqueeze(0), size=size, mode='nearest', align_corners=None).squeeze()
#     cropped_mask = np.array(c_mask)

#     return cropped_image, cropped_mask

def crop_3d(img, mask, size, ignore_value=255):
    """对三维图像进行随机裁剪image, mask (numpy.ndarray), ignore_value (int): 填充区域的掩码值.
    """
    h, l, w= img.shape
    # u_size = (100, 330, 330)
    # l_size = (119, 270, 316)
    # if mode == 'train_l':
    #     size = (119, 270, 316)
    # elif mode == 'train_u':
    #     size = (100, 330, 330)
    while mask.shape[0] <= size[0] or mask.shape[1] <= size[1] or mask.shape[2] <= size[2]:
        ph = int(max((size[0] - h) // 2 + 1, 0))
        pl = int(max((size[1] - l) // 2 + 1, 0))
        pw = int(max((size[2] - w) // 2 + 1, 0))
        img = np.pad(img, [(ph, ph), (pl,pl), (pw, pw)],
                    mode='constant', constant_values=0)
        mask = np.pad(mask, [(ph, ph), (pl,pl), (pw, pw)],
                    mode='constant', constant_values=ignore_value)

    h, l, w = img.shape
    h1 = np.random.randint(0, h - size[0])
    l1 = np.random.randint(0, l - size[1])
    w1 = np.random.randint(0, w - size[2])
    
    # 处理溢出值, 处理为size大小
    label = mask[h1:h1 + size[0], l1:l1 + size[1], w1:w1 + size[2]]
    img = img[h1:h1 + size[0], l1:l1 + size[1], w1:w1 + size[2]]

    return img, label
        
        
def resize_3d(img, mask, ratio_range, n_class):
    ratio = random.uniform(ratio_range[0], ratio_range[1])
    img = zoom(img, (ratio , ratio , ratio), mode='nearest')
    mask = zoom(mask, (ratio , ratio , ratio), mode='nearest')
    mask = mask.clip(0, n_class - 1)
    return img, mask


# scipy.ndimage.gaussian_filter
def blur_3d(img):
    sigma = np.random.randint(2, 10)
    if sigma % 2 == 0:
        sigma += 1
    img = cv2.GaussianBlur(img, (sigma, sigma), 0, 0)
    return img


def random_rotate_3d(img, ax_range):
    # angle = np.random.uniform(0, 360.0)
    angle = np.random.uniform(0, 90.0)
    img_rotate = rotate(img, angle, axes=ax_range, reshape=True, order=3, mode='grid-constant', cval=0)
    return img_rotate


def gamma_transform_3d(img, gamma):
    out = np.array(np.power((img/255), gamma)*255, dtype=np.uint8)
    adjusted = np.clip(out, 0, 255)
    return adjusted


# 对于医学图像来说可有可无
def normalize_3d(img, mask=None):
    mask_1 = img != 0
    desired = img[mask_1]
    mean_val, std_val = 0.03775, 0.25098
    img = np.array(img, dtype=np.float32)
    # mean_val, std_val = desired.mean(), desired.std()
    img = (img - mean_val) / std_val
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = torch.from_numpy(np.array(img)).float()

    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


# 颜色增强， img: numpy array
def ColorJitter3d(image, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
    max = image.shape[0]
    # img_3d = np.zeros((1, image.shape[1], image.shape[2]), dtype=np.uint8)
    img_3d_list = []
    color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    # 从l维度开始施加colorjitter
    for i in range(max):
        img = image[i]
        if np.max(img) - np.min(img) != 0:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = Image.fromarray(np.uint8(255*img))
        else:
            img = (img - np.min(img))
            img = Image.fromarray(np.uint8(255*img))
        img = color_jitter(img)
        img = np.array(img)
        img = np.reshape(img, (img.shape[0], img.shape[1]))
        
        # if len(np.unique(img_3d[0])) == 1 and int(np.unique(img_3d[0])) == 0:
        #     img_3d = img
        # else:
        #     img_3d = np.concatenate((img_3d, img), axis=0)
        img_3d_list.append(img)  
        
    img_3d = np.stack(img_3d_list, axis=0)
    img_3d = img_3d.clip(0, 255)
            
    return img_3d

# cutmix_3d
# ratio_min 0.2, 0.3, 0.5
def obtain_cutmix_box_3d(img, size_min=0.01, size_max=0.5,ratio_min = 0.5):
    l, w, h = img.shape
    mask = torch.zeros(l, w, h)
    ratio_max = 1 / ratio_min
    size = np.random.uniform(size_min, size_max) * min(l, w, h) * max(l, w, h)
    
    while True:
        ratio = np.random.uniform(ratio_min, ratio_max)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        cutmix_l = np.random.randint(0, l)
        cutmix_l = np.random.randint(cutmix_l, l)
        x = np.random.randint(0, cutmix_w)
        y = np.random.randint(0, cutmix_h)
        # z = np.random.randint(0, cutmix_l)
        if cutmix_l > 0:
            z = np.random.randint(0, cutmix_l)
            z = np.random.randint(z, cutmix_l)
        else:
            z = 0
        x = np.random.randint(x, cutmix_w)
        y = np.random.randint(y, cutmix_h)
        # z = np.random.randint(z, cutmix_l)

        if z + cutmix_l <= l and x + cutmix_w <= w and y + cutmix_h <= h:
            break
    mask[z:z + cutmix_l, x:x + cutmix_w, y:y + cutmix_h] = 1
    
    return mask









