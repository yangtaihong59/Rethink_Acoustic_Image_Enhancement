from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder_param,
                                    paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2
import os
import os.path as osp

def input_mask(image, prob_=0.75, value=0.1):
    """
    Multiplicative bernoulli
    """
    if prob_ > 1:
        prob_ = 1
    try:
        x = image.shape[0]
        y = image.shape[1]

        mask = np.random.choice([0, 1], size=(x, y), p=[prob_, 1 - prob_])
        # mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        noise_image = np.multiply(image, mask)
        noise_image = noise_image - value + value * mask
    except:
        # print(image.shape,mask.shape)
        return image
    return noise_image

def scandir(dir):
    """A simple directory scanner for listing files"""
    return [f for f in os.listdir(dir) if osp.isfile(osp.join(dir, f))]

def paired_mutipaths_from_folder(folders, keys, filename_tmpl, num_pairs=3, stride_range=[1, 5]):
    """Generate paired paths from folders based on closely timestamped files, with adjustable group size and random stride.

    Args:
        folders (list[str]): A list of folder paths, should be [input_folder, gt_folder].
        keys (list[str]): Keys identifying folders, should be ['lq', 'gt'].
        filename_tmpl (str): Template for filename generation, typically for files in the input folder.
        num_pairs (int): Number of image pairs in each group.
        stride_range (tuple): Tuple indicating the minimum and maximum stride possible.

    Returns:
        list[dict]: A list of dictionaries with paired paths.
    """
    assert len(folders) == 2, 'The len of folders should be 2 with [input_folder, gt_folder].'
    assert len(keys) == 2, 'The len of keys should be 2 with [lq_key, gt_key].'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_files = sorted(scandir(input_folder), key=lambda x: int(x.split('_')[0]))
    gt_files = sorted(scandir(gt_folder), key=lambda x: int(x.split('_')[0]))

    assert len(input_files) == len(gt_files), 'Different number of images in datasets.'

    paths = []
    for i in range(0, len(gt_files) - max(stride_range) * (num_pairs - 1)):
        stride = random.randint(*stride_range)  # Random stride for each group
        paired_data = []
        if i + (num_pairs - 1) * stride >= len(gt_files):  # Check if the index goes out of range
            continue
        for j in range(num_pairs):
            index = i + j * stride
            if index >= len(gt_files):  # Additional check to prevent index error
                break
            gt_file = gt_files[index]
            input_file = input_files[index]
            paired_data.append({
                f'{input_key}_path': osp.join(input_folder, input_file),
                f'{gt_key}_path': osp.join(gt_folder, gt_file)
            })
        if len(paired_data) == num_pairs:
            paths.append(paired_data)

    return paths

def add_random_noise(image_stacked, mean=0.3, std=0.7):
    noise = np.random.normal(mean, std, image_stacked.shape)
    noisy_image_stacked = image_stacked + noise
    noisy_image_stacked = np.clip(noisy_image_stacked, 0, 1)
    return noisy_image_stacked.astype(np.float32)
def resize_images(image_list, shape):
    resized_images = []
    for img in image_list:
        # 调整图像大小到标准形状
        # print('resize image to', shape)
        resized_img = np.resize(img, shape)
        resized_images.append(resized_img)
    return resized_images
# 定义一个函数来填充图像
def pad_image(image, target_height, target_width):
    current_height, current_width = image.shape
    top_pad = (target_height - current_height) // 2
    bottom_pad = target_height - current_height - top_pad
    left_pad = (target_width - current_width) // 2
    right_pad = target_width - current_width - left_pad
    padded_image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=0)
    return padded_image
    
class Dataset_PairedMutiImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedMutiImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_mutipaths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            self.filename_tmpl, num_pairs = opt['num_pairs'], stride_range = opt['stride_range'])

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # gt_path = self.paths[index]['gt_path']
        pathes = self.paths[index]
        # print(pathes)
        gt_images = []
        lq_images = []
        for path in pathes:
            gt_path = path['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
                gt_images.append(img_gt)  
                # print(img_gt.shape)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            lq_path = path['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            try:
                img_lq = imfrombytes(img_bytes, flag='grayscale', float32=True)
                lq_images.append(img_lq)
            except:
                raise Exception("lq path {} not working".format(lq_path))
        target_height = max(img.shape[0] for img in gt_images)
        target_width = max(img.shape[1] for img in gt_images)
        # 应用填充
        gt_images_padded = [pad_image(img, target_height, target_width) for img in gt_images]
        lq_images_padded = [pad_image(img, target_height, target_width) for img in lq_images]

        
        # standard_shape = gt_images[0].shape
        # # 调整图像大小
        # gt_images_resized = resize_images(gt_images, standard_shape)
        # lq_images_resized = resize_images(lq_images, standard_shape)

        gt_images_stacked = np.stack(gt_images_padded, axis=-1)  
        lq_images_stacked = np.stack(lq_images_padded, axis=-1)


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']

            # padding
            gt_images_stacked, lq_images_stacked = padding(gt_images_stacked, lq_images_stacked, gt_size)

            # random crop
            gt_images_stacked, lq_images_stacked = paired_random_crop(gt_images_stacked, lq_images_stacked, gt_size, scale,
                                                gt_path)
            
            # MaskedDenoising
            # Ref: https://arxiv.org/abs/2303.13132
            if random.random() < 0.64:
                prob = self.opt['prob']
                for i in range(lq_images_stacked.shape[2]):
                    if random.random() > 0.64:
                        lq_images_stacked[:,:,i] = input_mask(lq_images_stacked[:,:,i],prob_=self.opt['prob']+0.5, value=0.1)
                        continue
                    lq_images_stacked[:,:,i] = input_mask(lq_images_stacked[:,:,i],prob_=self.opt['prob'], value=0.1)
            else:
                # 插帧
                assert lq_images_stacked.shape[2] % 2 ==1 
                for i in range(lq_images_stacked.shape[2]):
                    if i%2 == 1:
                        # 获取两个相邻的切片
                        lq_images_prev = lq_images_stacked[:, :, i-1]
                        lq_images_next = lq_images_stacked[:, :, i+1]
                        # 在第三个维度叠加这两个切片
                        # 计算叠加后的均值
                        mean_images = np.mean(np.stack((lq_images_prev, lq_images_next), axis=2), axis=2)
                        lq_images_stacked[:,:,i] = input_mask(mean_images,prob_=self.opt['prob'] + 0.5, value=0.1)
                        continue
                    lq_images_stacked[:,:,i] = input_mask(lq_images_stacked[:,:,i],prob_=self.opt['prob'], value=0.1)

            # prevent 0 input
            zero_ratio = max(np.mean(lq_images_stacked == 0) ,np.mean(lq_images_stacked == 1))
            if (zero_ratio >0.64):
                lq_images_stacked = add_random_noise(lq_images_stacked)

            # flip, rotation augmentations
            if self.geometric_augs:
                gt_images_stacked, lq_images_stacked = random_augmentation(gt_images_stacked, lq_images_stacked)

        
        if self.opt['phase'] == 'test1':
            for i in range(lq_images_stacked.shape[2]):
                if random.random() < 0.2:
                    lq_images_stacked[:,:,i] = input_mask(lq_images_stacked[:,:,i],prob_=self.opt['prob']+0.6, value=0.1)
                    continue
                lq_images_stacked[:,:,i] = input_mask(lq_images_stacked[:,:,i],prob_=self.opt['prob'], value=0.1)

        if self.opt['phase'] == 'interpolation':
            assert lq_images_stacked.shape[2] % 2 ==1 
            for i in range(lq_images_stacked.shape[2]):
                if i%2 == 1:
                    # 获取两个相邻的切片
                    lq_images_prev = lq_images_stacked[:, :, i-1]
                    lq_images_next = lq_images_stacked[:, :, i+1]
                    # 在第三个维度叠加这两个切片
                    # 计算叠加后的均值
                    mean_images = np.mean(np.stack((lq_images_prev, lq_images_next), axis=2), axis=2)
                    lq_images_stacked[:,:,i] = input_mask(mean_images,prob_=self.opt['prob'] + 0.5, value=0.1)
                    continue
                lq_images_stacked[:,:,i] = input_mask(lq_images_stacked[:,:,i],prob_=self.opt['prob'], value=0.1)

        # # BGR to RGB, HWC to CHW, numpy to tensor
        # gt_images_stacked, lq_images_stacked = img2tensor([gt_images_stacked, gt_images_stacked],
        #                             bgr2rgb=False,
        #                             float32=True)
        # print(gt_images_stacked.shape)
        gt_tensors = torch.from_numpy(gt_images_stacked.transpose(2, 0, 1))
        lq_tensors = torch.from_numpy(lq_images_stacked.transpose(2, 0, 1))


        # normalize
        if self.mean is not None or self.std is not None:
            normalize(lq_tensors, self.mean, self.std, inplace=True)
            normalize(gt_tensors, self.mean, self.std, inplace=True)

        return {
            'lq': lq_tensors,
            'gt': gt_tensors,
            'pathes': pathes,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
    
class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            
            # provide all 0 1 input
            zero_ratio = max(np.mean(img_lq == 0) ,np.mean(img_lq == 1))
            if (zero_ratio >0.20):
                # print('adding noise')
                img_lq = add_random_noise(img_lq)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_SuperRestoration(data.Dataset):
    """支持SR双倍尺寸的三模态数据集"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.scale = opt['scale']
        self.sr_scale = 2  # SR相对GT的放大倍数
        self.gt_size = opt.get('gt_size', 256)
        self.sr_size = self.gt_size * self.sr_scale
        self.lq_size = self.gt_size

        # 初始化文件路径
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.sr_folder = opt['dataroot_sr']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        

        # 其他配置
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        self.geometric_augs = opt.get('geometric_augs', False)
        
        self.paths = self._init_paths(opt)

    def _init_paths(self, opt):
        """初始化三模态路径"""
        if self.io_backend_opt['type'] == 'lmdb':
            raise NotImplementedError("LMDB后端需要特殊处理")
        elif 'meta_info_file' in opt:
            raise NotImplementedError("元文件模式暂不支持")
        else:
            return paired_paths_from_folder(
                [self.lq_folder, self.gt_folder, self.sr_folder],
                ['lq', 'gt', 'sr'],
                self.filename_tmpl
            )

    def __getitem__(self, index):
        # 初始化文件客户端
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), 
                **self.io_backend_opt
            )

        # 读取原始数据
        paths = self.paths[index % len(self.paths)]
        img_gt = self._read_img(paths['gt_path'], 'gt')
        img_lq = self._read_img(paths['lq_path'], 'lq')
        img_sr = self._read_img(paths['sr_path'], 'sr')

        # 训练阶段处理
        if self.opt['phase'] == 'train':
            # 尺寸验证
            self._validate_sizes(img_gt, img_sr, paths['gt_path'])
            
            # 多尺度填充
            img_gt, img_lq, img_sr = self.multi_scale_padding(img_gt, img_lq, img_sr)
            
            # 多尺度裁剪
            img_gt, img_lq, img_sr = self.multi_scale_crop(img_gt, img_lq, img_sr, paths['gt_path'])
            
            # 添加噪声（仅限LQ）
            if np.random.uniform() < 0.1:  # 10%概率加噪
                img_lq = self.add_gaussian_noise(img_lq)
            
            # 同步几何增强
            if self.geometric_augs:
                img_gt, img_lq, img_sr = self.sync_augment(img_gt, img_lq, img_sr)

            # provide all 0 1 input
            zero_ratio = max(np.mean(img_lq == 0) ,np.mean(img_lq == 1))
            if (zero_ratio >0.10):
                img_lq = img_lq + 0.00000000000001
            # if (zero_ratio >0.80):
            #     # print('adding noise')
            #     img_lq = self.add_random_noise(img_lq)

        # 转换为张量
        imgs = img2tensor([img_gt, img_lq, img_sr], bgr2rgb=True, float32=True)
        img_gt, img_lq, img_sr = imgs

        # 标准化处理
        if self.mean is not None or self.std is not None:
            self.normalize_tensor(img_lq, self.mean, self.std)
            self.normalize_tensor(img_gt, self.mean, self.std)
            self.normalize_tensor(img_sr, self.mean, self.std)

        return {
            'lq': img_lq,
            'gt': {
                'hq': img_gt,
                'sr': img_sr
            },
            'lq_path': paths['lq_path'],
            'gt_path': paths['gt_path']
        }

    def __len__(self):
        return len(self.paths)

    def _read_img(self, path, key):
        """读取图像并转换色彩空间"""
        img_bytes = self.file_client.get(path, key)
        img = imfrombytes(img_bytes, float32=True)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _validate_sizes(self, gt, sr, path):
        """验证尺寸比例"""
        h_gt, w_gt = gt.shape[:2]
        h_sr, w_sr = sr.shape[:2]
        
        # 验证SR尺寸是GT的双倍
        if h_sr != h_gt * self.sr_scale or w_sr != w_gt * self.sr_scale:
            raise ValueError(
                f"SR尺寸不匹配: GT={h_gt}x{w_gt}, SR={h_sr}x{w_sr}\n路径: {path}"
            )

    def multi_scale_padding(self, gt, lq, sr):
        """多尺度同步填充"""
        target_sizes = {
            'gt': self.gt_size,
            'lq': self.lq_size,
            'sr': self.sr_size
        }

        def pad_image(img, target):
            h, w = img.shape[:2]
            pad_h = max(target - h, 0)
            pad_w = max(target - w, 0)
            if pad_h > 0 or pad_w > 0:
                img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, 
                                       cv2.BORDER_REFLECT_101)
            return img

        return (
            pad_image(gt, target_sizes['gt']),
            pad_image(lq, target_sizes['lq']),
            pad_image(sr, target_sizes['sr'])
        )

    def multi_scale_crop(self, gt, lq, sr, path):
        """多尺度联合裁剪"""
        # 生成随机裁剪位置
        h_gt, w_gt = gt.shape[:2]
        top = random.randint(0, h_gt - self.gt_size)
        left = random.randint(0, w_gt - self.gt_size)

        # GT裁剪
        gt_crop = gt[top:top+self.gt_size, left:left+self.gt_size]
        # LQ裁剪 
        lq_crop = lq[top:top+self.lq_size, left:left+self.lq_size]

        # SR裁剪 (双倍位置和尺寸)
        sr_top = top * self.sr_scale
        sr_left = left * self.sr_scale
        sr_crop = sr[sr_top:sr_top+self.sr_size, sr_left:sr_left+self.sr_size]



        return gt_crop, lq_crop, sr_crop

    def sync_augment(self, gt, lq, sr):
        """同步几何增强"""
        # 水平翻转
        if random.random() < 0.5:
            gt = cv2.flip(gt, 1)
            lq = cv2.flip(lq, 1)
            sr = cv2.flip(sr, 1)
        
        # 垂直翻转
        if random.random() < 0.5:
            gt = cv2.flip(gt, 0)
            lq = cv2.flip(lq, 0)
            sr = cv2.flip(sr, 0)
        
        # 旋转增强
        rot_type = random.choice([None, 90, 180, 270])
        if rot_type is not None:
            rot_map = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE
            }
            gt = cv2.rotate(gt, rot_map[rot_type])
            lq = cv2.rotate(lq, rot_map[rot_type])
            sr = cv2.rotate(sr, rot_map[rot_type])
        
        return gt, lq, sr

    def add_gaussian_noise(self, img, sigma_range=(1, 30)):
        """添加高斯噪声"""
        sigma = np.random.uniform(*sigma_range)
        noise = np.random.randn(*img.shape) * sigma / 255.0
        noisy_img = img + noise
        return np.clip(noisy_img, 0, 1).astype(np.float32)

    def normalize_tensor(self, tensor, mean, std):
        """张量标准化"""
        if mean is not None:
            mean = torch.tensor(mean).view(1, -1, 1, 1)
            tensor.sub_(mean)
        if std is not None:
            std = torch.tensor(std).view(1, -1, 1, 1)
            tensor.div_(std)

    def add_random_noise(self, image_stacked, mean=0.0, std=0.02):
        """添加随机高斯噪声"""
        noise = np.random.normal(mean, std, image_stacked.shape)
        noisy_image_stacked = image_stacked + noise
        noisy_image_stacked = np.clip(noisy_image_stacked, 0, 1)
        return noisy_image_stacked.astype(np.float32)

import json
class Dataset_S_IQA(data.Dataset):
    """支持SR双倍尺寸的三模态数据集"""

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.scale = opt['scale']
        self.gt_size = opt.get('gt_size', 256)
        self.lq_size = self.gt_size

        # 初始化文件路径
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.param_folder = opt['dataroot_param']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        

        # 其他配置
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        self.geometric_augs = opt.get('geometric_augs', False)
        
        self.paths = self._init_paths(opt)

    def _init_paths(self, opt):
        """初始化三模态路径"""
        if self.io_backend_opt['type'] == 'lmdb':
            raise NotImplementedError("LMDB后端需要特殊处理")
        elif 'meta_info_file' in opt:
            raise NotImplementedError("元文件模式暂不支持")
        else:
            return paired_paths_from_folder_param(
                [self.lq_folder, self.gt_folder, self.param_folder],
                ['lq', 'gt', 'param'],
                # self.filename_tmpl
            )

    def __getitem__(self, index):
        # 初始化文件客户端
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), 
                **self.io_backend_opt
            )

        # 读取原始数据
        paths = self.paths[index % len(self.paths)]
        img_gt = self._read_img(paths['gt_path'], 'gt')
        img_lq = self._read_img(paths['lq_path'], 'lq')
        param = self._read_json(paths['param_path'], 'param')
        # print('param',param)
        score = param['score']

        # 训练阶段处理
        if self.opt['phase'] == 'train':
            # 尺寸验证
            self._validate_sizes(img_gt, img_lq, paths['gt_path'])
            
            # 多尺度填充
            img_gt, img_lq = self.multi_scale_padding(img_gt, img_lq)
            
            # 多尺度裁剪
            img_gt, img_lq = self.multi_scale_crop(img_gt, img_lq, paths['gt_path'])
            
            # 同步几何增强
            if self.geometric_augs:
                img_gt, img_lq = self.sync_augment(img_gt, img_lq)


        # 转换为张量
        imgs = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        img_gt, img_lq = imgs

        # 标准化处理
        if self.mean is not None or self.std is not None:
            self.normalize_tensor(img_lq, self.mean, self.std)
            self.normalize_tensor(img_gt, self.mean, self.std)
            # self.normalize_tensor(img_sr, self.mean, self.std)

        # denoise_rate = np.tile(denoise_rate, (1, img_lq.shape[1], img_lq.shape[2]))


        return {
            'lq': {
                'img':img_lq,
                'score': score
            },
            'gt': {
                'hq': img_gt,
                # 'sr': img_sr,

            },
            'lq_path': paths['lq_path'],
            'gt_path': paths['gt_path']
        }

    def __len__(self):
        return len(self.paths)

    def _read_img(self, path, key):
        """读取图像并转换色彩空间"""
        img_bytes = self.file_client.get(path, key)
        img = imfrombytes(img_bytes, float32=True)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _read_json(self, path, key):
        """读取JSON文件"""
        json_bytes = self.file_client.get(path, key)

        return json.loads(json_bytes.decode('utf-8'))
    
    def _validate_sizes(self, gt, lq, path):
        """验证尺寸比例"""
        h_gt, w_gt = gt.shape[:2]
        h_lq, w_lq = lq.shape[:2]
        
        # 验证SR尺寸是GT的双倍
        if h_lq != h_gt * self.scale or w_lq != w_gt * self.scale:
            raise ValueError(
                f"LQ尺寸不匹配: GT={h_gt}x{w_gt}, LQ={h_lq}x{w_lq}\n路径: {path}"
            )

    def multi_scale_padding(self, gt, lq):
        """多尺度同步填充"""
        target_sizes = {
            'gt': self.gt_size,
            'lq': self.lq_size,
        }

        def pad_image(img, target):
            h, w = img.shape[:2]
            pad_h = max(target - h, 0)
            pad_w = max(target - w, 0)
            if pad_h > 0 or pad_w > 0:
                img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, 
                                       cv2.BORDER_REFLECT_101)
            return img

        return (
            pad_image(gt, target_sizes['gt']),
            pad_image(lq, target_sizes['lq']),
        )

    def multi_scale_crop(self, gt, lq, path):
        """多尺度联合裁剪"""
        # 生成随机裁剪位置
        h_gt, w_gt = gt.shape[:2]
        top = random.randint(1, h_gt - 1 - self.gt_size)
        left = random.randint(1, w_gt - 1 - self.gt_size)

        # GT裁剪
        gt_crop = gt[top:top+self.gt_size, left:left+self.gt_size]
        # LQ裁剪 
        lq_crop = lq[top:top+self.lq_size, left:left+self.lq_size]


        return gt_crop, lq_crop

    def sync_augment(self, gt, lq):
        """同步几何增强"""
        # 水平翻转
        if random.random() < 0.5:
            gt = cv2.flip(gt, 1)
            lq = cv2.flip(lq, 1)

        
        # 垂直翻转
        if random.random() < 0.5:
            gt = cv2.flip(gt, 0)
            lq = cv2.flip(lq, 0)

        
        # 旋转增强
        rot_type = random.choice([None, 90, 180, 270])
        if rot_type is not None:
            rot_map = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE
            }
            gt = cv2.rotate(gt, rot_map[rot_type])
            lq = cv2.rotate(lq, rot_map[rot_type])

        
        return gt, lq

    def add_gaussian_noise(self, img, sigma_range=(1, 30)):
        """添加高斯噪声"""
        sigma = np.random.uniform(*sigma_range)
        noise = np.random.randn(*img.shape) * sigma / 255.0
        noisy_img = img + noise
        return np.clip(noisy_img, 0, 1).astype(np.float32)

    def normalize_tensor(self, tensor, mean, std):
        """张量标准化"""
        if mean is not None:
            mean = torch.tensor(mean).view(1, -1, 1, 1)
            tensor.sub_(mean)
        if std is not None:
            std = torch.tensor(std).view(1, -1, 1, 1)
            tensor.div_(std)

    def add_random_noise(self, image_stacked, mean=0.0, std=0.02):
        """添加随机高斯噪声"""
        noise = np.random.normal(mean, std, image_stacked.shape)
        noisy_image_stacked = image_stacked + noise
        noisy_image_stacked = np.clip(noisy_image_stacked, 0, 1)
        return noisy_image_stacked.astype(np.float32)
    
class Dataset_SuperRestoration_param(data.Dataset):
    """支持SR双倍尺寸的三模态数据集"""

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.scale = opt['scale']
        self.sr_scale = 2  # SR相对GT的放大倍数
        self.gt_size = opt.get('gt_size', 256)
        self.sr_size = self.gt_size * self.sr_scale
        self.lq_size = self.gt_size

        # 初始化文件路径
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.sr_folder = opt['dataroot_sr']
        self.param_folder = opt['dataroot_param']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        

        # 其他配置
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        self.geometric_augs = opt.get('geometric_augs', False)

        self.denoise_rate = opt.get('denoise_rate', True)
        
        self.paths = self._init_paths(opt)

    def _init_paths(self, opt):
        """初始化三模态路径"""
        if self.io_backend_opt['type'] == 'lmdb':
            raise NotImplementedError("LMDB后端需要特殊处理")
        elif 'meta_info_file' in opt:
            raise NotImplementedError("元文件模式暂不支持")
        else:
            return paired_paths_from_folder_param(
                [self.lq_folder, self.gt_folder, self.sr_folder, self.param_folder],
                ['lq', 'gt', 'sr', 'param'],
                # self.filename_tmpl
            )

    def __getitem__(self, index):
        # 初始化文件客户端
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), 
                **self.io_backend_opt
            )

        # 读取原始数据
        paths = self.paths[index % len(self.paths)]
        img_gt = self._read_img(paths['gt_path'], 'gt')
        img_lq = self._read_img(paths['lq_path'], 'lq')
        img_sr = self._read_img(paths['sr_path'], 'sr')
        param = self._read_json(paths['param_path'], 'param')
        # print('param',param)
        if self.denoise_rate:
            denoise_rate = param['denoise_rate'] if param['denoise_rate'] is not None else 1
        else:
            denoise_rate = 1

        # 训练阶段处理
        if self.opt['phase'] == 'train':
            # 尺寸验证
            self._validate_sizes(img_gt, img_sr, paths['gt_path'])
            
            # 多尺度填充
            img_gt, img_lq, img_sr = self.multi_scale_padding(img_gt, img_lq, img_sr)
            
            # 多尺度裁剪
            img_gt, img_lq, img_sr = self.multi_scale_crop(img_gt, img_lq, img_sr, paths['gt_path'])
            
            # 添加噪声（仅限LQ）
            if np.random.uniform() < 0.1:  # 10%概率加噪
                img_lq = self.add_gaussian_noise(img_lq)
            
            # 同步几何增强
            if self.geometric_augs:
                img_gt, img_lq, img_sr = self.sync_augment(img_gt, img_lq, img_sr)

            # provide all 0 1 input
            zero_ratio = max(np.mean(img_lq == 0) ,np.mean(img_lq == 1))
            if (zero_ratio >0.10):
                img_lq = img_lq + 0.00000000000001
            # if (zero_ratio >0.80):
            #     # print('adding noise')
            #     img_lq = self.add_random_noise(img_lq)

        # 转换为张量
        imgs = img2tensor([img_gt, img_lq, img_sr], bgr2rgb=True, float32=True)
        img_gt, img_lq, img_sr = imgs

        # 标准化处理
        if self.mean is not None or self.std is not None:
            self.normalize_tensor(img_lq, self.mean, self.std)
            self.normalize_tensor(img_gt, self.mean, self.std)
            self.normalize_tensor(img_sr, self.mean, self.std)

        # print('img_lq',img_lq.shape)
        # 将clear_point_alpha复制成img_lq大小的张量
        denoise_rate = np.tile(denoise_rate, (1, img_lq.shape[1], img_lq.shape[2]))
        # print('clear_point_alpha',clear_point_alpha.shape)

        # 给img_lq添加clear_point_alpha层[1, 3, 1429, 1000]-> [1, 4, 1429, 1000]
        # if clear_point_alpha is not None:
        #     img_lg = img_lq * clear_point_alpha
            # img_lq = np.concatenate([img_lq, clear_point_alpha], axis=0)
            # img_lq = torch.from_numpy(img_lq).float()

        return {
            'lq': {
                'img':img_lq,
                'denoise_rate': torch.from_numpy(denoise_rate).float()
            },
            'gt': {
                'hq': img_gt,
                'sr': img_sr,

            },
            'lq_path': paths['lq_path'],
            'gt_path': paths['gt_path']
        }

    def __len__(self):
        return len(self.paths)

    def _read_img(self, path, key):
        """读取图像并转换色彩空间"""
        img_bytes = self.file_client.get(path, key)
        img = imfrombytes(img_bytes, float32=True)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _read_json(self, path, key):
        """读取JSON文件"""
        json_bytes = self.file_client.get(path, key)

        return json.loads(json_bytes.decode('utf-8'))
    
    def _validate_sizes(self, gt, sr, path):
        """验证尺寸比例"""
        h_gt, w_gt = gt.shape[:2]
        h_sr, w_sr = sr.shape[:2]
        
        # 验证SR尺寸是GT的双倍
        if h_sr != h_gt * self.sr_scale or w_sr != w_gt * self.sr_scale:
            raise ValueError(
                f"SR尺寸不匹配: GT={h_gt}x{w_gt}, SR={h_sr}x{w_sr}\n路径: {path}"
            )

    def multi_scale_padding(self, gt, lq, sr):
        """多尺度同步填充"""
        target_sizes = {
            'gt': self.gt_size,
            'lq': self.lq_size,
            'sr': self.sr_size
        }

        def pad_image(img, target):
            h, w = img.shape[:2]
            pad_h = max(target - h, 0)
            pad_w = max(target - w, 0)
            if pad_h > 0 or pad_w > 0:
                img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, 
                                       cv2.BORDER_REFLECT_101)
            return img

        return (
            pad_image(gt, target_sizes['gt']),
            pad_image(lq, target_sizes['lq']),
            pad_image(sr, target_sizes['sr'])
        )

    def multi_scale_crop(self, gt, lq, sr, path):
        """多尺度联合裁剪"""
        # 生成随机裁剪位置
        h_gt, w_gt = gt.shape[:2]
        top = random.randint(1, h_gt - 1 - self.gt_size)
        left = random.randint(1, w_gt - 1 - self.gt_size)

        # GT裁剪
        gt_crop = gt[top:top+self.gt_size, left:left+self.gt_size]
        # LQ裁剪 
        lq_crop = lq[top:top+self.lq_size, left:left+self.lq_size]

        # SR裁剪 (双倍位置和尺寸)
        sr_top = top * self.sr_scale
        sr_left = left * self.sr_scale
        sr_crop = sr[sr_top:sr_top+self.sr_size, sr_left:sr_left+self.sr_size]



        return gt_crop, lq_crop, sr_crop

    def sync_augment(self, gt, lq, sr):
        """同步几何增强"""
        # 水平翻转
        if random.random() < 0.5:
            gt = cv2.flip(gt, 1)
            lq = cv2.flip(lq, 1)
            sr = cv2.flip(sr, 1)
        
        # 垂直翻转
        if random.random() < 0.5:
            gt = cv2.flip(gt, 0)
            lq = cv2.flip(lq, 0)
            sr = cv2.flip(sr, 0)
        
        # 旋转增强
        rot_type = random.choice([None, 90, 180, 270])
        if rot_type is not None:
            rot_map = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE
            }
            gt = cv2.rotate(gt, rot_map[rot_type])
            lq = cv2.rotate(lq, rot_map[rot_type])
            sr = cv2.rotate(sr, rot_map[rot_type])
        
        return gt, lq, sr

    def add_gaussian_noise(self, img, sigma_range=(1, 30)):
        """添加高斯噪声"""
        sigma = np.random.uniform(*sigma_range)
        noise = np.random.randn(*img.shape) * sigma / 255.0
        noisy_img = img + noise
        return np.clip(noisy_img, 0, 1).astype(np.float32)

    def normalize_tensor(self, tensor, mean, std):
        """张量标准化"""
        if mean is not None:
            mean = torch.tensor(mean).view(1, -1, 1, 1)
            tensor.sub_(mean)
        if std is not None:
            std = torch.tensor(std).view(1, -1, 1, 1)
            tensor.div_(std)

    def add_random_noise(self, image_stacked, mean=0.0, std=0.02):
        """添加随机高斯噪声"""
        noise = np.random.normal(mean, std, image_stacked.shape)
        noisy_image_stacked = image_stacked + noise
        noisy_image_stacked = np.clip(noisy_image_stacked, 0, 1)
        return noisy_image_stacked.astype(np.float32)
    
class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)


            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:            
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)
            
            # flip, rotation            
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
