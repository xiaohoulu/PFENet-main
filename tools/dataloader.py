import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image



########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        mask /= 255
        return image, mask


class RandomCrop(object):
    def __call__(self, image, mask):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return image[p0:p1, p2:p3], mask[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2) == 0:
            return image[:, ::-1], mask[:, ::-1]
        else:
            return image, mask


class RandomRotate(object):
    def __call__(self, image, mask):
        degree = 10
        rows, cols, channels = image.shape
        random_rotate = random.random() * 2 * degree - degree
        rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), random_rotate, 1)
        '''
        第一个参数：旋转中心点
        第二个参数：旋转角度
        第三个参数：缩放比例
        '''
        image = cv2.warpAffine(image, rotate, (cols, rows))
        mask = cv2.warpAffine(mask, rotate, (cols, rows))
        # contour = cv2.warpAffine(contour, rotate, (cols, rows))

        return image, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask


class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        # image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        return image, mask


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, cfg):
        self.trainsize = cfg.trainsize
        self.images = [cfg.image_root + f for f in os.listdir(cfg.image_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [cfg.gt_root + f for f in os.listdir(cfg.gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


# class PolypDataset(data.Dataset):
#     """
#     dataloader for polyp segmentation tasks
#     """
#
#     def __init__(self, cfg):
#         self.trainsize = cfg.trainsize
#         self.images = [cfg.image_root + f for f in os.listdir(cfg.image_root) if
#                        f.endswith('.jpg') or f.endswith('.png')]
#         self.gts = [cfg.gt_root + f for f in os.listdir(cfg.gt_root) if f.endswith('.png')]
#         self.images = sorted(self.images)
#         self.gts = sorted(self.gts)
#         self.filter_files()
#         self.size = len(self.images)
#         mean = np.array([[[124.55, 118.90, 102.94]]])
#         std = np.array([[[56.77, 55.97, 57.50]]])
#         self.Normalize = Normalize(mean=mean, std=std)
#         self.RandomCrop = RandomCrop()
#         self.RandomFlip = RandomFlip()
#         self.RandomRotate = RandomRotate()
#         self.Resize = Resize(352, 352)
#         self.ToTensor = ToTensor()
#
#     def __getitem__(self, index):
#
#         image = cv2.imread(self.images[index])[:, :, ::-1].astype(np.float32)
#         mask = cv2.imread(self.gts[index], 0).astype(np.float32)
#         image, mask = self.Normalize(image, mask)
#         image, mask = self.Resize(image, mask)
#         image, mask = self.RandomCrop(image, mask)
#         image, mask = self.RandomFlip(image, mask)
#         image, mask = self.RandomRotate(image, mask)
#         image = cv2.resize(image, dsize=(352, 352), interpolation=cv2.INTER_LINEAR)
#         mask = cv2.resize(mask, dsize=(352, 352), interpolation=cv2.INTER_LINEAR)
#         image = torch.from_numpy(image).permute(2, 0, 1)
#         mask = torch.from_numpy(mask).unsqueeze(0)
#         return image, mask
#
#     def filter_files(self):
#         assert len(self.images) == len(self.gts)
#         images = []
#         gts = []
#         for img_path, gt_path in zip(self.images, self.gts):
#             img = Image.open(img_path)
#             gt = Image.open(gt_path)
#             if img.size == gt.size:
#                 images.append(img_path)
#                 gts.append(gt_path)
#         self.images = images
#         self.gts = gts
#
#     def rgb_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')
#
#     def binary_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             # return img.convert('1')
#             return img.convert('L')
#
#     def resize(self, img, gt):
#         assert img.size == gt.size
#         w, h = img.size
#         if h < self.trainsize or w < self.trainsize:
#             h = max(h, self.trainsize)
#             w = max(w, self.trainsize)
#             return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
#         else:
#             return img, gt
#
#     def __len__(self):
#         return self.size


def get_loader(cfg, shuffle=True, num_workers=4, pin_memory=True):
    dataset = PolypDataset(cfg)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=cfg.batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    """load test dataset (batchsize=1)"""

    def __init__(self, image_root, gt_root, testsize, val=True):
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        if val:
            self.images = random.sample(self.images, len(self.images) // 10)
            self.gts = random.sample(self.gts, len(self.gts) // 10)

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_ori = self.images[index]
        image = self.rgb_loader(self.images[index])
        image = self.transform(image).unsqueeze(0)
        gt = self.gts[index]
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0]
        elif name.endswith('.png'):
            name = name.split('.png')[0]
        return image_ori, image, gt, name

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class test_loader_faster(data.Dataset):
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.size = len(self.images)

    def __getitem__(self, index):
        images = self.rgb_loader(self.images[index])
        images = self.transform(images)

        img_name_list = self.images[index]

        return images, img_name_list

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
