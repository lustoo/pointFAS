import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import platform
import sys


# filp and RGB
def OriImg_loader(path, flip=False):
    if flip:
        random_int = np.random.randint(low=0, high=1)
        RGBimg = Image.open(path).convert('RGB')
        HSVimg = Image.open(path).convert('HSV')
        if random_int == 1:
            RGBimg = RGBimg.transpose(Image.FLIP_LEFT_RIGHT)
            HSVimg = HSVimg.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        # 读取图片,转化成rgb和hsv,现在不要hsv
        RGBimg = Image.open(path).convert('RGB')
        # HSVimg = Image.open(path).convert('HSV')
    return RGBimg, None


# 插值 深度图缩放时用,现在不要缩放
def DepthImg_loader(path, imgsize=32):
    img = Image.open(path)
    # re_img = img.resize((imgsize, imgsize), resample=Image.BICUBIC)
    return img  # re_img


#
class DatasetLoader(Dataset):
    def __init__(self, name, getreal, input_channels=3, trans_depth=None, trans_img=None, trans_hsv=None, \
                 oriimg_loader=OriImg_loader, depthimg_loader=DepthImg_loader, root='', args=None):
        if 'tlinux3' in str(platform.platform()):  # 机智
            root = '/apdcephfs/private_v_loystelu/data/FAS'
        elif 'tlinux2' in str(platform.platform()):  # p40
            root = '/data/home/v_loystelu/data/FAS/'
        else:
            print('data root error in DatasetLoader************************************')

        self.name = name
        self.root = os.path.expanduser(root)  # 等于~/,自己的用户名目录下
        self.input_channels = input_channels
        imgs = []  # 放路径元组

        if name == 'replay':
            self.root_img = os.path.join(self.root, 'replayattack_face_small/train')
            self.root_dep = os.path.join(self.root, 'replayattack_face_small_depth/train')
            if getreal:
                filename = 'list_train_real.txt'
            else:
                filename = 'list_train_attack.txt'
            fh_image = open(os.path.join(self.root_img, filename), 'r')
            fh_depth = open(os.path.join(self.root_dep, filename), 'r')

            # 真脸的深度图也拿出来
            if getreal:
                for (image_path, depth_path) in zip(fh_image, fh_depth):
                    image_path = image_path.strip('\n')
                    depth_path = depth_path.strip('\n')
                    imgs.append((self.root_img + '/real/' + image_path, self.root_dep + '/real/' + depth_path, 0))
            else:
                for image_path in fh_image:
                    image_path = image_path.strip('\n')
                    imgs.append((self.root_img + '/attack/' + image_path, '', 1))


        elif name == 'oulu':
            self.root_img = os.path.join(self.root, 'Oulu_face_small')
            self.root_dep = os.path.join(self.root, 'Oulu_face_small_depth')
            if getreal:
                filename = 'list_train_real.txt'
            else:
                filename = 'list_train_fake.txt'
            fh_image = open(os.path.join(self.root_img, filename), 'r')
            fh_depth = open(os.path.join(self.root_dep, filename), 'r')

            if getreal:
                for (image_path, depth_path) in zip(fh_image, fh_depth):
                    image_path = image_path.strip('\n')
                    depth_path = depth_path.strip('\n')
                    imgs.append((self.root + '/' + image_path, self.root + '/' + depth_path, 0))
            else:
                for image_path in fh_image:
                    image_path = image_path.strip('\n')
                    imgs.append((self.root + '/' + image_path, '', 1))


        elif name == 'casia':
            self.root_img = os.path.join(self.root, 'CASIA_face_small')
            self.root_dep = os.path.join(self.root, 'CASIA_face_small_depth')
            if getreal:
                filename = 'list_train_A.txt'
            else:
                filename = 'list_train_B.txt'
            fh_image = open(os.path.join(self.root_img, filename), 'r')
            fh_depth = open(os.path.join(self.root_dep, filename), 'r')

            if getreal:
                for (image_path, depth_path) in zip(fh_image, fh_depth):
                    image_path = image_path.strip('\n')
                    depth_path = depth_path.strip('\n')
                    imgs.append((self.root_img + '/train/' + image_path, self.root_dep + '/train/' + depth_path, 0))
            else:
                for image_path in fh_image:
                    image_path = image_path.strip('\n')
                    imgs.append((self.root_img + '/train/' + image_path, '', 1))

        elif name == 'msu':
            self.root_img = os.path.join(self.root, 'MSU_MFSD_face_small')
            self.root_dep = os.path.join(self.root, 'MSU_MFSD_face_small_depth')
            if getreal:
                filename = 'list_train_real.txt'
            else:
                filename = 'list_train_fake.txt'

            fh_image = open(os.path.join(self.root_img, 'list', filename), 'r')
            fh_depth = open(os.path.join(self.root_dep, 'list', filename), 'r')

            if getreal:
                for (image_path, depth_path) in zip(fh_image, fh_depth):
                    image_path = image_path.strip('\n')
                    depth_path = depth_path.strip('\n')
                    imgs.append((self.root_img + '/' + image_path, self.root_dep + '/' + depth_path, 0))
            else:
                for image_path in fh_image:
                    image_path = image_path.strip('\n')
                    imgs.append((self.root_img + '/' + image_path, '', 1))

        self.imgs = imgs
        self.trans_depth = trans_depth
        self.trans_img = trans_img
        self.trans_hsv = trans_hsv
        self.oriimg_loader = oriimg_loader
        self.depthimg_loader = depthimg_loader
        if args is not None:
            self.return_path = args.return_path
        else:
            self.return_path = False

    def __getitem__(self, index):
        ori_img_dir, depth_img_dir, label = self.imgs[index]
        ori_rgbimg, ori_hsvimg = self.oriimg_loader(ori_img_dir)  # 读取图片,转为rgb和hsv
        if depth_img_dir == '':  # 假脸的时候路径为空
            depth_img = Image.new('L', (256, 256))  # 原来是32
        else:
            depth_img = self.depthimg_loader(depth_img_dir)

        ori_rgbimg = self.trans_img(ori_rgbimg)

        depth_img = self.trans_depth(depth_img)

        if self.input_channels == 6:
            ori_hsvimg = self.trans_hsv(ori_hsvimg)
            ori_catimg = torch.cat([ori_rgbimg, ori_hsvimg], 0)
            if self.return_path:
                return ori_catimg, depth_img, label, ori_img_dir
            else:
                return ori_catimg, depth_img, label

        elif self.input_channels == 3:
            if self.return_path:
                return ori_rgbimg, depth_img, label, ori_img_dir
            else:
                return ori_rgbimg, depth_img, label
        else:
            raise NotImplementedError('invlid input_channels')

    def __len__(self):
        return len(self.imgs)


def get_dataset_loader(name, getreal, batch_size, input_channels=3, args=None):
    # torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
    trans_depth = transforms.Compose([transforms.ToTensor()])
    trans_img = transforms.Compose([transforms.ToTensor()])
    trans_hsv = transforms.Compose([transforms.ToTensor()])

    # dataset and data loader
    dataset = DatasetLoader(name=name,
                            getreal=getreal,
                            input_channels=input_channels,
                            trans_depth=trans_depth,
                            trans_img=trans_img,
                            trans_hsv=trans_hsv,
                            args=args
                            )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True)

    return data_loader


if __name__ == '__main__':
    sys.path.append('../')
    from utils_.config import config
    #不要给args,
    loader = get_dataset_loader(name='msu', getreal=True, batch_size=10, input_channels=3)
    loader_iter = iter(loader)
    length = len(loader_iter)
    img_real, depth_real, label_real = loader_iter.next()
    print(loader, img_real.shape, depth_real.shape, label_real)

    loader2 = get_dataset_loader(name='msu', getreal=False, batch_size=10, input_channels=3)
    loader_iter2 = iter(loader2)
    length2 = len(loader_iter2)
    img_real2, depth_real2, label_real2 = loader_iter2.next()
    print(loader2, img_real2.shape, depth_real2.shape, label_real2)
    print('ed')
