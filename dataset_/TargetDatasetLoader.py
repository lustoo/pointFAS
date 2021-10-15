import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset
from PIL import Image
import platform


# 去掉了hsv转换
def default_loader(path):
    RGBimg = Image.open(path).convert('RGB')
    # HSVimg = Image.open(path).convert('HSV')
    return RGBimg, None  # HSVimg


'''
return: rgbimg, label, video_id
'''


class DatasetLoader(Dataset):
    def __init__(self, name, input_channels=3, trans_img=None, trans_hsv=None, loader=default_loader, root=''):

        if 'tlinux3' in str(platform.platform()):  # 机智
            root = '/apdcephfs/private_v_loystelu/data/FAS'
        elif 'tlinux2' in str(platform.platform()):  # p40
            root = '/data/home/v_loystelu/data/FAS/'
        else:
            print('data root error in DatasetLoader************************************')
        self.name = name
        self.root = os.path.expanduser(root)
        self.input_channels = input_channels

        imgs = []

        if name == 'replay':
            self.root_img = os.path.join(self.root, 'replayattack_face_small/test')
            for filename in ['list_test_real_with_videoID.txt', 'list_test_fake_with_videoID.txt']:
                fh_image = open(os.path.join(self.root_img, filename), 'r')
                for item in fh_image:
                    image_path = item.strip('\n').split(' ')[0]
                    video_id = item.strip('\n').split(' ')[1]
                    if 'real' in filename:
                        imgs.append((self.root_img + '/real/' + image_path, 0, int(video_id)))
                    else:
                        imgs.append((self.root_img + '/attack/' + image_path, 1, int(video_id)))

        elif name == 'oulu':
            self.root_img = os.path.join(self.root, 'Oulu_face_small')
            for filename in ['list_test_real_with_videoID.txt', 'list_test_fake_with_videoID.txt']:
                fh_image = open(os.path.join(self.root_img, filename), 'r')
                for item in fh_image:
                    image_path = item.strip('\n').split(' ')[0]
                    video_id = item.strip('\n').split(' ')[1]
                    if 'real' in filename:
                        imgs.append((self.root + '/' + image_path, 0, int(video_id)))
                    else:
                        imgs.append((self.root + '/' + image_path, 1, int(video_id)))

        elif name == 'casia':
            self.root_img = os.path.join(self.root, 'CASIA_face_small')
            for filename in ['list_test_real_with_videoID.txt', 'list_test_fake_with_videoID.txt']:
                fh_image = open(os.path.join(self.root_img, filename), 'r')
                for item in fh_image:
                    image_path = item.strip('\n').split(' ')[0]
                    video_id = item.strip('\n').split(' ')[1]
                    if 'real' in filename:
                        imgs.append((self.root_img + '/test/' + image_path, 0, int(video_id)))
                    else:
                        imgs.append((self.root_img + '/test/' + image_path, 1, int(video_id)))

        elif name == 'msu':
            self.root_img = os.path.join(self.root, 'MSU_MFSD_face_small')
            for filename in ['list_test_real_with_videoID.txt', 'list_test_fake_with_videoID.txt']:
                fh_image = open(os.path.join(self.root_img, 'list', filename), 'r')
                for item in fh_image:
                    image_path = item.strip('\n').split(' ')[0]
                    video_id = item.strip('\n').split(' ')[1]
                    # 把real和fake也都加进去了
                    if 'real' in filename:
                        imgs.append((self.root_img + '/' + image_path, 0, int(video_id)))
                    else:
                        imgs.append((self.root_img + '/' + image_path, 1, int(video_id)))

        self.imgs = imgs
        self.trans_img = trans_img
        self.trans_hsv = trans_hsv
        self.loader = loader

    def __getitem__(self, index):
        fn, label, video_id = self.imgs[index]
        fn = os.path.join(self.root, fn)
        rgbimg, hsvimg = self.loader(fn)
        if self.trans_img is not None:
            rgbimg = self.trans_img(rgbimg)
            # hsvimg = self.trans_hsv(hsvimg)


        if self.input_channels == 6:
            catimg = torch.cat([rgbimg, hsvimg], 0)

            return catimg, label, video_id
        elif self.input_channels == 3:
            return rgbimg, label, video_id
        else:
            raise NotImplementedError('invlid input_channels')

    def __len__(self):
        return len(self.imgs)


def get_tgtdataset_loader(name, batch_size, shuffle, input_channels=3):
    # pre_process = transforms.Compose([transforms.ToTensor(),
    #                                   transforms.Normalize(
    #                                   mean=[0.485, 0.456, 0.406],
    #                                   std=[0.229, 0.224, 0.225])])      

    # pre_process = transforms.Compose([transforms.ToTensor()])  

    if name == 'replay':
        img_normMean = [0.5268871, 0.37428004, 0.3215838]
        img_normStd = [0.27957556, 0.24527827, 0.24756251]
        img_hsv_normMean = [0.1605213, 0.46848887, 0.5346703]
        img_hsv_normStd = [0.2352692, 0.27611148, 0.280031]
    elif name == 'oulu':
        img_normMean = [0.6176847, 0.47137412, 0.38603282]
        img_normStd = [0.2421483, 0.2268885, 0.22432703]
        img_hsv_normMean = [0.088722385, 0.411151, 0.6216195]
        img_hsv_normStd = [0.14018315, 0.22907063, 0.23983885]
    elif name == 'casia':
        img_normMean = [0.48128667, 0.38795075, 0.36555228]
        img_normStd = [0.18872307, 0.1659293, 0.1703307]
        img_hsv_normMean = [0.22680725, 0.31751105, 0.49798396]
        img_hsv_normStd = [0.26527423, 0.16641639, 0.1819909]
    elif name == 'msu':
        img_normMean = [0.62782997, 0.5074975, 0.44612858]
        img_normStd = [0.32331365, 0.31647, 0.3080219]
        img_hsv_normMean = [0.094000496, 0.38674647, 0.63201404]
        img_hsv_normStd = [0.15820774, 0.28629726, 0.32039723]
    else:
        pass

    trans_img = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.43170033500000005, 0.3417055925, 0.29942842000000003],
        #             std=[0.188546255, 0.17732195, 0.1756699075])
    ])

    trans_hsv = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.10238253274999999, 0.27885213000000003, 0.437904375],
        #             std=[0.14091628, 0.17044607, 0.185556745])
    ])

    # dataset and data loader
    dataset = DatasetLoader(name=name,
                            input_channels=input_channels,
                            trans_img=trans_img,
                            trans_hsv=trans_hsv,
                            )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True)

    return data_loader


if __name__ == '__main__':
    import sys

    sys.path.append('../')
    from utils_.config import config

    loader = get_tgtdataset_loader(name='msu', batch_size=10, shuffle=False, input_channels=3)
    loader_iter = iter(loader)
    length = len(loader_iter)
    img_real, label, vid = loader_iter.next()
    print(loader, img_real.shape, label, vid)
