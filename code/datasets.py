import os
from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

ANNOTATION_PATH_TRAIN = './dataset/train_annotation.txt'
ANNOTATION_PATH_TEST = './dataset/test_annotation.txt'
IMG_PATH_TRAIN = './dataset/train'
IMG_PATH_TEST = './dataset/test'

Transformer = transforms.Compose([transforms.CenterCrop((480, 480)),
                                transforms.Resize((240, 240)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class ThermalLoader(Dataset):
    def __init__(self, args, split='train'):
        assert split in ['train', 'test']
        self.dict = {}
        self.split = split
        self.args = args

        if self.split == 'test':
            self.img_path = IMG_PATH_TEST
            self.anno_path = ANNOTATION_PATH_TEST
        else:
            self.img_path = IMG_PATH_TRAIN
            self.anno_path = ANNOTATION_PATH_TRAIN
        
        with open(self.anno_path, 'r') as f:
            data = f.readlines()
            for idx, line in enumerate(data):
                img_name = line.split(' ')[0]
                img_label = line.split(' ')[1]
                self.dict[idx] = (img_name, float(img_label))

        self.transfomer = Transformer

    def __len__(self) -> int:
        return len(self.dict)

    def __getitem__(self, idx):
        img_name, img_label = self.dict[idx]

        img = Image.open((self.img_path + '/' + img_name))
        img = self.transfomer(img)

        sample = {'img': img, 'label': torch.tensor(img_label)}
        return sample