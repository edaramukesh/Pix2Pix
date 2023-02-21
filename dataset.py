from torch.utils.data import Dataset
import torchvision.transforms as T
import random
from PIL import Image,ImageEnhance
import numpy as np

img_transform = T.Compose([
                T.Resize((500,500)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

trg_transform = T.Compose([
                T.Resize((500,500)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                ])

def cv_random_flip(img:Image, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomCrop(image, label):
    border=20
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomRotation( image, label):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-25, 25)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
    return image,label

def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


# class CustomDataset(Dataset):
#     def __init__(self,inputs,targets,phase="train"):
#         self.phase = phase
#         self.inputs = inputs
#         self.targets = targets

#     def __getitem__(self, index):
#         img = img_transform(self.inputs[index])
#         trg = trg_transform(self.targets[index])
#         if self.phase == "Train":
#             aug = random.choice([1,2,3,4,5])
#             if aug == 1:
#                 img,trg = cv_random_flip(img,trg)
#             elif aug == 2:
#                 img,trg = randomCrop(img,trg)
#             elif aug == 3:
#                 img,trg = randomRotation(img,trg)
#             elif aug == 4:
#                 img,trg = colorEnhance(img,trg)
#             else:
#                 img,trg = img,trg

#         return img,trg

#     def __len__(self):
#         return len(self.inputs)

import random
import torch
from glob import glob
import os

tra = T.Compose([T.PILToTensor(),T.Resize((256,512))])
transform = T.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1))
class custom_dataset(Dataset):
  def __init__(self,path,train):
    self.path = path
    self.train = train
    self.images = [x for x in glob(os.path.join(path,"*"))]
  def __len__(self):
    return len(self.images)
  def __getitem__(self,idx):
    img = self.images[idx]
    img = Image.open(img)
    img = tra(img)
    img,trg = img[:,:,:256],img[:,:,256:]
    if self.train == "True":
      if random.randint(0,1)==1:
        img = transform(img)
      if random.randint(0,1)==1:
        img = torch.flip(img,[0])
        trg = torch.flip(trg,[0])
    return img.float(),trg.float()

