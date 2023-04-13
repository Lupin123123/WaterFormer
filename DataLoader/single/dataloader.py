import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import os

random.seed(1143)

def populate_train_list(images_path):
    train_list = []
    val_list = []

    file_path = os.path.join(images_path, "*.png")
    image_list = glob.glob(file_path)
    upper = 0.9*len(image_list)

    i = 0
    for image_path in image_list:
        if (i<upper): train_list.append(image_path)
        else: val_list.append(image_path)
        i += 1

    random.shuffle(train_list)
    random.shuffle(val_list)
    
    return train_list, val_list


class dehazing_loader(data.Dataset):
    def __init__(self, images_path, mode='train'):
        self.train_list, self.val_list = populate_train_list(images_path)
        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):
        data_path = self.data_list[index]
        data = Image.open(data_path)
        data = data.resize((480, 640), Image.ANTIALIAS)
        data = (np.asarray(data) / 255.0)
        data = torch.from_numpy(data).float()
        return data.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)
