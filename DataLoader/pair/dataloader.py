import torch
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import os
from torch.utils.data import DataLoader


def populate_train_list(orig_images_path, hazy_images_path):

    train_list = []
    val_list = []

    image_list_haze = glob.glob(os.path.join(hazy_images_path, "*.png"))
    image_list_haze += glob.glob(os.path.join(hazy_images_path, "*.jpg"))
    
    ##############################################################

    tmp_dict = {}
    for image_path in image_list_haze:
        key = image_path.split("\\")[-1]
        tmp_dict[key] = []
        tmp_dict[key].append(image_path)
    
    ##############################################################

    train_keys = []
    len_keys = int(0.7*len(tmp_dict.keys()))
    for i in range(len_keys):
        train_keys.append(list(tmp_dict.keys())[i])

    for key in list(tmp_dict.keys()):
        if key in train_keys:
            ori_path = os.path.join(orig_images_path, key)
            hazy_path = os.path.join(orig_images_path, key)
            # print(ori_path)
            train_list.append((ori_path, hazy_path))
        else:
            val_list.append((os.path.join(orig_images_path, key), os.path.join(orig_images_path, key)))

    random.shuffle(train_list)
    random.shuffle(val_list)

    return train_list, val_list


class dehazing_loader(data.Dataset):
    def __init__(self, orig_images_path, hazy_images_path, mode='train'):
        self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path)
        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):
        data_orig_path, data_hazy_path = self.data_list[index]

        data_orig = Image.open(data_orig_path)
        data_hazy = Image.open(data_hazy_path)

        data_orig = data_orig.resize((256, 256), Image.ANTIALIAS)
        data_hazy = data_hazy.resize((256, 256), Image.ANTIALIAS)

        data_orig = (np.asarray(data_orig) / 255.0)
        data_hazy = (np.asarray(data_hazy) / 255.0)

        data_orig = torch.from_numpy(data_orig).float()
        data_hazy = torch.from_numpy(data_hazy).float()

        return data_orig.permute(2, 0, 1), data_hazy.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)