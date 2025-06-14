
import os
import sys
import json
import glob
import torch
import itertools
import numpy as np
from PIL import Image
from scipy import misc
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class NeuralPhysDataset(Dataset):
    def __init__(self, data_filepath, flag, seed, object_name="ndvi"):
        self.seed = seed
        self.flag = flag
        self.object_name = object_name
        self.data_filepath = data_filepath
        self.all_filelist = self.get_all_filelist()

    def get_all_filelist(self):
        filelist = []
        obj_filepath = os.path.join(self.data_filepath, self.object_name)
        # get the video ids based on training or testing data
        with open(os.path.join('datainfo', self.object_name, f'data_split_dict_{self.seed}.json'), 'r') as file:
            seq_dict = json.load(file)
        vid_list = seq_dict[self.flag]

        # go through all the selected videos and get the triplets: input(t, t+1), output(t+2)
        for vid_idx in vid_list:
            seq_filepath = os.path.join(obj_filepath, str(vid_idx))
            num_frames = len(os.listdir(seq_filepath))
            suf = os.listdir(seq_filepath)[0].split('.')[-1]
            for p_frame in range(num_frames - 3):
                par_list = []
                for p in range(4):
                    par_list.append(os.path.join(seq_filepath, str(p_frame + p) + '.' + suf))
                filelist.append(par_list)
        # print(filelist)
        return filelist

    def __len__(self):
        return len(self.all_filelist)

    # 0, 1 -> 2, 3
    def __getitem__(self, idx):
        par_list = self.all_filelist[idx]
        data = []
        for i in range(2):
            data.append(self.get_data(par_list[i])) # 0, 1
        data = torch.cat(data, 2)
        target = []
        target.append(self.get_data(par_list[-2])) # 2
        target.append(self.get_data(par_list[-1])) # 3
        target = torch.cat(target, 2)
        filepath = '_'.join(par_list[0].split('/')[-2:])
        return data, target, filepath

    def get_data(self, filepath):
        # print(filepath)
        data = np.load(filepath)
        data = (torch.tensor(data).unsqueeze(0).float() - (-3000)) / (9951 - (-3000))
        # [0,1]
        # 1: (-3000, 9994)
        return data


if __name__ == '__main__':
    data_filepath = 'data'
    dataset = NeuralPhysDataset(data_filepath, 'train', 1, 'ndvi')
    