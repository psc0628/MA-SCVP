import numpy as np
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SCVPDataset(Dataset):
    def __init__(self, grid_path, vss_path, label_path, transform=None):
        self.transform  = transform
        self.grid = np.load(grid_path, allow_pickle=True)
        self.label = np.load(label_path, allow_pickle=True)
        self.vss = np.load(vss_path, allow_pickle=True)

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, index):
        sample = {'grid': self.grid[index], 
                  'vss': self.vss[index], 
                  'label': self.label[index]}
        if self.transform:
            sample = self.transform(sample)

        return sample['grid'], sample['vss'], sample['label']


class ToTensor(object):
    def __call__(self, sample):
        grid = sample['grid']
        vss = sample['vss']
        label = sample['label']

        return {'grid': torch.from_numpy(grid).to(torch.float32),
                'vss': torch.from_numpy(vss).to(torch.float32), 
                'label': torch.from_numpy(label).to(torch.float32)}

class To3DGrid(object):
    def __call__(self, sample):
        grid = sample['grid']
        label = sample['label']
        vss = sample['vss']

        grid = np.reshape(grid, (1, 32, 32, 32))

        return {'grid': grid,
                'vss': vss,
                'label': label}
    
            
if __name__ == "__main__":
    dataset = SCVPDataset('/home/huhao/code_python/Datasets/LongTail_MA-SCVP/MASCVP_LongTailSample_8_grid.npy', '/home/huhao/code_python/Datasets/LongTail_MA-SCVP/MASCVP_LongTailSample_8_vss.npy', '/home/huhao/code_python/Datasets/LongTail_MA-SCVP/MASCVP_LongTailSample_8_label.npy', transform=transforms.Compose([To3DGrid(), ToTensor()]))
    
    print(dataset[0][0].shape, type(dataset[0][0]))
    print(dataset[0][1].shape, type(dataset[0][1]))
    print(dataset[0][2].shape, type(dataset[0][2]))
    
    print(dataset[1][0].shape, type(dataset[1][0]))
    print(dataset[1][1].shape, type(dataset[1][1]))
    print(dataset[1][2].shape, type(dataset[1][2]))
    
    
    
    

   