from genericpath import exists
from operator import index
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as transforms
import numpy as np
from model import MyNBVNetV3
import pickle

class VOXELDataset(Dataset):
    def __init__(self, path, transform=None, processed_data=True, save_path='./process_data.dat'):
        self.path = path
        self.processed_data = processed_data
        self.save_path = save_path
        self.transform = transform
        
        # print(len(self.grid_path))
        if self.processed_data:
            self.grid_path = []
            self.label_path = []

            for root, _, files in os.walk(self.path):
                if len(files) == 2:
                    grid_path = os.path.join(root, 'grid.txt')
                    label_path = os.path.join(root, 'view_ids.txt')
                    # print(grid_path, label_path)
                    self.grid_path.append(grid_path)
                    self.label_path.append(label_path)
            
            if not os.path.exists(self.save_path):
                print('processing data, only running in the first epoch')
                self.list_of_grid = [None] * len(self.grid_path)
                self.list_of_label = [None] * len(self.label_path)

                for index in range(len(self.grid_path)):
                    grid_path = self.grid_path[index]
                    label_path = self.label_path[index]
                    # print(grid_path,label_path)
                    grid = np.genfromtxt(grid_path)[:, [-1]]
                    # print(grid.shape)
                    label_list = np.genfromtxt(label_path, dtype=np.int32).tolist()
                    label = np.zeros(64)
                    label[label_list] = 1

                    if self.transform:
                        sample = {'grid': grid, 'label': label}
                        sample = self.transform(sample)

                    self.list_of_grid[index] = sample['grid']
                    self.list_of_label[index] = sample['label']

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_grid, self.list_of_label], f)
            else:
                print('loading data from processed data')
                with open(self.save_path, 'rb') as f:
                    self.list_of_grid, self.list_of_label = pickle.load(f)
        # else:


    def __len__(self):
        return len(self.grid_path)

    def __getitem__(self, index):
        return self.list_of_grid[index], self.list_of_label[index]


class VOXELDataset2(Dataset):
    def __init__(self, grid_path, label_path, transform=None):
        self.transform  = transform
        self.grid = np.load(grid_path)
        self.label = np.load(label_path)

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, index):
        sample = {'grid': self.grid[index], 'label': self.label[index]}
        if self.transform:
            sample = self.transform(sample)

        return sample['grid'], sample['label']


class ToTensor(object):
    def __call__(self, sample):
        grid = sample['grid']
        label = sample['label']

        return {'grid': torch.from_numpy(grid).to(torch.float32),
                'label': torch.from_numpy(label).to(torch.float32)}

class To3DGrid(object):
    def __call__(self, sample):
        grid = sample['grid']
        label = sample['label']

        # grid = np.reshape(grid, (1, 40, 40, 40))        # ?????
        grid = np.reshape(grid, (1, 32, 32, 32))

        return {'grid': grid,
                'label': label}

def test():
    for root, _, files in os.walk('../data/SC_label_data'):
        if len(files) == 2:
            label_path = os.path.join(root, files[1])
            print(label_path)
            label = torch.zeros(64)
            label_list = np.genfromtxt(label_path, dtype=np.int64).tolist()
            label[label_list] = 1
            print(label)
    
            
if __name__ == "__main__":
    # dataset = VOXELDataset('../data/02747177', transform=transforms.Compose([To3DGrid(),ToTensor()]), processed_data=True, save_path='02747177.dat')
    dataset = VOXELDataset2('../grids.npy', '../labels.npy', transform=transforms.Compose([To3DGrid(), ToTensor()]))
    
    
    

   