import torch
import torch.nn as nn
from torch.nn.functional import dropout
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from dataset import VOXELDataset, VOXELDataset2, ToTensor, To3DGrid
from model import MyNBVNetV2, MyNBVNetV3
from torch.autograd import Variable
import sys
import time

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

learning_rate = 2e-4
batch_size = 64
num_epochs = 500

name_of_model = ''

def eval(datapath):
    
    test_data = np.genfromtxt(datapath).reshape(1, 1, 32, 32, 32)
    test_data = torch.from_numpy(test_data).to(torch.float32)

    model = MyNBVNetV3()
    model = model.to(device)

    checkpoint = torch.load('./last.pth.tar',map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    print('EVALUATING')
    model.eval()
    grid = test_data.to(device)
    
    startTime = time.time()
    
    output = model(grid)

    endTime = time.time()
    print('run time is ' + str(endTime-startTime))
    np.savetxt('./run_time/'+name_of_model+'.txt',np.asarray([endTime-startTime]))
    
    output[output >= 0.5] = 1
    output[output < 0.5] = 0
    # print(output.shape)

    return output
    

if __name__ == '__main__':
    name_of_model = str(sys.argv[2])
    pred = eval('./data/'+name_of_model+'_voxel.txt')
    ans = []
    for i in range(pred.shape[1]):
        if pred[0][i] == 1:
            print(i)
            ans.append(i)
    np.savetxt('./log/'+name_of_model+'.txt',ans,fmt='%d')
