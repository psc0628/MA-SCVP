import sys
import os
from torch.nn.functional import dropout
from nbvnet import NBV_Net
import torch
import numpy as np

def get_single_view_point(path):
    net = NBV_Net(dropout_prob=0)
    checkpoint = torch.load('my_checkpoint.pth.tar',map_location = torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    grid = np.genfromtxt(path).reshape(1, 1, 32, 32, 32)
    grid = torch.tensor(grid, dtype=torch.float32)
    ids = net(grid)
    return ids


name_of_model = []
with open('./all_name.TXT', 'r') as f:
    for line in f:
        name_of_model.append(line.strip('\n'))

max_iteration = 11

for model in name_of_model:
    print('testing '+ model)
    iteration = 0
    while iteration<max_iteration:
        print('./data/'+model+'_'+str(iteration)+'.txt')
        while os.path.isfile('./data/'+model+'_'+str(iteration)+'.txt')==False:
            pass
        ids = get_single_view_point('./data/'+model+'_'+str(iteration)+'.txt')
        ids = ids.argmax(dim=1)
        np.savetxt('./log/'+model+'_'+str(iteration)+'.txt',ids,fmt='%d')
        f = open('./log/ready.txt','a')
        f.close()
        iteration += 1
    print('testing '+ model + ' over.')
