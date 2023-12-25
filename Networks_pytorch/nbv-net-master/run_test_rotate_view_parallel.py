import sys
import os
from torch.nn.functional import dropout
from nbvnet import NBV_Net
import torch
import time
import numpy as np

network_file = './pth/nbvnet_longtail32.pth'
# network_file = './pth/nbvnet_nbv32.pth'

name_of_model = []
user_input = input('input object name:')
while user_input != '-1':
    name_of_model.append(user_input)
    user_input = input('input object name:')

rotate_ids = []
rotate_ids.append(0)
# rotate_ids.append(1)
rotate_ids.append(2)
rotate_ids.append(3)
rotate_ids.append(4)
rotate_ids.append(5)
# rotate_ids.append(6)
# rotate_ids.append(7)

first_view_ids = []
first_view_ids.append(0)
first_view_ids.append(1)
first_view_ids.append(2)
# first_view_ids.append(3)
first_view_ids.append(4)
# first_view_ids.append(5)
first_view_ids.append(6)
first_view_ids.append(7)
first_view_ids.append(8)
# first_view_ids.append(9)
# first_view_ids.append(10)
# first_view_ids.append(11)
first_view_ids.append(12)
# first_view_ids.append(13)
first_view_ids.append(14)
# first_view_ids.append(15)
first_view_ids.append(16)
# first_view_ids.append(17)
first_view_ids.append(18)
first_view_ids.append(19)
first_view_ids.append(20)
first_view_ids.append(21)
first_view_ids.append(22)
# first_view_ids.append(23)
first_view_ids.append(24)
first_view_ids.append(25)
# first_view_ids.append(26)
# first_view_ids.append(27)
first_view_ids.append(28)
first_view_ids.append(29)
first_view_ids.append(30)
# first_view_ids.append(31)

model = ''
rotate_id = -1
view_id = -1

max_iteration = -1
iteration = -1

def get_single_view_point(path, net):
    grid = np.genfromtxt(path).reshape(1, 1, 32, 32, 32)
    grid = torch.tensor(grid, dtype=torch.float32)
    startTime = time.time()
    ids = net(grid)
    endTime = time.time()
    print('run time is ' + str(endTime-startTime))
    np.savetxt('./run_time/'+model+'_r'+str(rotate_id)+'_v'+str(view_id)+'_'+str(iteration)+'.txt',np.asarray([endTime-startTime]))
    return ids

print(network_file)
net = NBV_Net(dropout_prob=0)
checkpoint = torch.load(network_file,map_location = torch.device('cpu'))
net.load_state_dict(checkpoint['net'])

for model in name_of_model:
    print('testing '+ model)
    for rotate_id in rotate_ids:
        for view_id in first_view_ids:
            #if os.path.isfile('E:\\MA-SCVP\\Longtail\\32\\'+model+'_r'+str(rotate_id)+'_v'+str(view_id)+'_m7/all_needed_views.txt')==False:
            #    max_iteration = 20
            #else:
            #    with open('E:\\MA-SCVP\\Longtail\\32\\'+model+'_r'+str(rotate_id)+'_v'+str(view_id)+'_m7/all_needed_views.txt', 'r') as f:
            #        for line in f:
            #            max_iteration = int(line.strip('\n'))
            max_iteration = 20
            print('max_iteration is '+str(max_iteration))
            iteration = 0
            while iteration<max_iteration:
                print('./data/'+model+'_r'+str(rotate_id)+'_v'+str(view_id)+'_'+str(iteration)+'.txt')
                while os.path.isfile('./data/'+model+'_r'+str(rotate_id)+'_v'+str(view_id)+'_'+str(iteration)+'.txt')==False:
                    pass
                time.sleep(1)
                ids = get_single_view_point('./data/'+model+'_r'+str(rotate_id)+'_v'+str(view_id)+'_'+str(iteration)+'.txt', net)
                ids = ids.argmax(dim=1)
                print('next view is ' + str(ids))
                np.savetxt('./log/'+model+'_r'+str(rotate_id)+'_v'+str(view_id)+'_'+str(iteration)+'.txt',ids,fmt='%d')
                f = open('./log/'+model+'_r'+str(rotate_id)+'_v'+str(view_id)+'_'+str(iteration)+'_ready.txt','a')
                f.close()
                iteration += 1
            print('testing '+ model+'_r'+str(rotate_id) + '_v'+ str(view_id) + ' over.')
