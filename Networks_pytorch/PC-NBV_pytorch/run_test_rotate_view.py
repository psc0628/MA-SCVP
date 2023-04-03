import sys
import os
from torch.nn.functional import dropout
from models.pc_nbv import AutoEncoder
import torch
import time
import numpy as np

network_file = './pth/PCNBV_LongTailSample_32last.pth'

name_of_model = []
with open('./all_name.TXT', 'r') as f:
    for line in f:
        name_of_model.append(line.strip('\n'))

rotate_ids = []
#rotate_ids.append(0)
#rotate_ids.append(1)
#rotate_ids.append(2)
rotate_ids.append(3)
#rotate_ids.append(4)
#rotate_ids.append(5)
rotate_ids.append(6)
#rotate_ids.append(7)

first_view_ids = []
first_view_ids.append(0)
first_view_ids.append(2)
first_view_ids.append(4)
first_view_ids.append(14)
first_view_ids.append(27)
#for i in range(0,32):
#    first_view_ids.append(i)

model = ''
rotate_id = -1
view_id = -1

max_iteration = -1
iteration = -1

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def infer_once(accumulate_pointcloud, view_states, network):
    network.eval()
    network.to(DEVICE)
    accumulate_pointcloud = accumulate_pointcloud.to(DEVICE)
    view_states = view_states.to(DEVICE)
    accumulate_pointcloud = accumulate_pointcloud.permute(0, 2, 1)
    startTime = time.time()
    _, pred_value = network(accumulate_pointcloud, view_states)
    endTime = time.time()
    print('run time is ' + str(endTime-startTime))
    np.savetxt('./run_time/'+model+'_r'+str(rotate_id)+'_v'+str(view_id)+'_'+str(iteration)+'.txt',np.asarray([endTime-startTime]))
    return pred_value

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

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
            max_iteration = 1
            print('max_iteration is '+str(max_iteration))
            iteration = 0
            while iteration<max_iteration:
                print('./data/'+model+'_r'+str(rotate_id)+'_v'+str(view_id)+'_vs'+str(iteration)+'.txt')
                while os.path.isfile('./data/'+model+'_r'+str(rotate_id)+'_v'+str(view_id)+'_vs'+str(iteration)+'.txt')==False:
                    pass
                time.sleep(1)
                model_param_path = network_file
                cloud_name = './data/'+model+'_r'+str(rotate_id)+'_v'+str(view_id)+'_pc'+str(iteration)+'.txt'
                score_name = './data/'+model+'_r'+str(rotate_id)+'_v'+str(view_id)+'_vs'+str(iteration)+'.txt'
                network = AutoEncoder(views=32)
                network.load_state_dict(torch.load(model_param_path,map_location = torch.device('cpu')))
                accumulate_pointcloud = np.genfromtxt(cloud_name, dtype=np.float32).reshape(-1, 3)
                #accumulate_pointcloud = resample_pcd(accumulate_pointcloud, 1024)
                view_state = np.genfromtxt(score_name, dtype=np.float32)
                accumulate_pointcloud = torch.from_numpy(accumulate_pointcloud).unsqueeze(0)
                view_state = torch.from_numpy(view_state).unsqueeze(0)
                #print(accumulate_pointcloud.shape, view_state.shape)
                ids = torch.argmax(infer_once(accumulate_pointcloud, view_state, network), dim=1)
                print('next view is ' + str(ids))
                np.savetxt('./log/'+model+'_r'+str(rotate_id)+'_v'+str(view_id)+'_'+str(iteration)+'.txt',ids,fmt='%d')
                f = open('./log/ready.txt','a')
                f.close()
                iteration += 1
            print('testing '+ model+'_r'+str(rotate_id) + '_v'+ str(view_id) + ' over.')
