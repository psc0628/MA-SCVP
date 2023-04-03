import os
from open3d import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from dataset.dataset import ShapeNet
from models.pc_nbv import AutoEncoder
from utils import resample_pcd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

viewspace_path = '/home/zengrui/IROS/pcn/render/viewspace_shapenet_33.txt'
result_dir = '/home/zhanhz/PC-NBV-pytorch/pytorch_version/Simulation/Results/'
start_view_dir = '/home/zhanhz/PC-NBV-pytorch/pytorch_version/Simulation/shapenet_start/'
PC_cloud_dir = '/home/zengrui/IROS/pcn/PC_results/ShapeNetv1/test/pcd'
model_list_path = '/home/zengrui/IROS/pcn/Simulation/Results/ShapeNet_2_12/test.txt'
checkpoint = '/home/zhanhz/PC-NBV-pytorch/pytorch_version/log/lowest_loss.pth'

ex_num = 1
scan_num = 10  

viewspace = np.loadtxt(viewspace_path)
with open(os.path.join(model_list_path)) as file:
    model_list = [line.strip() for line in file]

network = AutoEncoder()
print('Loaded trained model from {}.'.format(checkpoint))
network.load_state_dict(torch.load(checkpoint))
network.to(DEVICE)
network.eval()

time_count_num = 0
time_all = 0

for ex_i in range(ex_num):

    start_view = np.loadtxt(os.path.join(start_view_dir, "test_init_" + str(ex_i) + ".txt"))

    cur_ressult_dir = result_dir + str(ex_i)
    if not os.path.exists(cur_ressult_dir):
        os.makedirs(cur_ressult_dir)

    for i in range(len(model_list)):

        start_time = time.time()
        # model_id = str(int(model_list[i]))
        model_id = model_list[i]
        cur_view = int(start_view[i])
        view_state = np.zeros(viewspace.shape[0], dtype=np.int) 

        scan_pc = np.zeros((1, 3))
        scan_view_list = np.zeros(scan_num)

        for scan_id in range(scan_num):
            time_count_num += 1
            view_state[cur_view] = 1
            scan_view_list[scan_id] = cur_view  

            partial_path = os.path.join(PC_cloud_dir, model_id, str(cur_view) + ".pcd")    

            partial = open3d.io.read_point_cloud(partial_path)
            partial = np.array(partial.points)  

            scan_pc = np.append(scan_pc, partial, axis=0)
            partial = resample_pcd(scan_pc, 1024)

            partial_tensor = torch.tensor(partial[np.newaxis, ...].astype(np.float32)).permute(0, 2, 1).to(DEVICE)
            view_state_tensor = torch.tensor(view_state.astype(np.float32))[np.newaxis, ...].to(DEVICE)


            _, eval_value = network(partial_tensor, view_state_tensor)
            eval_value = eval_value[0].cpu().detach().numpy()      

            new_view = np.argmax(eval_value, axis = 0)

            cur_view = new_view  

        np.savetxt(os.path.join(cur_ressult_dir, model_id + "_selected_views.txt"), scan_view_list)

        end_time = time.time()
        print("model:" + model_id + " spend time:" + str(end_time - start_time))
        time_all += end_time - start_time

print("ave time per scan:" + str(time_all / time_count_num))