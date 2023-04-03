import os
import numpy as np
import scipy.io as sio
from open3d import *
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import random
from distance.chamfer_distance import ChamferDistanceFunction
import time
import pdb

if __name__ == '__main__':
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # path
    model_dir = "/home/zengrui/IROS/pcn/data/ShapeNetv1/test"
    pc_dir = "/home/zengrui/IROS/pcn/PC_results/ShapeNetv1/test/pcd"
    ShapeNetv1_dir = '/home/zengrui/IROS/pcn/data/ShapeNetv1/'
    result_dir = '/home/zhanhz/PC-NBV-pytorch/pytorch_version/Simulation/Results/'
    scan_num = 10
    ex_num = 1

    class_list = os.listdir(model_dir)
    for class_id in class_list:
        model_list = os.listdir(os.path.join(model_dir, class_id))

        for ex_id in range(ex_num):
            for model_id in model_list:
                cov_list = np.zeros(scan_num)
                gt_points = sio.loadmat(os.path.join(model_dir, class_id, model_id, 'model.mat'))
                gt_points = np.array(gt_points['points'])
                batch_gt = gt_points[np.newaxis, :, :].astype(np.float32) 
                gt_tensor = torch.tensor(batch_gt).to(DEVICE)     

                partial = np.zeros((0, 3))  
                model_view_list_path  =os.path.join(result_dir, str(ex_id), model_id + "_selected_views.txt")
                model_view_list = np.loadtxt(model_view_list_path, dtype=np.float32)    

                for scan_id in range(scan_num):
                    partial_path = os.path.join(pc_dir, model_id, str(int(model_view_list[scan_id])) + ".pcd")
                    cur_partial = open3d.io.read_point_cloud(partial_path)
                    cur_partial = np.array(cur_partial.points)
                    partial = np.append(partial, cur_partial, axis=0)
                            
                    batch_part = partial[np.newaxis, :, :].astype(np.float32)
                    part_tensor = torch.tensor(batch_part).to(DEVICE)
                    dist1, dist2 =  ChamferDistanceFunction.apply(part_tensor, gt_tensor)  
                    dist2 = dist2.cpu().numpy()      

                    dis_flag_new = dist2 < 0.00005
                    cover_sum = np.sum(dis_flag_new == True)
                    cover = cover_sum / dis_flag_new.shape[1]       

                    cov_list[scan_id] = cover       

                print(model_id)
                np.savetxt(os.path.join(result_dir, str(ex_id), model_id + "_covrage.txt"), cov_list)
