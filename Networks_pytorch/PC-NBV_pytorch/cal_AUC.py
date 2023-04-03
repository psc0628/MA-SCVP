import argparse
import array
import numpy as np
import os
from open3d import *
import matplotlib.pyplot as plt
import sys
import pdb


if __name__ == '__main__':

    data_type = 'test'

    ShapeNetv1_dir = '/home/zengrui/IROS/pcn/data/ShapeNetv1/'
    model_dir = '/home/zengrui/IROS/pcn/data/ShapeNetv1/' + data_type
    result_dir = '/home/zhanhz/PC-NBV-pytorch/pytorch_version/Simulation/Results/origin/'
    ex_times = 1
    scan_num = 10

    coverage_ave_our = np.zeros((16, 10))
    i = 0

    name_dict = {'02691156':'Airplane', '02933112':'Cabinet', '02958343':'Car', '03001627':'Chair',
                 '03636649':'Lamp', '04256520':'Sofa', '04379243':'Table', '04530566':'Vessel',
                 '02924116':'Bus', '02818832':'Bed', '02871439':'BookShelf','02828884':'Bench', 
                 '03467517':'Guitar', '03790512':'Motorbike', '04225987':'Skateboard', '03948459':'Pistol'}
    ave_AUC = 0

    ans = ""
    # for class_id in class_list:
    for class_id in name_dict:
        model_list = os.listdir(os.path.join(ShapeNetv1_dir, data_type, class_id))
        for ex_id in range(ex_times):
            for model_id in model_list:
                cov_cur = np.loadtxt(os.path.join(result_dir, str(ex_id), model_id + "_covrage.txt"))
                coverage_ave_our[i, :] += cov_cur[:scan_num] 

        coverage_ave_our[i, :] /= (len(model_list) * ex_times)
        AUC = (coverage_ave_our[i, :9].sum() + 0.5 * coverage_ave_our[i, 9]) / 10
        ave_AUC += AUC
        print(name_dict[class_id])
        print (AUC)
        i += 1

    ave_AUC = ave_AUC / 16
    print('average: ', ave_AUC)
