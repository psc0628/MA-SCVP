import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pdb

if __name__ == '__main__':
    
    
    # path
    model_dir = '/home/zengrui/IROS/pcn/data/ShapeNetv1/test'
    class_list_path = "/home/zengrui/IROS/pcn/data/ShapeNetv1/test_all_class.txt"
    result_dir = '/home/zhanhz/PC-NBV-pytorch/pytorch_version/Simulation/Results'
    model_sum = 800
    scan_num = 10

    with open(os.path.join(class_list_path)) as file:
        class_list = [line.strip() for line in file]

    coverage_ave_torch = np.zeros(10)
    ex_times = 1
    for class_id in class_list:
        model_list = os.listdir(os.path.join(model_dir, str(class_id)))
        for ex_id in range(ex_times):
            for model_id in model_list:
                model_id = str(model_id)       
                cov_cur = np.loadtxt(os.path.join(result_dir, 'torch', str(ex_id), model_id + "_covrage.txt"))
                coverage_ave_torch += cov_cur 
    coverage_ave_torch /= (model_sum * ex_times)
    print(coverage_ave_torch)

    # plt.figure()
    # plt.subplot(1, 1, 1)

    coverage_ave_tf = np.zeros(10)
    ex_times = 1
    for class_id in class_list:
        model_list = os.listdir(os.path.join(model_dir, str(class_id)))
        for ex_id in range(ex_times):
            for model_id in model_list:
                model_id = str(model_id)   
                cov_cur = np.loadtxt(os.path.join(result_dir, 'origin', str(ex_id), model_id + "_covrage.txt"))
                coverage_ave_tf += cov_cur[:10]
    coverage_ave_tf /= (model_sum * ex_times)
    print(coverage_ave_tf)


    plt.figure()
    plt.subplot(1, 1, 1)

    fsize = 18 #font size
    
    # draw plot
    x = np.arange(1, 11)

    plt.plot(x, coverage_ave_torch, marker = '^', label = 'torch', color = 'red')
    plt.plot(x, coverage_ave_tf, marker = '^', label = 'tf', color = 'green')

    plt.xlabel('Number of rounds', fontsize=fsize)
    plt.ylabel('Surface coverage (Average)', fontsize=fsize)
    plt.title('Testing Dataset', fontsize=fsize)
    plt.legend(fontsize=fsize)
    plt.savefig('shapenet_compare_test')
    plt.show()

