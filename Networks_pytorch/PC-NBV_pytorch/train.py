import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import datetime
import csv

from dataset.dataset import ShapeNet, ShapeNet2, ShapeNet3
from models.pc_nbv import AutoEncoder

def train(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    Loss = nn.MSELoss(reduction='sum')

    # train_dataset = ShapeNet(NBV_dir=args.partial_root, gt_dir=args.gt_root, split='train')
    # val_dataset = ShapeNet(NBV_dir=args.partial_root, gt_dir=args.gt_root, split='test')
    train_dataset = ShapeNet3("./dataset/accumulate_pointclouds_train.pt",
                              "./dataset/gt_point_clouds_train.pt",
                              "./dataset/view_statess_train.pt",
                              "./dataset/target_values_train.pt")
    
    val_dataset = ShapeNet3("./dataset/accumulate_pointclouds_test.pt",
                              "./dataset/gt_point_clouds_test.pt",
                              "./dataset/view_statess_test.pt",
                              "./dataset/target_values_test.pt")
    # NBV_dir = '/home/huhao/code_python/data/PCNBV_lable_data'
    # train_dataset = ShapeNet2(NBV_dir, '/home/huhao/code_python/PC-NBV_pytorch/Name_of_Trainning_Objects.txt')
    # val_dataset = ShapeNet2(NBV_dir, '/home/huhao/code_python/PC-NBV_pytorch/test.txt')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    network = AutoEncoder(views=32)
    if args.model is not None:
        print('Loaded trained model from {}.'.format(args.model))
        network.load_state_dict(torch.load(args.model))
    else:
        print('Begin training new model.')
    network.to(DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    max_iter = int(len(train_dataset) / args.batch_size + 0.5)
    minimum_loss = 1e4
    best_epoch = 0

    with open(os.path.join(args.log_dir, 'args.txt'), 'w') as log:
        for arg in sorted(vars(args)):
            log.write(arg + ': ' + str(getattr(args, arg)) + '\n')     # log of arguments

    csv_file = open(os.path.join(args.log_dir, 'loss.csv'), 'a+')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train loss', 'train loss_eval', 'valid loss', 'valid loss_eval'])
    
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        # training
        network.train()
        total_loss, iter_count = 0, 0
        epoch_start = time.time()
        for i, data in enumerate(train_dataloader, 1):
            accumulate_pointcloud, gt_point_cloud, view_states, target_value = data
            accumulate_pointcloud = accumulate_pointcloud.to(DEVICE)
            gt_point_cloud = gt_point_cloud.to(DEVICE)
            view_states = view_states.to(DEVICE)
            target_value = target_value.to(DEVICE)
            
            accumulate_pointcloud = accumulate_pointcloud.permute(0, 2, 1)
         
            optimizer.zero_grad()

            _, pred_value = network(accumulate_pointcloud, view_states)

            loss_var_encoder = sum([Loss(param, torch.zeros_like(param).to(DEVICE)) for param in network.encoder.parameters()])* args.alpha
            loss_var_decoder = sum([Loss(param, torch.zeros_like(param).to(DEVICE)) for param in network.decoder.parameters()])* args.alpha
            loss_eval = Loss(pred_value, target_value)

            loss = loss_var_encoder  + loss_var_decoder  + loss_eval
            loss.backward()
            optimizer.step()
            
            iter_count += 1
            total_loss += loss.item()
            
            if i % 100 == 0:
                print("Training epoch {}/{}, iteration {}/{}: loss is {:.4f} {:.4f} {:.4f} {:.4f}".format(epoch, args.epochs, i, max_iter, loss.item(), loss_var_encoder.item(), loss_var_decoder.item(), loss_eval.item()))
        scheduler.step()
        epoch_time = time.time() - epoch_start
        print("\033[31mTraining epoch {}/{}: avg loss = {}  time per epoch: {:.2f}s\033[0m".format(epoch, args.epochs, total_loss / iter_count, epoch_time))

        # evaluation
        network.eval()
        with torch.no_grad():
            total_loss, iter_count, total_eval = 0, 0, 0
            for i, data in enumerate(val_dataloader, 1):
                accumulate_pointcloud, gt_point_cloud, view_states, target_value = data
                accumulate_pointcloud = accumulate_pointcloud.to(DEVICE)
                gt_point_cloud = gt_point_cloud.to(DEVICE)
                view_states = view_states.to(DEVICE)
                target_value = target_value.to(DEVICE)

                accumulate_pointcloud = accumulate_pointcloud.permute(0, 2, 1)
                
                _, pred_value = network(accumulate_pointcloud, view_states)

                loss_var_encoder = sum([Loss(param, torch.zeros_like(param).to(DEVICE)) for param in network.encoder.parameters()])
                loss_var_decoder = sum([Loss(param, torch.zeros_like(param).to(DEVICE)) for param in network.decoder.parameters()])
                valid_loss_eval = Loss(pred_value, target_value)

                valid_loss = loss_var_encoder * args.alpha + loss_var_decoder * args.alpha + valid_loss_eval
                total_loss += valid_loss.item()
                total_eval += valid_loss_eval.item()
                iter_count += 1

            mean_loss = total_loss / iter_count
            mean_loss_eval = total_eval / iter_count
            print("\033[31mValidation epoch {}/{}, loss: {:.4f}, {:.4f}\033[0m".format(epoch, args.epochs, mean_loss, mean_loss_eval))
            
            csv_writer.writerow([epoch, loss.item(), loss_eval.item(), mean_loss , mean_loss_eval])
            if epoch % 5 == 0:
                torch.save(network.state_dict(), args.log_dir + '/{}.pth'.format(epoch))
            torch.save(network.state_dict(), args.log_dir + 'last.pth')
            # records the best model and epoch
            if mean_loss < minimum_loss:
                best_epoch = epoch
                minimum_loss = mean_loss
                torch.save(network.state_dict(), args.log_dir + '/lowest_loss.pth')

        print("\033[31mBest model (lowest loss) in epoch {}\033[0m".format(best_epoch))
    print('Total time', datetime.timedelta(seconds=time.time() - train_start))
    csv_file.close()

if __name__ == '__main__':        
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial_root', type=str, default="/home/huhao/ShapeNetCore.v1/PC_results/ABC_patch")
    parser.add_argument('--gt_root', type=str, default="/home/huhao/IROS/pcn/data/ShapeNetv1/")
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--num_input', type=int, default=512)
    parser.add_argument('--num_gt', type=int, default=16384)
    parser.add_argument('--views', type=int, default=33)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    train(args)