import torch
import torch.nn as nn
from torch.nn.functional import dropout
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from dataset import VOXELDataset, VOXELDataset2, ToTensor, To3DGrid
from model import MyNBVNetV2, MyNBVNetV3
from torch.autograd import Variable
from loss import NBVLoss,NBVLoss2
import argparse

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def check_model_acc(model, test_loader, test_case_num):
    print('EVALUATING')
    model.eval()
    accuracy_exp = 0
    recall = 0
    percision = 0
    accuracy = []
    for sample in test_loader:
        grid = sample[0].to(device)
        label = sample[1].to(device)

        output = model(grid)
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        for i in range(label.shape[0]):
            correct1 = 0
            wrong1 = 0
            cnt1 = 0
            for j in range(32):
                if label[i][j] == 1 and output[i][j] == 1:
                    correct1 += 1
                    cnt1 += 1
                elif label[i][j] == 1 and output[i][j] == 0:
                    cnt1 += 1
                elif label[i][j] == 0 and output[i][j] == 1:
                    wrong1 += 1
            # print(cnt1 - correct1)

            correct_exp = (output[i]==label[i]).sum().item()
            accuracy_exp += 1 / np.exp(64-correct_exp)
            recall += (correct1/cnt1)
            percision += (correct1 / (correct1 + wrong1 + 1e-6))
            # print(64-correct)
        correct = (output == label).sum().item()
        acc = correct / (output.shape[0] * output.shape[1])
        accuracy.append(acc)
        
    accuracy_exp /= test_case_num
    recall /= test_case_num
    percision /= test_case_num
    mean_acc = sum(accuracy) / len(accuracy)
    print(f'test accuracy_exp:{accuracy_exp}, recall:{recall}, percision:{percision}, accuracy:{mean_acc}')
    model.train()

if __name__ == "__main__":
    model = MyNBVNetV3()
    model = model.to(device)
    dataset = VOXELDataset2('../grids.npy', '../labels.npy', transform=transforms.Compose([To3DGrid(), ToTensor()]))
    test_loader = DataLoader(dataset, batch_size=32)
    # print(len(dataset))
    # print(len(test_loader))
    checkpoint = torch.load('my_checkpoint_nbvnetv3_32viewpoints_300.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    # print(checkpoint['epoch'])
    check_model_acc(model, test_loader, len(dataset))