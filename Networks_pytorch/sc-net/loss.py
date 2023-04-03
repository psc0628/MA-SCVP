import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import VOXELDataset,ToTensor, To3DGrid
from torchvision import transforms

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class NBVLoss(nn.Module):
    def  __init__(self):
        super(NBVLoss, self).__init__()

        self.mse = nn.MSELoss()
        self.entropy = nn.BCELoss()
        # self.l1loss = nn.L1Loss()
        # self.sigmoid = nn.Sigmoid()

        self.lambda_for0 = 1
        self.lambda_for1 = 1
        self.lambda_l2   = 1
        self.lambda_cnt1 = 1

    def forward(self, predictions, target):
        # Euclidean distance
        # l2loss = self.mse(predictions, target)

        # loss_where_1
        # index_1 = torch.nonzero(target)
        loss_where_1 = 0
        loss_where_0 = 0
        # for index in index_1:
        #     i, j = index[0].item(), index[1].item() 
        #     loss_where_1 += self.entropy(predictions[i][j],  target[i][j])

        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if target[i][j] == 0:
                    loss_where_0 += self.entropy(predictions[i][j], target[i][j]).to(device)
                else:
                    loss_where_1 += self.entropy(predictions[i][j], target[i][j]).to(device)

        # cnt_target1 = torch.nonzero(target).shape[0]
        # cnt_pred1   = torch.nonzero(predictions[predictions > 0.5]).shape[0]

        # loss_cnt1 = torch.exp(torch.tensor(abs(cnt_target1-cnt_pred1)))
        # loss_cnt1 = torch.tensor(abs(cnt_target1-cnt_pred1)**2)
                    
        return (
            self.lambda_for1 * loss_where_1
            + self.lambda_for0 * loss_where_0
            # + self.lambda_l2 * l2loss
            # + self.lambda_cnt1 * loss_cnt1
        )



class NBVLoss2(nn.Module):
    def  __init__(self):
        super(NBVLoss2, self).__init__()

        self.mse = nn.MSELoss()
        self.entropy = nn.BCELoss()
        # self.l1loss = nn.L1Loss()
        # self.sigmoid = nn.Sigmoid()

        self.lambda_for0 = 1
        self.lambda_for1 = 2
        self.lambda_l2   = 1
        self.lambda_cnt1 = 1

    def forward(self, predictions, target):
        # Euclidean distance
        # l2loss = self.mse(predictions, target)

        # loss_where_1
        # index_1 = torch.nonzero(target)
        loss_where_1 = 0
        loss_where_0 = 0
        # for index in index_1:
        #     i, j = index[0].item(), index[1].item() 
        #     loss_where_1 += self.entropy(predictions[i][j],  target[i][j])

        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if target[i][j] == 0:
                    loss_where_0 += self.mse(predictions[i][j], target[i][j]).to(device)
                else:
                    loss_where_1 += self.mse(predictions[i][j], target[i][j]).to(device)

        # cnt_target1 = torch.nonzero(target).shape[0]
        # cnt_pred1   = torch.nonzero(predictions[predictions > 0.5]).shape[0]

        # # loss_cnt1 = torch.exp(torch.tensor(abs(cnt_target1-cnt_pred1)))
        # loss_cnt1 = torch.tensor(abs(cnt_target1-cnt_pred1)**2)
                    
        return (
            self.lambda_for1 * loss_where_1
            + self.lambda_for0 * loss_where_0
            # + self.lambda_l2 * l2loss
            # + self.lambda_cnt1 * loss_cnt1
        )




if __name__ == "__main__":
    test_dataset = VOXELDataset('../data/novel_test_data2/Armadillo', transform=transforms.Compose([To3DGrid(), ToTensor()]))
    loader  = DataLoader(test_dataset, batch_size=64)
    dataiter = iter(loader)
    data1 = dataiter.next()

    grid, label = data1
    print(label)

    cnt1 = torch.nonzero(label)
    print(cnt1)
    print(cnt1.shape[0])

