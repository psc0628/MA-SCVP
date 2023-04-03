import torch.nn as nn
import config

class NBVLoss(nn.Module):
    def  __init__(self, lambde_for1):
        super(NBVLoss, self).__init__()

        self.mse = nn.MSELoss()
        self.entropy = nn.BCELoss()

        self.lambda_for0 = 1
        self.lambda_for1 = lambde_for1
        self.lambda_l2   = 1
        self.lambda_cnt1 = 1

    def forward(self, predictions, target):
        loss_where_1 = 0
        loss_where_0 = 0

        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if target[i][j] == 0:
                    loss_where_0 += self.entropy(predictions[i][j], target[i][j]).to(config.DEVICE)
                else:
                    loss_where_1 += self.entropy(predictions[i][j], target[i][j]).to(config.DEVICE)
        return (
            self.lambda_for1 * loss_where_1
            + self.lambda_for0 * loss_where_0
        )



