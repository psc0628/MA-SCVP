import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class RVPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual):
        super(RVPBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual

        self.conv1 = self._make_conv()
        self.conv2 = self._make_conv() 
        self.relu = nn.ReLU()

    def _make_conv(self):
        layers = []

        for i in range(2):
            layers += [CNNBlock(in_channels=self.in_channels, out_channels=self.out_channels)]
            self.in_channels = self.out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        identity = x1
        x2 = self.conv2(x1)

        if self.residual:
            x2 += identity

        return self.relu(x2)


class MyNBVNetV2(nn.Module):

    def __init__(self, residual=True):
        super(MyNBVNetV2, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)       
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)                  
        self.residual = residual

        self.block1 = RVPBlock(32, 64, residual=self.residual)
        self.block2 = RVPBlock(64, 128, residual=self.residual)
        # self.block3 = RVPBlock(128, 256, residual=self.residual)

        self.fc1 = nn.Linear(128*5*5*5, 1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)

        self.relu = nn.ReLU()
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        # x3 = self.pool(self.block3(x))

        x = x.view(x.shape[0],  -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sig(self.fc5(x))

        return x


class MyNBVNetV3(nn.Module):

    def __init__(self, residual=True, dropout_prob=0.5):
        super(MyNBVNetV3, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)       
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)                  
        self.residual = residual

        self.block1 = RVPBlock(32, 64, residual=self.residual)
        self.block2 = RVPBlock(64, 128, residual=self.residual)
        self.block3 = RVPBlock(128, 256, residual=self.residual)
        self.block4 = RVPBlock(256, 512, residual=self.residual)
        
        self.fc1 = nn.Linear(32 + 64 + 128 + 256 + 512, 512)
        self.fc1_drop = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(512, 256)
        self.fc2_drop = nn.Dropout(dropout_prob)
    
        self.fc3 = nn.Linear(256, 128)
        self.fc3_drop = nn.Dropout(dropout_prob)

        self.fc4 = nn.Linear(128, 64)
        self.fc4_drop = nn.Dropout(dropout_prob)

        self.fc5 = nn.Linear(64, 32)

        self.relu = nn.ReLU()
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        x1 = self.pool(self.conv1(x))
        x2 = self.pool(self.block1(x1))
        x3 = self.pool(self.block2(x2))
        x4 = self.pool(self.block3(x3))
        x5 = self.pool(self.block4(x4))

        x2 = torch.cat([x2, self.pool(x1)], dim=1)
        x3 = torch.cat([x3, self.pool(x2)], dim=1)
        x4 = torch.cat([x4, self.pool(x3)], dim=1)
        x5 = torch.cat([x5, self.pool(x4)], dim=1)

        x5 = x5.view(x5.shape[0],  -1)

        x5 = self.fc1_drop(self.relu(self.fc1(x5)))
        x5 = self.fc2_drop(self.relu(self.fc2(x5)))
        x5 = self.fc3_drop(self.relu(self.fc3(x5)))
        x5 = self.fc4_drop(self.relu(self.fc4(x5)))
        x5 = self.sig(self.fc5(x5))
        # x = self.sig(self.fc5(x))

        return x5



if __name__ == "__main__":
    model = MyNBVNetV3().to('cuda:0')
    # block = RVPBlock(64, 128, residual=True)
    x = torch.randn(64, 1, 32, 32, 32).to('cuda:0')
    print(model(x).shape)
    # # print(model) 