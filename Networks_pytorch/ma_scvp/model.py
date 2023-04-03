import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)

class SCVPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual):
        super(SCVPBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual

        self.conv1 = self._make_conv()
        self.conv2 = self._make_conv() 
        self.relu = nn.LeakyReLU(0.1)

    def _make_conv(self):
        layers = []

        for _ in range(4):
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


class SCVP(nn.Module):
    def __init__(self, residual=True, net_type="SCVP"):
        super(SCVP, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool  = nn.MaxPool3d(2, 2)
        self.residual = residual
        self.net_type = net_type


        self.block1 = SCVPBlock(33 if self.net_type == "MASCVP" else 32, 64, residual=self.residual)
        self.down1  = nn.Conv3d(64, 64, 2, 2)

        self.block2 = SCVPBlock(64, 128, residual=self.residual)
        self.down2  = nn.Conv3d(128, 128, 2, 2)

        self.block3 = SCVPBlock(128, 256, residual=self.residual)
        self.down3  = nn.Conv3d(256, 256, 2, 2)
        
        
        self.fc1 = nn.Linear((481 if self.net_type == "MASCVP" else 480) * 2 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)

        self.relu = nn.LeakyReLU(0.1)
        self.sig  = nn.Sigmoid()

    def forward(self, x, vs):
        b = x.shape[0]
        x1 = self.pool(self.conv1(x))       # 16 ^ 3 

        if self.net_type == "MASCVP":
            # (32, 16, 16, 16) --> (32, 4096) + (1, 32 * 128) == (33, 4096) --> (33, 16, 16, 16)
            vs = vs.repeat(1, 1, 4096//32)
            x1 = torch.cat([x1.reshape(b, 32, -1), vs], dim=1).reshape(b, 33, 16, 16, 16)

        x2 = self.down1(self.block1(x1))    # 8 ^ 3
        x3 = self.down2(self.block2(x2))    # 4 ^ 3
        x4 = self.down3(self.block3(x3))

        x2 = torch.cat([x2, self.pool(x1)], dim=1)
        x3 = torch.cat([x3, self.pool(x2)], dim=1)
        x4 = torch.cat([x4, self.pool(x3)], dim=1)   
        x4 = x4.view(x4.shape[0],  -1)
        
        x4 = F.dropout(self.relu(self.fc1(x4)), training=self.training)
        x4 = F.dropout(self.relu(self.fc2(x4)), training=self.training)
        x4 = F.dropout(self.relu(self.fc3(x4)), training=self.training)
        x4 = F.dropout(self.relu(self.fc4(x4)), training=self.training)
        x4 = self.sig(self.fc5(x4))
        return x4


if __name__ == "__main__":
    model = SCVP(net_type='MASCVP').to('cuda:1')
    x = torch.randn(128, 1, 32, 32, 32).to('cuda:1')
    vs = torch.randn(128, 1, 32).to('cuda:1')
    print(model(x, vs).shape)