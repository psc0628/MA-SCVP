import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from dataset import SCVPDataset, ToTensor, To3DGrid
from model import SCVP
from loss import NBVLoss
import os
import config


def train_fn(loader, model, optimizer, criterion, scaler):
    print('------Training------')
    loop = tqdm(loader, leave=True)
    losses = []

    for _, data in enumerate(loop):
            grid = data[0]
            grid = grid.to(config.DEVICE)
            vss = data[1].unsqueeze(1)
            vss = vss.to(config.DEVICE)
            label = data[2]
            label = label.to(config.DEVICE)

            output = model(grid, vss)
            loss = criterion(output, label)

            losses.append(loss)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)
    


def check_accuracy(model, test_loader, test_case_num): 
    print('EVALUATING')
    model.eval()
    recall = 0
    percision = 0
    label_thresh = config.THRESHOLD_GAMMA
    for sample in test_loader:
        grid = sample[0].to(config.DEVICE)
        vss = sample[1].unsqueeze(1).to(config.DEVICE)
        label = sample[2].to(config.DEVICE)

        output = model(grid, vss)
        output[output >= label_thresh] = 1
        output[output < label_thresh] = 0
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

            recall += (correct1/cnt1)
            percision += (correct1 / (correct1 + wrong1 + 1e-6))
        
    recall /= test_case_num
    percision /= test_case_num
    print(f'test recall:{recall}, percision:{percision}')
    model.train()


def load_checkpoint(path, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print(f'loaded epoch: {checkpoint["epoch"]}')
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(model, optimizer, epoch, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        'epoch' : epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def main():
    vss_path = config.GRID_PATH.replace('grid', 'vss')
    label_path = config.GRID_PATH.replace('grid', 'label')
    dataset= SCVPDataset(config.GRID_PATH, vss_path, label_path, transform=transforms.Compose([To3DGrid(), ToTensor()]))

    batch_size = config.BATCH_SIZE
    learning_rate = config.LEARNING_RATE
    num_epochs = config.NUM_EPOCHS
    validation_split = config.VALIDATION_SPILT

    dataset_size = len(dataset)

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if config.SHUFFLE_DATASET :
        np.random.seed(config.RANDOM_SEED)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
        
    # train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
   
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    model = SCVP(net_type=config.NET_TYPE)
    model = model.to(config.DEVICE)

    if config.OPTIMIZER == 'Adam':
        print('Optimizer: Adam')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    else:
        print('Optimizer: SGD')
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    scaler = torch.cuda.amp.GradScaler()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    if config.LOSS == 'NBVLoss':
        print('Loss : NBVLoss')
        criterion = NBVLoss(lambde_for1=config.LOSS_LAMBDA)
    else:
        print('Loss : MSELoss')
        criterion = nn.MSELoss()
    criterion = criterion.to(config.DEVICE)

    if config.LOAD_MODEL:
        print('Loading saved model')
        load_checkpoint(config.LOAD_PATH, model, optimizer, learning_rate)
    else:
        print('Training new model')
    
    # all_losses = []
    for epoch in range(num_epochs):
        if epoch % 15 == 0 and epoch != 0:
            print('On test loader')
            check_accuracy(model, validation_loader, len(val_indices))

        print(f'epoch : {epoch+1}/{num_epochs}')
        train_fn(train_loader, model, optimizer, criterion, scaler)
        scheduler.step()

        if (epoch+1)%10==0:
            save_checkpoint(model, optimizer, epoch, config.SAVE_PATH)


if __name__ == "__main__":
    print(f'process id : {os.getpid()}')
    main()
    

