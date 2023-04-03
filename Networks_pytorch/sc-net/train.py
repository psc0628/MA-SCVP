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
import os

shuffle_dataset = True
random_seed = 42
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('training')
    # parser.add_argument('--use_gpu', action='store_true', default=True, help='use cpu mode')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training[default: 64]')
    parser.add_argument('--model', default='NBVNet', help='model name [default: NBVNet]')
    parser.add_argument('--epochs', default=300, type=int, help='number of epoch in training [default: 300]')
    parser.add_argument('--learning_rate', default=2e-4, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training[default: Adam]')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--validation_split', default=0.2, type=float, help='split rate for validation data[default: 0.2]')
    parser.add_argument('--loss', type=str, default='NBVLoss', help='specify the loss for the model[defaul: NBVLoss]')
    parser.add_argument('--load_model', action='store_true', default=False, help='load stored model parameters for traing[default: False]')
    parser.add_argument('--model_path', type=str, default='./LM1_my_checkpoint_forevery10epochs.pth.tar', help='path to stored model parameters[default: ./model.pth.tar]')
    
    return parser.parse_args()


def train_fn(loader, model, optimizer, criterion):
    print('------Training------')
    loop = tqdm(loader, leave=True)
    losses = []

    for batch_idx, data in enumerate(loop):
            grid = data[0]
            grid = grid.to(device)
            label = data[1]
            label = label.to(device)

            output = model(grid)
            # print(output.shape)
            # print(label.shape)
            loss = criterion(output, label)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)

    return sum(losses)/len(losses)


def check_accuracy(model, test_loader, test_case_num): 
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


def load_checkpoint(path, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(path, map_location=device)
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


def main(opt, clss):
    print('------LOADING DATA------')
    root = '../data/SC_NPY'
    grid_path = os.path.join(root, clss)
    label_path = os.path.join(root, clss.replace('grids', 'labels'))
    dataset= VOXELDataset2(grid_path, label_path, transform=transforms.Compose([To3DGrid(), ToTensor()]))

    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    num_epochs = opt.epochs
    validation_split = opt.validation_split

    dataset_size = len(dataset)
    print(f'len(dataset) : {dataset_size}')

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
        
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
   
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    if opt.model == 'NBVNet':
        print('Model : MyNBVNetV3')
        model = MyNBVNetV3()
    model = model.to(device)

    if opt.optimizer == 'Adam':
        print('Optimizer: Adam')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        print('Optimizer: SGD')
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if opt.loss == 'NBVLoss':
        print('Loss : NBVLoss')
        criterion = NBVLoss2()
    else:
        print('Loss : MSELoss')
        criterion = nn.MSELoss()
    criterion = criterion.to(device)

    if opt.load_model:
        print('Loading saved model')
        load_checkpoint(opt.model_path, model, optimizer, learning_rate)
    else:
        print('Training new model')
    
    # all_losses = []
    for epoch in range(num_epochs):
    # while epoch < num_epochs:
        print(f'epoch : {epoch+1}/{num_epochs}')
        mean_loss = train_fn(train_loader, model, optimizer, criterion)
        # all_losses.append(mean_loss)

        if (epoch+1)%10==0:
            save_checkpoint(model, optimizer, epoch, filename='LM1_my_checkpoint_forevery10epochs.pth.tar')
            # load_checkpoint('my_checkpoint_forevery10epochs.pth.tar', model, optimizer, learning_rate, epoch)

        print('On test loader')
        check_accuracy(model, validation_loader, len(val_indices))
        # epoch += 1

    # with open('loss.txt', 'w') as f:
    #     for item in all_losses:
    #         f.write(item + '\n')

if __name__ == "__main__":
    opt = parse_args()
    main(opt, 'grids_LM1.npy')
    

