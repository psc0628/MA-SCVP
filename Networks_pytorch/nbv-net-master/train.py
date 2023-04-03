import numpy as np
import csv
import classification_nbv as cnbv
import nbvnet
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, utils
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print(device)
display_dataset = True
display_fwd_pretraining = True
load_weights = False
reading_weights_file = 'weights/paper_param.pth'
saving_weights_file = 'log/weights.pth'
epochs = 1000
batch_size = 1000
learning_rate = 0.001
dropout_prob= 0.3
validation_split = 0.2
shuffle_dataset = True
random_seed = 42

# save parameters used
params = {'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate, 'dropout_prob': dropout_prob}
with open("log/parameters.csv", 'w') as csvfile:
    fieldnames = params.keys()  #['first_name', 'last_name', 'Grade']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
 
    writer.writeheader()
    writer.writerow(params)


nbv_positions = np.genfromtxt('points_in_sphere.txt')

# This function converts a class to its corresponding pose
def getPosition(nbv_class, nbv_positions):
    return nbv_positions[nbv_class]

dataset = cnbv.NBVClassificationDatasetFull(grid_file='./grids.npy', 
                                    nbv_class_file='./labels.npy',
                                    transform=transforms.Compose([
                                    # Reshapes the plain grid
                                    cnbv.To3DGrid(),
                                    #converts to tensors
                                    cnbv.ToTensor()
                                    ]))


dataset_size = len(dataset)
print(f'len(dataset) : {dataset_size}')

indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
    
# train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)


net = nbvnet.NBV_Net(dropout_prob)
net.to(device)
# print(net)

if load_weights:
    state_dict = torch.load(reading_weights_file)
    #print(state_dict.keys())
    net.load_state_dict(state_dict)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr= learning_rate)


def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for sample in testloader:
        
        # get sample data: images and ground truth keypoints
        grids = sample['grid']
        nbvs = sample['nbv_class']
        
        # convert images to FloatTensors
        grids = grids.type(torch.FloatTensor)
        
        # wrap them in a torch Variable
        grids = Variable(grids)    
        grids = grids.to(device)
        
        # wrap them in a torch Variable
        nbvs = Variable(nbvs) 
        nbvs = nbvs.to(device)

        output = model.forward(grids)
        test_loss += criterion(output, nbvs).item()

        # for log.  ps = torch.exp(output)
        equality = (nbvs.data == output.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy


running_loss = 0
save_after = 100

history_epoch = []
history_train_loss = []
history_validation_loss = []
history_train_accuracy = []
history_validation_accuracy = []

import time
tic = time.time()

for e in range(epochs):
    # Cambiamos a modo entrenamiento
    net.train()
    
    for i, sample in enumerate(tqdm(train_loader, leave=True)):        
        # get sample data: images and ground truth keypoints
        grids = sample['grid']
        nbvs = sample['nbv_class']

        # convert grids to FloatTensors
        grids = grids.type(torch.FloatTensor)
        
        # wrap them in a torch Variable
        grids = Variable(grids)    
        grids = grids.to(device)
        
        # wrap them in a torch Variable
        nbvs = Variable(nbvs) 
        nbvs = nbvs.to(device)
        
        optimizer.zero_grad()

        # forward pass to get net output
        output = net(grids)
        # _, output = torch.max(output, dim=1)
        
        # ot = output.cpu()
        # print(ot) 
        # output = torch.unsqueeze(output, 0)  

        # print(output.shape)
        # print(nbvs)
        loss = criterion(output, nbvs)
        # Backpropagation
        loss.backward()
        # Optimización
        optimizer.step()
        
        running_loss += loss.item()
            
    
    # Cambiamos a modo de evaluación
    net.eval()
    if (e+1)%20 == 0:
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, 'my_checkpoint.pth.tar')
            
    # Apagamos los gradientes, reduce memoria y cálculos
    with torch.no_grad():
        train_loss, train_accuracy = validation(net, train_loader, criterion)
        val_loss, val_accuracy = validation(net, validation_loader, criterion)
        
        train_loss, train_accuracy = train_loss, train_accuracy.cpu().numpy()
        val_loss, val_accuracy = val_loss, val_accuracy.cpu().numpy()
        
        train_accuracy = train_accuracy / len(train_loader)
        val_accuracy = val_accuracy / len(validation_loader)
        
        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss),
              "Val. Loss: {:.3f}.. ".format(val_loss),
              "Train Accuracy: {:.3f}".format(train_accuracy),
              "Val. Accuracy: {:.3f}".format(val_accuracy))
    
    history_epoch.append(e)
    history_train_loss.append(train_loss)
    history_validation_loss.append(val_loss)
    history_train_accuracy.append(train_accuracy)
    history_validation_accuracy.append(val_accuracy)
    
    running_loss = 0
    
    if(e % save_after == 0):
        np.save('log/train_loss'+str(e), history_train_loss)
        np.save('log/validation_loss'+str(e), history_validation_loss)
        np.save('log/train_accuracy'+str(e), history_train_accuracy)
        np.save('log/validation_accuracy'+str(e), history_validation_accuracy)
        torch.save(net.state_dict(), 'log/weights'+str(e)+'.pth')
    
    # Make sure training is back on
    net.train()
    
toc = time.time()
print(toc - tic)