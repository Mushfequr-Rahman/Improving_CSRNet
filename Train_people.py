import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models, transforms
from model import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter
from _datetime import date
import torchvision
import torch.nn as nn
from CIFAR_dataset import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch',type=int, default=5)
    parser.add_argument('--epoch', type=int , default=10)
    args = parser.parse_args()

    return args





if __name__=="__main__":
    """
    Get CIFAR-100 Dataset 
    Get VGG16 Network 
    Train on coarse labels 
    Save weights 
    """
    args =get_args()
    epochs = args.epoch
    batch_size =args.batch
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using Device: ", device)


    writer = SummaryWriter("runs_2/")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    dataset = CIFAR100(root='people_class/', train=True, download=True, coarse=True, transform=transform_train)
    test_dataset = CIFAR100(root='people_class/', train=False, download=True, coarse=True, transform=transform_test)

    #lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]

    train_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True,num_workers=4)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=1)
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096,20)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("My_model", model)
    prev_acc = 0

    for epoch in range(1,epochs+1):
        model.train()
        epoch_loss = 0.0

        for i,data in enumerate(train_loader):
            optimizer.zero_grad()
            label = data[2].to(device)
            #img = torch.squeeze(data[0]).permute(2,1,0)
            #img = data[0]
            output = model(data[0].to(device))
            #print(output.shape)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if(i%100==0):
                print('[%d, %5d] loss: %.3f' %(epoch, i + 1, epoch_loss / 100))
                writer.add_scalar("Training Loss",epoch_loss/100, epoch-1)
                epoch_loss = 0.0


        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i,data in enumerate(test_loader):
                img, label = data[0], data[2]
                img = img.to(device)
                label = label.to(device)
                test_out = model(img)
                _,predicted = torch.max(test_out.data,1)
                #print("Predicted Value: ", predicted)
                total += label.size(0)
                correct += (predicted==label).sum().item()
                #print("Total" , total)
                #print("Correct: ", correct)

            current_acc = (100 * correct / total)
            print('Current Test Accuracy: %d %%' %(current_acc) )
            writer.add_scalar("Testing Accuracy", (current_acc), epoch - 1)
            if(current_acc>=prev_acc):
                print("Saving Weights")
                torch.save(model.state_dict(),"checkpoints/best_cifar.pt")

    print('Training Model on Cifar compleleted')





