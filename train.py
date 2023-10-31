#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


import argparse

def test(model, test_loader, criterion, device):  
        
    model = model.to(device)

    model.eval()
    
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)  
        labels = labels.to(device)
        
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader)
    total_acc = running_corrects / len(test_loader.dataset)
    
    print(f"Test => total loss: {total_loss}")
    print(f"Test => Test Accuracry: {total_acc}")
          
    

def train(model, train_loader, criterion, optimizer, epoch, device):
    model = model.to(device)
    model.train()
    
    for e in range(epoch):
        running_loss=0
        correct=0
        for data, target in train_loader:
            data=data.to(device)
            target=target.to(device)
            optimizer.zero_grad()
            pred = model(data)               
            loss = criterion(pred, target)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Epoch {e}")
    
    print(f"average test loss: {running_loss/len(train_loader.dataset)}")
    print(f"Accuracy {100*(correct/len(train_loader.dataset))}%")
    
    return model


def net(num_classes):
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    
    model.fc = nn.Sequential(
                   nn.Linear(num_features, num_classes)
    )
    
    return model

def create_data_loaders(data, batch_size):
                
    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True
    )


def main(args):
    
    # Download and load the training data
    training_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    testing_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        
    ])

    trainset = datasets.ImageFolder(root=os.environ['SM_CHANNEL_TRAIN'], transform=training_transform)
    testset = datasets.ImageFolder(root=os.environ['SM_CHANNEL_TEST'], transform=testing_transform)
    train_loader = create_data_loaders(data=trainset, batch_size=args.batch_size)
    test_loader = create_data_loaders(data=testset, batch_size=args.batch_size)
    
    # Access the shape of the first data sample
    num_classes = len(trainset.classes)
    
    print("Number of classes:", num_classes)
    

    model=net(num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    
    model=model.to(device)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learningRate)

    model=train(model, train_loader, loss_criterion, optimizer, args.epochs, device)
    
    '''
    Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    
    '''
    Save the trained model
    '''
    torch.save(model, '/opt/ml/model/model.pt')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )
    
    parser.add_argument(
        "--learningRate", 
        type=float, 
        default=0.01
    )

    args=parser.parse_args()
    
    main(args)