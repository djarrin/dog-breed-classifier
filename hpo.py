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

# from sagemaker.debugger import Rule, DebuggerHookConfig


import argparse

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model = model.to(device)
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.shape[0], -1)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = len(test_loader.dataset)
    test_accuracy = correct/test_loss
    print(f'Test set: Accuracy: {test_accuracy} = {100*(test_accuracy)}%)')
    

def train(model, train_loader, criterion, optimizer, epoch, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
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
            
       
            data_np = data.cpu().numpy()
            target_np = target.cpu().numpy()
            pred_np = pred.cpu().detach().numpy()

            print(f"Data shape: {data_np.shape}, Target shape: {target_np.shape}")
            print(f"Data values: {data_np}")
            print(f"Target values: {target_np}")
            print(f"Predicted values: {pred_np}")
            
            loss = criterion(pred, target)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Epoch {e}")
    
    print(f"average test loss: {running_loss/len(train_loader.dataset)}")
    print(f"Accuracy {100*(correct/len(train_loader.dataset))}%")

def net(num_classes):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    
    model.fc = nn.Sequential(
                   nn.Linear(num_features, num_classes)
    )
    
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
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
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net(num_classes)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    
    model=model.to(device)

    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learningRate)


    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer, args.epochs, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
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