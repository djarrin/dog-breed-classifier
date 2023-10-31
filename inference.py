import json, logging, sys, os, io, requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def net(num_classes):
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    
    model.fc = nn.Sequential(
                   nn.Linear(num_features, num_classes)
    )
    
    return model


def model_fn(model_dir):
    print('model_fn being called')
    print(f'model_dir: ${model_dir}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device: ${device}')
    model = net(133).to(device)
    
    # List the contents of the model directory
    contents = os.listdir(model_dir)
    
    # Print the contents
    for item in contents:
        print(f'model path items {item}')
        
    model_path = os.path.join(model_dir, 'model.pt')
    
    print(f'model_path: {model_path}')
    
    with open(model_path, 'rb') as f:
        # Load the entire model directly
        model = torch.load(f, map_location=device)
    
    print('model returned')
    return model

def input_fn(request_body, content_type='image/jpeg'):
    return Image.open(io.BytesIO(request_body))

def predict_fn(input_object, model):   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    print('ready for prediction')
    testing_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    print('transforming input')
    input_object = testing_transform(input_object)
    
    if torch.cuda.is_available():
        input_object = input_object.cuda()
        
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    
    return prediction
    