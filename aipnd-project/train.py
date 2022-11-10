import argparse, os, time, json
import torch, torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image

def accuracy(model, loader, device='cpu'):
    correct, total = 0, 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def main():
    print("initializing training...")

    global args
    parser = argparse.ArgumentParser(description='training options')
    parser.add_argument('data_directory', help='data directory')
    parser.add_argument('--save_dir', type=str, dest='save_dir', default='', help='saved data directory')
    parser.add_argument('--arch', type=str, help='choose architecture')
    parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.001, help='choose learning rate')
    parser.add_argument('--hidden_units', action='store', type=int, dest='hidden_units', default=120, help='choose hidden units')
    parser.add_argument('--epochs', type=int, dest='epochs', default=3, help='choose epochs')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='use GPU for training')
    args = parser.parse_args()

    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("no gpu detected!")
    if (not os.path.isdir(args.data_directory)):
        raise Exception('invalid directory!')
    data_dir = os.listdir(args.data_directory)
    if (not set(data_dir).issubset({'test', 'train', 'valid'})):
        raise Exception('invalid sub-directories!')
    if args.arch not in ('vgg', 'densenet', None):
        raise Exception('invalid architecture choice!')

    train_dir = args.data_directory + '/train'
    test_dir = args.data_directory + '/test'
    valid_dir = args.data_directory + '/valid'
    data_dir = [train_dir, test_dir, valid_dir]

    train_dir, test_dir, valid_dir = data_dir
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    modified_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_datasets = datasets.ImageFolder(
        train_dir, transform=modified_transforms)
    valid_datasets = datasets.ImageFolder(
        valid_dir, transform=modified_transforms)
    test_datasets = datasets.ImageFolder(
        test_dir, transform=modified_transforms)

    trainloaders = torch.utils.data.DataLoader(
        train_datasets, batch_size=32, shuffle=True)
    validloaders = torch.utils.data.DataLoader(
        valid_datasets, batch_size=32, shuffle=True)
    testloaders = torch.utils.data.DataLoader(
        test_datasets, batch_size=32, shuffle=True)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    loaders = {'train': trainloaders, 'valid': validloaders,
               'test': testloaders, 'labels': cat_to_name}

    arch_type = 'vgg' if args.arch is None else args.arch
    if (arch_type == 'vgg'):
        model = models.vgg19(pretrained=True)
        input_node = 25088
    elif (arch_type == 'densenet'):
        model = models.densenet121(pretrained=True)
        input_node = 1024
    hidden_units = int(4096 if args.hidden_units is None else args.hidden_units)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_node, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    learn_rate = float(0.001 if args.learning_rate is None else args.learning_rate)
    epochs = int(3 if args.epochs is None else args.epochs)
    device = 'cuda' if args.gpu else 'cpu'

    train_loader = loaders['train']
    validation_loader = loaders['valid']
    test_loader = loaders['test']

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    steps = 0
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0
        for input, (inputs, labels) in enumerate(train_loader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % 40 == 0:
                valid_accuracy = accuracy(model, validation_loader, device)
                print("epoch: {}/{}... ".format(epoch+1, epochs), "validation_loss: {:.4f}".format(running_loss/40), "validation_accuracy: {}".format(round(valid_accuracy, 4)))
                running_loss = 0
    print("training completed successfully.")
    test_result = accuracy(model, test_loader, device)
    print('final accuracy: {}'.format(test_result))

    if (args.save_dir is None):
        save_dir = 'checkpoint.pth'
    else:
        save_dir = args.save_dir
    checkpoint = {
                'model': model.cpu(),
                'features': model.features,
                'classifier': model.classifier,
                'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)

main()