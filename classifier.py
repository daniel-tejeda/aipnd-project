# PROGRAMMER: Daniel Tejeda

import matplotlib.pyplot as plt

import torch
import numpy as np
import torch.nn.functional as F
import torchvision

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
#from workspace_utils import active_session, keep_awake
import json

class Classifier():

    def __init__(self, in_arg):

        self.base_models = ['vgg16','vgg13', 'alexnet']

        self.data_transforms = {

            'train': transforms.Compose([transforms.RandomRotation(30),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),

            'test': transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),

            'valid': transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        }

        self.image_datasets = { x: datasets.ImageFolder('{}{}'.format(in_arg.data_dir,x),
                                                   transform=self.data_transforms[x])
                          for x in ['train','test','valid'] }

        self.dataloaders = { x: torch.utils.data.DataLoader(self.image_datasets[x],
                                                       batch_size=64, shuffle=True)
                       for x in ['train','test','valid'] }

        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train','test','valid'] }

        self.in_arg = in_arg

        self.model = self.create_model(arch=in_arg.arch,
                                  data_dir=in_arg.data_dir,
                                  hidden_units=in_arg.hidden_units,
                                  learnrate=in_arg.learning_rate)



    def create_model(self, arch, data_dir, hidden_units=512, dropout=0.4, learnrate=0.02):

        #image_datasets, dataloaders, dataset_sizes = init_datasets(data_dir)

        error_msg = "Only {} supported at this time".format(
                    ", ".join(self.base_models))

        assert (arch in self.base_models), error_msg

        #_model = base_models[arch]
        _model = eval('models.{}(pretrained=True)'.format(arch))

        #freeze params, avoid backprop
        for param in _model.parameters():
            param.requires_grad = False

        #hyperparameters for classifier

        if arch == 'alexnet':
            input_size = _model.classifier[1].in_features
        else:
            input_size = _model.classifier[0].in_features

        output_size = 102

        # Create the network, define the criterion and optimizer
        classifier = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_units, output_size),
            nn.LogSoftmax(dim=1))

        _model.arch = arch

        #assign my classifier
        _model.classifier = classifier
        _model.class_to_idx = self.image_datasets['train'].class_to_idx
        _model.idx_to_class = { x: y for y, x in _model.class_to_idx.items() }

        _model.optimizer = optim.Adam( _model.classifier.parameters(), lr=learnrate)
        _model.train_epochs = 0

        return _model


    def train(self):
        device = ("cuda:0" if torch.cuda.is_available() and in_arg.gpu else "cpu")
        self.train_model(self.model, self.in_arg.epochs, device)

    def train_model(self, model, epochs=5, device='cpu'):


        criterion=nn.NLLLoss()
        optimizer = model.optimizer
        model.to(device)

        global_epochs =  model.train_epochs+epochs

        for epoch in range(model.train_epochs, global_epochs):

            print('-' * 40)
            print('Epoch {}/{}'.format(epoch+1, global_epochs))
            print('-' * 40)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                print("\n[{}-{}] Start | {} images".format(
                      epoch+1, phase, self.dataset_sizes[phase]))



                for step, (images, labels) in enumerate(self.dataloaders[phase]):

                    images, labels = images.to(device), labels.to(device)

                    #zero grad
                    optimizer.zero_grad()

                    #forward
                    with torch.set_grad_enabled(phase == 'train'):

                        output = model.forward(images)
                        loss = criterion(output, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(output, dim=1)

                    corrects = (preds == labels.data)
                    corrects_sum = torch.sum(corrects)

                    running_loss += loss.item() #check
                    running_corrects += corrects_sum

                    progbar.value += images.size(0)

                if phase == 'train':
                    model.train_epochs += 1
                    save_checkpoint(model)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]


                print("[{}-{}] End | Loss: {:.4f} ... Acc: {:.4f}\n".format(
                      epoch+1, phase, epoch_loss, epoch_acc))





    def validate_model(self, model, device='cpu'):

        criterion=nn.NLLLoss()

        model.to(device)
        model.eval()

        running_corrects = 0
        running_loss = 0

        phase = 'test'

        print("\n[{}] Start | {} images".format(
                      phase, dataset_sizes[phase]))


        for images, labels in dataloaders[phase]:

            images, labels = images.to(device), labels.to(device)

            with torch.set_grad_enabled(False):

                output = model.forward(images)
                loss = criterion(output, labels)

            _, preds = torch.max(output, dim=1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

            progbar.value += images.size(0)

        test_loss = running_loss / dataset_sizes[phase]
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy = running_corrects.double() / dataset_sizes[phase]


        print("[{}] End | Loss: {:.4f} ... Acc: {:.4f}\n".format(
                      phase, test_loss, accuracy))

        return test_loss, accuracy





def save_checkpoint(_model, path=''):

    if path ==  '':
        path = _model.arch + '-flowers-classifier.pth' #default

    # Basic details
    checkpoint = {
        'arch': _model.arch,
        'classifier': _model.classifier,
        'state_dict': _model.state_dict(),
        'optimizer' : _model.optimizer,
        'optimizer_state_dict': _model.optimizer.state_dict(),
        'class_to_idx': _model.class_to_idx,
        'idx_to_class': _model.idx_to_class,
        'train_epochs': _model.train_epochs
    }

    # Save the data to the path
    torch.save(checkpoint, path)





def load_checkpoint(path):

    # Load in checkpoint
    checkpoint = torch.load(path)

    arch = checkpoint['arch']

    error_msg = "Only {} supported at this time".format(
                ", ".join(base_models))

    assert (arch in base_models), error_msg

    _model = eval('models.{}(pretrained=True)'.format(arch))
    _model.arch = arch

    #freeze params, avoid backprop
    for param in _model.parameters():
        param.requires_grad = False

    _model.classifier = checkpoint['classifier']

    # Load state dict
    _model.load_state_dict(checkpoint['state_dict'])

    # Model basics
    _model.class_to_idx = checkpoint['class_to_idx']
    _model.idx_to_class = checkpoint['idx_to_class']

    # Optimizer
    _model.optimizer = checkpoint['optimizer']
    _model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #epochs
    _model.train_epochs = checkpoint['train_epochs']

    return _model






def predict_tensor(img_tensor, model, topk=5, device='cpu'):

    img_tensor.requires_grad_(False)

    #batch dimmension
    img_tensor.unsqueeze_(0)

    img_tensor = img_tensor.to(device)

    with torch.set_grad_enabled(False):
        model.to(device)
        model.eval()
        output = model.forward(img_tensor)
        ps = torch.exp(output)

    return ps.topk(topk, dim=1)


def predict(image_path, model, topk=5, device='cpu'):

    img_tensor = process_image(image_path)

    pred, classes = predict_tensor(img_tensor, model, topk, device='cpu')

    return pred, classes
