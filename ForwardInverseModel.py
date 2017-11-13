#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:32:41 2017

@author: nigno
"""

# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

from read_data import get_imgs_acts_dict
from pokeNet import PokeNet, PokeNet1C

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

plt.ion()   # interactive mode

def mse_loss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

train = True
restore = False

N_trials = 100

# Data augmentation and normalization for training
# Just normalization for validation
use_gpu = torch.cuda.is_available()

class AtabakDataset(Dataset):
    """Face trajectories dataset."""

    def __init__(self, is_train = True):
        
        img_h = 128
        img_w = 128
        
        if (is_train):
            bfn, acts_array_train = get_imgs_acts_dict()
        else:
            bfn, acts_array_train = get_imgs_acts_dict(False)
        
        img1 = bfn['imgs_array_before']
        img1 = torch.from_numpy(img1).float()
        self.img1 = img1.view(img1.size(0), 1, img_w, img_h)

        img2 = bfn['imgs_array_after']
        img2 = torch.from_numpy(img2).float()
        self.img2 = img2.view(img2.size(0), 1, img_w, img_h)
        
        nb_digits = 12
        acts_array_train = torch.from_numpy(acts_array_train).long()
        acts_array_train = acts_array_train.view(acts_array_train.size(0), 1)
        self.input_actions = torch.FloatTensor(acts_array_train.size(0), nb_digits)
        self.input_actions.zero_()
        self.input_actions.scatter_(1, acts_array_train, 1)

    def __len__(self):
        return len(self.img1)

    def __getitem__(self, idx):
        img1Element = self.img1[idx]
        img2Element = self.img2[idx]
        input_actionsElement = self.input_actions[idx]
        sample = {'img1': img1Element, 
                  'img2': img2Element, 
                  'input_actions': input_actionsElement}

        return sample

atabak_dataset_train = AtabakDataset()
atabak_dataset_test = AtabakDataset(False)

accuracyArray = np.zeros(N_trials)
confusionArrayNP = np.zeros((N_trials, 12, 12))
confusionArrayTensor = torch.Tensor(N_trials, 12, 12)

for trials in range(N_trials):
    print('-------------' + str(trials) + '-------------')
    dataloader_train = torch.utils.data.DataLoader(atabak_dataset_train, batch_size=110,
                                             shuffle=True, num_workers=8)
    dataloader_test = torch.utils.data.DataLoader(atabak_dataset_test, batch_size=110,
                                             shuffle=True, num_workers=8)
    
    print(len(atabak_dataset_train))
    print(len(atabak_dataset_test))
    
    
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    
    #def imshow(inp, title=None):
    #    """Imshow for Tensor."""
    #    inp = inp.numpy().transpose((1, 2, 0))
    #    inp = np.clip(inp, 0, 1)
    #    plt.imshow(inp)
    #    if title is not None:
    #        plt.title(title)
    #    plt.pause(0.001)  # pause a bit so that plots are updated
    
    def train_model(model, criterion1, criterion2, optimizer, scheduler, num_epochs=25, start_epoch=0, loss_list=[], acc_list=[], loss_list_val=[], acc_list_val=[]):
        since = time.time()
    
        best_model_wts = model.state_dict()
        best_acc = 0.0
    
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + start_epoch, num_epochs - 1 + start_epoch))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
            #for phase in ['train']:
                if phase == 'train':
                    #scheduler.step()
                    model.train(True)  # Set model to training mode
                    dataloader = dataloader_train
                    dataset_size = len(atabak_dataset_train)
                else:
                    model.train(False)  # Set model to evaluate mode
                    dataloader = dataloader_test
                    dataset_size = len(atabak_dataset_test)
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                samples_count = 0
                for data in dataloader:
                    samples_count += 1
                    # get the inputs
                    samples = data
    
                    # wrap them in Variable
                    if (use_gpu):
                        img1 = Variable(samples['img1'].cuda())
                        img2 = Variable(samples['img2'].cuda())
                        input_actions = Variable(samples['input_actions'].cuda())
                    else:
                        img1 = Variable(samples['img1'])
                        img2 = Variable(samples['img2'])
                        input_actions = Variable(samples['input_actions'])
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    output_actions, output_latent, lat2 = model(img1, img2, input_actions)
                    _, preds = torch.max(output_actions.data, 1)
                    _, labels = torch.max(input_actions.data, 1)
                    loss = 0.1 * criterion1(output_latent, Variable(lat2.data)) + criterion2(output_actions, Variable(labels))
                    #loss = mse_loss(output_latent, lat2) + 0.1 * criterion2(output_actions, Variable(labels))               
    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                    # statistics
                    running_loss += loss.data[0]
                    #print('sample {}/{}. Loss: {}. Dataset size: {}'.format(samples_count, len(dataloader), loss.data[0], dataset_size))
                    running_corrects += torch.sum(preds == labels)
                    printProgressBar(samples_count, len(dataloader), prefix = phase, suffix = 'Complete', length = 50)
                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corrects / dataset_size
    
                print('{} Loss: {:.8f} Acc: {:.8f}'.format(phase, epoch_loss, epoch_acc))
                if phase == 'val':
                    loss_list_val += [epoch_loss]
                    acc_list_val += [epoch_acc]
                else:
                    loss_list += [epoch_loss]
                    acc_list += [epoch_acc]
    
                # deep copy the model
                #if phase == 'val' and epoch_acc > best_acc:
                #if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save({
                            'epoch': epoch + start_epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_acc': best_acc,
                            'loss_list': loss_list,
                            'acc_list': acc_list,
                            'loss_list_val': loss_list_val,
                            'acc_list_val': acc_list_val,
                            }, 'checkpoint.tar' )
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
    
    #def visualize_model(model, num_images=6):
    #    images_so_far = 0
    #    fig = plt.figure()
    #
    #    for i, data in enumerate(dataloader_val):
    #        samples = data
    #        if use_gpu:
    #            inputs = Variable(samples['image'].cuda())
    #            trajectories = Variable(samples['trajectories'].cuda())
    #        else:
    #            inputs, trajectories = Variable(samples['image']), Variable(samples['trajectories'])
    #
    #        outputs = model(inputs)
    #        outData = outputs.cpu().data.resize_(outputs.size()[0], 200, 2)
    #
    #        for j in range(inputs.size()[0]):
    #            images_so_far += 1
    #            ax = plt.subplot(num_images//2, 2, images_so_far)
    #            ax.axis('off')
    #            #imshow(inputs.cpu().data[j])
    #            plt.scatter(trajectories[j][:, 0].cpu().data.numpy(), trajectories[j][:, 1].cpu().data.numpy(), s=10, marker='.', color='red')
    #            plt.scatter(outData[j][:, 0].numpy(), outData[j][:, 1].numpy(), s=10, marker='.', color='blue')
    #            plt.pause(0.001)
    #
    #            if images_so_far == num_images:
    #                return
    #   
             
    model = PokeNet1C(1024, 12)
    if (use_gpu):
        model = model.cuda()
        
    print(model)
      
    if (train == True):
        criterion1 = nn.MSELoss()
        criterion2 = nn.CrossEntropyLoss()
    
        # Observe that all parameters are being optimized
        #optimizer_ft = optim.ADAM(model.parameters(), lr=0.001, momentum=0.9)
        optimizer_ft = optim.Adam(model.parameters())
    
        # Decay LR by a factor of 0.1 every 7 epochs
        #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        exp_lr_scheduler = None
    
        model = train_model(model, criterion1, criterion2, optimizer_ft, exp_lr_scheduler,
                               num_epochs=40)
    else:
        if os.path.isfile('checkpoint.tar'):
            print("=> loading checkpoint '{}'".format('checkpoint.tar'))
            checkpoint = torch.load('checkpoint.tar')
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            loss_list = checkpoint['loss_list']
            acc_list = checkpoint['acc_list']
            loss_list_val = checkpoint['loss_list_val']
            acc_list_val = checkpoint['acc_list_val']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})"
                      .format(checkpoint['epoch']))
            
    if (restore == True):
        criterion1 = nn.MSELoss()
        criterion2 = nn.CrossEntropyLoss()
        
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        exp_lr_scheduler = None
        
        model = train_model(model, criterion1, criterion2, optimizer_ft, exp_lr_scheduler,
                               num_epochs=40, start_epoch=start_epoch,
                               loss_list=loss_list, acc_list=acc_list)
        
        
    dataloader_confusion = torch.utils.data.DataLoader(atabak_dataset_test, batch_size=1,
                                             shuffle=False, num_workers=8)
    
    
    import matplotlib.ticker as ticker
    
    act_dict = {'hook_draw':0, 'hook_push':1, 'hook_tap_from_right':2, 'hook_tap_from_left':3,
                'rake_draw':4, 'rake_push':5, 'rake_tap_from_right':6, 'rake_tap_from_left':7, 
                'stick_draw':8, 's_traintick_push':9, 'stick_tap_from_right':10, 'stick_tap_from_left':11}
    act_list = ['hook_draw', 'hook_push', 'hook_tap_from_right', 'hook_tap_from_left',
                'rake_draw', 'rake_push', 'rake_tap_from_right', 'rake_tap_from_left', 
                'stick_draw', 's_traintick_push', 'stick_tap_from_right', 'stick_tap_from_left']
    n_categories = 12
    
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    
    dataset_size = len(atabak_dataset_test)
    running_corrects = 0
    # Go through a bunch of examples and record which are correctly guessed
    for data in dataloader_confusion:
        samples = data
        # wrap them in Variable
        if (use_gpu):
            img1 = Variable(samples['img1'].cuda())
            img2 = Variable(samples['img2'].cuda())
            input_actions = Variable(samples['input_actions'].cuda())
        else:
            img1 = Variable(samples['img1'])
            img2 = Variable(samples['img2'])
            input_actions = Variable(samples['input_actions'])
        
        output_actions, output_latent, lat2 = model(img1, img2, input_actions)
        _, preds = torch.max(output_actions.data, 1)
        _, labels = torch.max(input_actions.data, 1)
        running_corrects += torch.sum(preds == labels)
        
    
        if (use_gpu):          
            confusion[labels.cpu().numpy()[0]][preds.cpu().numpy()[0]] += 1
        else:
            confusion[labels.numpy()[0]][preds.numpy()[0]] += 1
    
    epoch_acc = running_corrects / dataset_size
    
    accuracyArray[trials] = epoch_acc
    
    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    
    confusionArrayNP[trials] = confusion.numpy()
    
    confusionArrayTensor[trials] = confusion
    

np.save("accuracy.npy", accuracyArray)
np.save("confusionNP.npy", confusionArrayNP)
np.save("confusionTensor.npy", confusionArrayTensor)

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + act_list, rotation=90)
ax.set_yticklabels([''] + act_list)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()

##visualize_model(model_ft)
#
if (train == False):
    fig = plt.figure()
    plt.plot(loss_list, color='red')
    plt.plot(loss_list_val, color='blue')
    fig = plt.figure()
    plt.plot(acc_list, color='red')
    plt.plot(acc_list_val, color='blue')
#plt.plot(loss_list_val, color='blue')
#
raw_input('Press enter to continue: ')
#
##
##visualize_model(model_conv)
##
##plt.ioff()
##plt.show()
##
##plt.pause(2)
##
##
