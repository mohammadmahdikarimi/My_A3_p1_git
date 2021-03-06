import os
import numpy as np
import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
from sklearn.metrics import average_precision_score
from PIL import Image, ImageDraw
import PIL
import matplotlib.pyplot as plt
from kaggle_submission import output_submission_csv
from classifier import SimpleClassifier, Classifier, Classifier_maz, Classifier_moreConv#, AlexNet
from voc_dataloader import VocDataset, VOC_CLASSES
import shutil 
import simplejson       #save list to files
import csv
import tarfile

from pprint import pprint
import sys
import argparse
import time

#%matplotlib inline
#%load_ext autoreload
#%autoreload 2






def train_classifier(train_loader, classifier, criterion, optimizer):
    classifier.train()
    loss_ = 0.0
    losses = []
    for i, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = classifier(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss)
    return torch.stack(losses).mean().item()
    
def test_classifier(test_loader, classifier, criterion, print_ind_classes=True, print_total=True):
    classifier.eval()
    losses = []
    with torch.no_grad():
        y_true = np.zeros((0,21))
        y_score = np.zeros((0,21))
        for i, (images, labels, _) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            logits = classifier(images)
            y_true = np.concatenate((y_true, labels.cpu().numpy()), axis=0)
            y_score = np.concatenate((y_score, logits.cpu().numpy()), axis=0)
            loss = criterion(logits, labels)
            losses.append(loss.item())
        aps = []
        # ignore first class which is background
        for i in range(1, y_true.shape[1]):
            ap = average_precision_score(y_true[:, i], y_score[:, i])
            if print_ind_classes:
                print('-------  Class: {:<12}     AP: {:>8.4f}  -------'.format(VOC_CLASSES[i], ap))
            aps.append(ap)
        
        mAP = np.mean(aps)
        test_loss = np.mean(losses)
        if print_total:
            print('mAP: {0:.4f}'.format(mAP))
            print('Avg loss: {}'.format(test_loss))
        
    return mAP, test_loss, aps
    
def plot_losses(train, val, test_frequency, num_epochs):
    plt.plot(train, label="train")
    indices = [i for i in range(num_epochs) if ((i+1)%test_frequency == 0 or i ==0)]
    plt.plot(indices, val, label="val")
    plt.title("Loss Plot")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    
def plot_mAP(train, val, test_frequency, num_epochs):
    indices = [i for i in range(num_epochs) if ((i+1)%test_frequency == 0 or i ==0)]
    plt.plot(indices, train, label="train")
    plt.plot(indices, val, label="val")
    plt.title("mAP Plot")
    plt.ylabel("mAP")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    

def train(classifier, num_epochs, train_loader, val_loader, criterion, optimizer, lr_scheduler, test_frequency=5):
    train_losses = []
    train_mAPs = []
    val_losses = []
    val_mAPs = []
    #decayRate = 0.96
    

    for epoch in range(1,num_epochs+1):
        print("Starting epoch number " + str(epoch))
        train_loss = train_classifier(train_loader, classifier, criterion, optimizer)
        train_losses.append(train_loss)
        lr_scheduler.step()
        print('learning rate :', get_lr(lr_scheduler.optimizer))

        print("Loss for Training on Epoch " +str(epoch) + " is "+ str(train_loss))
        if(epoch%test_frequency==0 or epoch==1):
            mAP_train, _, _ = test_classifier(train_loader, classifier, criterion, False, False)
            train_mAPs.append(mAP_train)
            mAP_val, val_loss, _ = test_classifier(val_loader, classifier, criterion)
            print('Evaluating classifier')
            print("Mean Precision Score for Testing on Epoch " +str(epoch) + " is "+ str(mAP_val))
            val_losses.append(val_loss)
            val_mAPs.append(mAP_val)
    
    return classifier, train_losses, val_losses, train_mAPs, val_mAPs
    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    #Print the start time of the script
    pprint("Start Time %s" % time.strftime('%x %X'))

    #Instantiate parser object to parse the input parameters
    parser = argparse.ArgumentParser(description="Load action files and call appropriate JAMA actions")
    parser.add_argument("-e", "--epoch", default=50, help="Neural network training epoch")
    parser.add_argument("-t", "--test_freq", default=5, help="Testing Frequency")
    parser.add_argument("-b", "--b_size", default=256, help="Batch size for training ML")
    parser.add_argument("-o", "--optimizer", default="ADAM", help="Optimizer; SGD or ADAM")
    args = parser.parse_args()
    
    num_epochs = args.epoch
    test_frequency = args.test_freq
    batch_size = args.b_size
    opt_sel = args.optimizer

    print("===============Extract train====================")
    tar = tarfile.open("/MMK_data/pascal_data/VOCtrainval_06-Nov-2007.tar")
    tar.extractall("/raid/pascal_data/")
    print("===============Move train====================")
    shutil.move("/raid/pascal_data/VOCdevkit/", "/raid/pascal_data/VOCdevkit_2007")

    print("===============Extract test====================")
    tar = tarfile.open("/MMK_data/pascal_data/VOCtest_06-Nov-2007.tar")
    tar.extractall("/raid/pascal_data/")

    shutil.move("/raid/pascal_data/VOCdevkit/VOC2007", "/raid/pascal_data/VOCdevkit_2007/VOC2007test")
    print("===============Done with Dataset====================")


      

      
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std= [0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
                torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),                       
                transforms.Resize(227),
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                normalize
            ])

    test_transform = transforms.Compose([
                transforms.Resize(227),
                transforms.CenterCrop(227),
                transforms.ToTensor(),
                normalize,
            ])

    ds_train = VocDataset('/raid/pascal_data/VOCdevkit_2007/VOC2007/','train',train_transform)
    ds_val = VocDataset('/raid/pascal_data/VOCdevkit_2007/VOC2007/','val',test_transform)
    ds_test = VocDataset('/raid/pascal_data/VOCdevkit_2007/VOC2007test/','test', test_transform)


    train_loader = torch.utils.data.DataLoader(dataset=ds_train,
                                                   batch_size=batch_size, 
                                                   shuffle=True,
                                                   num_workers=1)

    val_loader = torch.utils.data.DataLoader(dataset=ds_val,
                                                   batch_size=batch_size, 
                                                   shuffle=True,
                                                   num_workers=1)

    test_loader = torch.utils.data.DataLoader(dataset=ds_test,
                                                   batch_size=batch_size, 
                                                   shuffle=False,
                                                   num_workers=1)


    classifier = Classifier_moreConv().to(device)


    # TODO: Run your own classifier here
    #classifier = Classifier_maz().to(device)
    #classifier = Classifier().to(device)


    criterion = nn.MultiLabelSoftMarginLoss()
    #optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.9)

    if opt_sel == "SGD":
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
        decayRate = 0.97
    elif opt_sel == "ADAM":
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
        decayRate = 1
        
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    # optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

    #classifier, train_losses, val_losses, train_mAPs, val_mAPs = train(classifier, num_epochs, train_loader, val_loader, criterion, optimizer, test_frequency)

    classifier, train_losses, val_losses, train_mAPs, val_mAPs = train(classifier, num_epochs, train_loader, val_loader, criterion, optimizer,lr_scheduler, test_frequency)       


    f = open('/results/train_losses.txt', 'w')
    simplejson.dump(train_losses, f)
    f.close()
    f = open('/results/val_losses.txt', 'w')
    simplejson.dump(val_losses, f)
    f.close()
    f = open('/results/train_mAPs.txt', 'w')
    simplejson.dump(train_mAPs, f)
    f.close()
    f = open('/results/val_mAPs.txt', 'w')
    simplejson.dump(val_mAPs, f)
    f.close()


    plot_losses(train_losses, val_losses, test_frequency, num_epochs)
    plot_mAP(train_mAPs, val_mAPs, test_frequency, num_epochs)

    mAP_test, test_loss, test_aps = test_classifier(test_loader, classifier, criterion)
    print("MAP for test is :" , mAP_test)

    f = open('/results/mAP_test.txt', 'w')
    simplejson.dump(mAP_test, f)
    f.close()
    torch.save(classifier.state_dict(), '/results/voc_my_best_classifier.pth')
    output_submission_csv('/results/my_solution.csv', test_aps)