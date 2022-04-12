import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter
import copy
import time

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from image_folder import image_folder
from torchvision import transforms
import random
import os
import xlsxwriter
from net import *
import re
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import random
import datetime

class TLMyConvNet(nn.Module):
    def __init__(self, conv1, conv2):
        super(TLMyConvNet, self).__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.classifier = nn.Sequential(
            nn.Linear(32 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def train_model_GPU(device, model, traindataloader, train_rate, testdataloader):
    early_stopping = EarlyStopping(patience=10, verbose=True)

    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * train_rate)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.CrossEntropyLoss()  

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    test_acc_all = []
    since = time.time()

    epoch = 0
    acc = 0.0

    while True:
        epoch += 1
        print('Epoch {}'.format(epoch))
        print('-' * 10)
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        test_corrects = 0
        test_num = 0
        for step, (b_x, b_y) in enumerate(traindataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            if step < train_batch_num:
                model.train() 
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)
                train_corrects += torch.sum(pre_lab == b_y.data)
                train_num += b_x.size(0)
            else:
                model.eval()  
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)
        for step, (b_x, b_y) in enumerate(testdataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval() 
            output = model(b_x)
            pre_lab = torch.argmax(output, 1)
            test_corrects += torch.sum(pre_lab == b_y.data)
            test_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        test_acc_all.append(test_corrects.double().item() / test_num)
        print('{} Train Loss: {:.4f}  Train Acc: {:.4f}'.format(
            epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f}  val Acc: {:.4f}'.format(
            epoch, val_loss_all[-1], val_acc_all[-1]))
        print('test Acc: {:.4f}'.format(test_acc_all[-1]))
        
        if best_loss > val_loss_all[-1]:
            best_loss = val_loss_all[-1]
            acc = test_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        
        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(
            time_use // 60, time_use % 60))
            
        if epoch >= 10:
            early_stopping(val_loss_all[-1], model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
    model.load_state_dict(best_model_wts)
    print("acc_end = ",acc)
    return model, acc

def get_all_files(rootDir):
    def sort_key(s):
        if s:
            try:
                c = re.findall('\d+', s)[0]
            except:
                c = -1
            return int(c)
    def strsort(alist):
        alist.sort(key=sort_key, reverse=False)
        return alist
    l = []
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            l.append(file)
    l = strsort(l)
    return l

def main():
    if not torch.cuda.is_available():
        print("GPU is not available!")
        return
    else:
        print("GPU is available!")
        device = torch.device("cuda")

    File = open('SMS_accuracy.txt', 'w')

    net_list = get_all_files("../Candidate Set")

    class_list = ["airplane", "bird", "ship", "dog"]
    # class_list = ["airplane", "dog"]

    for net_number in range(len(net_list)):
        net_name = "../Candidate Set/"+net_list[net_number]
        print('load ' + net_name + ' ...')

        myconvnet_old = LeNet(2)
        myconvnet_old.load_state_dict(torch.load(net_name))
        myconvnet_old = myconvnet_old.to(device)
        
        print('load ' + net_name + ' complete')
        conv1 = myconvnet_old.conv1
        conv2 = myconvnet_old.conv2

        for param in conv1.parameters():
            param.requires_grad_(False)
        for param in conv2.parameters():
            param.requires_grad_(False)

        transform_n = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([28, 28]),
            transforms.ToTensor(),
        ])

        train_data = image_folder(
            "../Target Dataset/train",
            class_list,
            transform=transform_n,
        )
        train_loader = Data.DataLoader(
            dataset=train_data, 
            batch_size=64, 
            shuffle=False,
            num_workers=10, 
            pin_memory=True
        )

        val_data = image_folder(
            "../Target Dataset/val",
            class_list,
            transform=transform_n,
        )
        test_loader = Data.DataLoader(
            dataset=val_data, 
            batch_size=64, 
            shuffle=False,
            num_workers=10,  
            pin_memory=True
        )

        test_acc = 0.0
        n_times = 3
        print(str(net_name),end="|", flush=True, file=File)
        for i in range(n_times):
            print("\nthe "+str(i)+" times.....\n")
            myconvnet_new = TLMyConvNet(conv1, conv2).to(device)
            myconvnet_new, acc = train_model_GPU(
                device, myconvnet_new, train_loader, 0.9, test_loader
            )
            test_acc += acc
            print("best acc = ", acc)
            print(acc,end="|", flush=True, file=File)
            print("test_acc = ", test_acc)

        print(str(test_acc/n_times), flush=True, file=File)
        print(str(test_acc/n_times)+"\n", flush=True)

        # print(str(net_name),end="|", flush=True, file=File)
        # myconvnet_new = TLMyConvNet(conv1, conv2).to(device)
        # myconvnet_new, acc = train_model_GPU(
        #     device, myconvnet_new, train_loader, 0.9, test_loader
        # )
        # print(str(acc), flush=True, file=File)
        # print(str(acc)+"\n", flush=True)

if __name__ == '__main__':
    main()