import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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
from SMS_methods import get_ave_OVR_new
import datetime

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

def norm_pdf(x,mu,sigma, a): 
    biGauss1 = multivariate_normal(mean = mu, cov = sigma)
    pdf = a * biGauss1.pdf(x) 
    return pdf

def main():
    if not torch.cuda.is_available():
        print("GPU is not available!")
        return
    else:
        print("GPU is available!")
        device = torch.device("cuda")

    File = open('SMS_separation degree.txt', 'w')

    net_list = get_all_files("../Candidate Set")

    class_list = ["airplane", "bird", "ship", "dog"]
    # class_list = ["airplane", "dog"]

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
        batch_size=60000,
        # batch_size=60,
        # batch_size=int(len(train_data)*0.01),  
        shuffle=False,
        num_workers=0,  
        pin_memory=True
    )

    for step, (X, y) in enumerate(train_loader):
        break

    X = X.to(device)
    end = time.time()
    
    for net_number in range(len(net_list)):
        net_name = "../Candidate Set/"+net_list[net_number]
        print('load ' + net_name + ' ...')

        myconvnet_old = LeNet(2)
        myconvnet_old.load_state_dict(torch.load(net_name))
        myconvnet_old = myconvnet_old.to(device)
        
        print('load ' + net_name + ' complete')

        output = myconvnet_old(X)

        output_new = nn.Softmax(dim=1)(output/2.0).cpu().detach().numpy() 
        # new_X = output_new
        new_X = np.delete(output_new, 0, axis = 1)

        X_list = [0]*len(class_list)
        for i in range(len(y)):
            if type(X_list[y[i]]) != type(new_X[i]):
                X_list[y[i]] = new_X[i]
            else:
                a = X_list[y[i]]
                b = new_X[i]
                c = np.row_stack((a,b))
                X_list[y[i]] = c
        means = []
        covariances = []
        weights = []

        for i in range(len(X_list)):
            MOG = GaussianMixture(n_components=1, covariance_type='full').fit(X_list[i])
            # means.append(MOG.means_[0][0])
            # covariances.append(MOG.covariances_[0][0][0])
            # weights.append(MOG.weights_[0])
            means.append(MOG.means_[0])
            covariances.append(MOG.covariances_[0])
            weights.append(MOG.weights_[0])
            
        ave_OLR = get_ave_OVR_new(means,covariances,weights)
        print("net_number = ",net_number)
        print("ave_OLR = ",ave_OLR)
        print("\n")

        # break

        File.write(str(net_name)+"|"+str(ave_OLR) +"\n")
    
    print("all time = ", (time.time() - end))


if __name__ == '__main__':
    main()