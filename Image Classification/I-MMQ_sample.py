import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter
import copy
import time
import cv2
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torchvision import transforms
import random
import os
import xlsxwriter
import re
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import datetime
import torchvision.models as models
from SMS_methods import *
from data_loader import *
from helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate

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
    net_list = get_all_files("/home/yau/zmj/VLDB21/ImageNet-master/ImageNet-master/output_PR18")
        
    class_list_num = list(range(700,1000))

    for rate in [0.2, 0.4, 0.6, 0.8]:
        # for op in [5,10,15,20,25,30]:
        op = 25
        File = open(str(rate)+'_rate_MOG_ImageNet_rank.txt', 'w')
        for net_number in range(0, len(net_list)):
        # for id in ["oa_everything.pth_resnet18.pth"]:
        # for id in ["0_resnet18.pth"]:
            id = net_list[net_number]
            print("current net = ", id)
            print("----------start----------")
            batch_time = 0.0
            data_time = 0.0
            MOG_time_G = 0.0
            MOG_time_O = 0.0

            output_all = None
            y_all = None
            end = time.time()
            ALL = torch.load("/home/yau/zmj/VLDB21/ImageNet-master/ImageNet-master/output/"+id)
            output_all = ALL['output_all']
            y_all = ALL['y_all']
            print("load complete")
            batch_time += (time.time() - end)
            end = time.time()

            '''rate'''
            length = output_all.size()[0]
            new_len = round(length * rate)
            output_all = output_all[:new_len]
            y_all = y_all[:new_len]
            '''rate'''

            '''I-SMS'''
            if output_all.size()[1] > op:
                net1 = nn.Linear(output_all.size()[1], op)
                output_all = net1(output_all)
            '''I-SMS'''

            output_new = nn.Softmax(dim=1)(output_all/2.0).detach().numpy() 
            new_X = output_new

            X_list = [0]*len(class_list_num)
            for i in range(len(y_all)):
                if type(X_list[y_all[i]]) != type(new_X[i]):
                    X_list[y_all[i]] = new_X[i]
                else:
                    a = X_list[y_all[i]]
                    b = new_X[i]
                    c = np.row_stack((a,b))
                    X_list[y_all[i]] = c
            
            '''rate'''
            num = 0
            for i in range(len(X_list)):
                i = i-num
                if type(X_list[i]) != type(new_X[i]):
                    del X_list[i]
                    num += 1
                elif len(X_list[i].shape) == 1:
                    del X_list[i]
                    num += 1
            '''rate'''
            
            # means = []
            # covariances = []
            # weights = []
            # from sklearn.mixture import GaussianMixture
            # for i in range(len(X_list)):
            #     MOG = GaussianMixture(n_components=1, covariance_type='full').fit(X_list[i])
            #     # mean = np.mean(X_list[i],axis=0)
            #     # covariance = np.cov(X_list[i], rowvar=0)
            #     means.append(MOG.means_[0])
            #     covariances.append(MOG.covariances_[0])
            #     weights.append(MOG.weights_[0])
            #     # means.append(mean)
            #     # covariances.append(covariance)
            #     # weights.append(1.0)
            
            
            # MOG_time_G += (time.time() - end)
            # end = time.time()
            # # print(means[0])

            # ave_OLR = get_ave_OVR_new(means,covariances,weights)

            ave_OLR = EMS(X_list)

            MOG_time_O += (time.time() - end)

            # batch_time = 0.0
            # data_time = 0.0
            # MOG_time_G = 0.0
            # MOG_time_O = 0.0
            
            print(id+"  "+str(ave_OLR)+"  "+str(batch_time)+"  "+str(data_time)+
                "  "+str(MOG_time_G)+"  "+str(MOG_time_O))
            
            # print("net_number = ",net_number)
            print("ave_OLR = ",ave_OLR)
            print("\n")

            # break

            print(id+"  "+str(ave_OLR)+"  "+str(batch_time)+"  "+str(data_time)+
                "  "+str(MOG_time_G)+"  "+str(MOG_time_O), flush=True, file=File)


if __name__ == '__main__':
    main()