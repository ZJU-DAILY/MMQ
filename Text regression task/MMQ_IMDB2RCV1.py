import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import re
import string
import copy
import time
from sklearn.metrics import accuracy_score,confusion_matrix
import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import jieba
from torchtext import data
from torchtext.vocab import Vectors

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from image_folder import image_folder
from torchvision import transforms
from torchvision.datasets import *
import random
import os
import xlsxwriter
from net import *
import re
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from SMS_methods import get_ave_OVR_new
import datetime

class LSTMNet(nn.Module):
    def __init__(self, vocab_size,embedding_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        embeds = self.embedding(x)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)  
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.lstm(embeds, None)   
        out = self.fc1(r_out[:, -1, :]) 
        return out

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


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("GPU is not available!")
        exit(0)
    else:
        print("GPU is available!")
        device = torch.device("cuda")

    File = open('MOG_LSTM_imdb_2_rcv1_'+
    datetime.datetime.strftime(datetime.datetime.now(),
    '%Y-%m-%d %H:%M:%S')+'.txt', 'w')

    net_list = get_all_files("LSTM")
    
    mytokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=mytokenize, 
                    include_lengths=True, use_vocab=True,
                    batch_first=True, fix_length=400)
    LABEL = data.Field(sequential=False, use_vocab=False, 
                    pad_token=None, unk_token=None)
    text_data_fields = [
        ("labelcode", LABEL), 
        ("cutword", TEXT) 
    ]
    # # traindata,valdata,testdata = data.TabularDataset.splits(
    # traindata,testdata = data.TabularDataset.splits(
    #     path="data/chap6", format="csv", 
    #     train="imdb_train.csv", fields=text_data_fields, 
    #     # validation="imdb_val.csv",
    #     test = "imdb_test.csv", skip_header=True
    # )
    traindata, testdata = data.TabularDataset.splits(
        path="/home/yau/zmj/IJCAI21/LSTM_new", format="csv", 
        train="rcv1_train.csv", fields=text_data_fields, 
        test = "rcv1_test.csv", skip_header=True
    )
    train_data, val_data = traindata.split(split_ratio=0.1)
    traindata = train_data

    vec = Vectors("glove.6B.100d.txt","/home/yau/zmj/IJCAI21/LSTM_new")
    TEXT.build_vocab(traindata,max_size=20000,vectors = vec)
    # TEXT.build_vocab(traindata,max_size=20000,vectors = None)
    LABEL.build_vocab(traindata)

    BATCH_SIZE = 128
    # BATCH_SIZE = 32
    train_iter = data.BucketIterator(traindata,batch_size = BATCH_SIZE)
    
    for net_number in range(len(net_list)):
        net_name = "LSTM/"+net_list[net_number]
        print('load ' + net_name + ' ...')

        c_num = 51
        lstmmodel = torch.load(net_name)
        
        print('load ' + net_name + ' complete')

        # since = time.time()

        lstmmodel.cuda()
        lstmmodel = nn.DataParallel(lstmmodel)
        lstmmodel.eval() 
        batch_time = 0.0
        data_time = 0.0
        MOG_time_G = 0.0
        MOG_time_O = 0.0
        end = time.time()

        for step,batch in enumerate(train_iter):
            X,y0 = batch.cutword[0],batch.labelcode.view(-1)
            X = X.to(device)
            output = lstmmodel(X)
            
            output_new = nn.Softmax(dim=1)(output/2.0).cpu().detach().numpy()
            y0 = y0.numpy()
            if not 'new_X' in dir():
                new_X = output_new
                y = y0
            else:
                new_X = np.row_stack((new_X, output_new))
                y = np.hstack((y, y0))

        batch_time += (time.time() - end)
        end = time.time()

        # time_use = time.time() - since
        # print("Train and val complete in {:.0f}m {:.0f}s".format(
        #     time_use // 60, time_use % 60))

        X_list = [0]*c_num
        for i in range(len(y)):
            if type(X_list[y[i]]) != type(new_X[i]):
                X_list[y[i]] = new_X[i]
            else:
                a = X_list[y[i]]
                b = new_X[i]
                c = np.row_stack((a,b))
                X_list[y[i]] = c
        '''rate'''
        num = 0
        for i in range(len(X_list)):
            i = i-num
            if type(X_list[i]) != type(new_X[i]):
                del X_list[i]
                num += 1
            elif len(X_list[i].shape) < 2:
                del X_list[i]
                num += 1
            # elif X_list[i].shape[0] < 200:
            #     del X_list[i]
            #     num += 1
        '''rate'''

        means = []
        covariances = []
        weights = []
        del new_X

        print("start MOG")
        for i in range(len(X_list)):
            MOG = GaussianMixture(n_components=1, covariance_type='full').fit(X_list[i])
            # means.append(MOG.means_[0][0])
            # covariances.append(MOG.covariances_[0][0][0])
            # weights.append(MOG.weights_[0])
            means.append(MOG.means_[0])
            covariances.append(MOG.covariances_[0])
            weights.append(MOG.weights_[0])

        # time_use = time.time() - since
        # print("Train and val complete in {:.0f}m {:.0f}s".format(
        #     time_use // 60, time_use % 60))

        # print("end MOG")
        
        MOG_time_G += (time.time() - end)
        end = time.time()

        ave_OLR = get_ave_OVR_new(means,covariances,weights)

        MOG_time_O += (time.time() - end)
        
        # time_use = time.time() - since

        print(net_name+"  "+str(ave_OLR)+"  "+str(batch_time)+"  "+str(data_time)+
            "  "+str(MOG_time_G)+"  "+str(MOG_time_O))
            
        print("net_number = ",net_number)
        print("ave_OLR = ",ave_OLR)
        print("\n")

        # break

        print(net_name+"  "+str(ave_OLR)+"  "+str(batch_time)+"  "+str(data_time)+
            "  "+str(MOG_time_G)+"  "+str(MOG_time_O), flush=True, file=File)