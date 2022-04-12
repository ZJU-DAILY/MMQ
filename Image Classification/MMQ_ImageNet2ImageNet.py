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

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module # that I actually define.
    def forward(self, x):
        return self.module(x)

def main():
    if not torch.cuda.is_available():
        print("GPU is not available!")
        return
    else:
        print("GPU is available!")
        device = torch.device("cuda")

    root = "/home/yau/zmj/VLDB21/ImageNet-master/ImageNet-master/source_models"
    File = open('MOG_ImageNet_rank.txt', 'w')
        
    net_list = get_all_files(root)
    class_list_num = list(range(700,1000))
    rate = 0.5
    train_loader, val_loader, test_loader = data_loader(class_list_num, "/ImageNet", 1024, 20, rate)
    
    for net_number in range(0, len(net_list)):
    # for id in ["nothing.pth_resnet18.pth"]:
        print("root = ", root)
        id = net_list[net_number]
        print("current net = ", id)
        print("----------start----------")
        source = torch.load(root+"/"+id)
        class_num = source['class_num']
        
        ''' -----------resnet18----------- '''
        if id == "everything.pth_resnet18.pth":
            myconvnet = models.resnet18(pretrained=True)
            class_num = 1000
        elif id == "nothing.pth_resnet18.pth":
            myconvnet = models.resnet18(pretrained=False, num_classes=300)
            class_num = 300
        else:
            myconvnet = models.resnet18(pretrained=False, num_classes=class_num)
            myconvnet = WrappedModel(myconvnet)
            myconvnet.load_state_dict(source['state_dict'])
        myconvnet = myconvnet.to(device)
        # myconvnet.cpu()
        # myconvnet.to(device)
        print("model loading complete")
        print("class_num = ",class_num)
        ''' -----------infer----------- '''
        myconvnet.eval()
        batch_time = 0.0
        data_time = 0.0
        MOG_time_G = 0.0
        MOG_time_O = 0.0

        output_all = None
        y_all = None
        end = time.time()
        for step, (b_x, b_y) in enumerate(train_loader):
            # b_x = b_x.cuda(async=True)
            b_x = b_x.to(device)
            # print(b_y[0:10])
            with torch.no_grad():
                myconvnet.eval()
                # measure data loading time
                data_time += (time.time() - end)
                end = time.time()

                # compute output
                output = myconvnet(b_x)
                b_x = b_x.cpu()
                output = output.cpu()

                if type(output_all) == type(None):
                    output_all = output
                    y_all = b_y
                else:
                    output_all = torch.cat((output_all, output),0)
                    y_all = torch.cat((y_all, b_y),0)

                # measure elapsed time
                batch_time += (time.time() - end)
                end = time.time()

        # torch.save({
        #         'output_all': output_all,
        #         'y_all': y_all},
        #         "/home/yau/zmj/VLDB21/ImageNet-master/ImageNet-master/output_PR18/oa_"+id)
        # print("save complete")

        # output_new = nn.Softmax(dim=1)(output_all/2.0).detach().numpy() 
        # new_X = output_new

        # X_list = [0]*len(class_list_num)
        # for i in range(len(y_all)):
        #     if type(X_list[y_all[i]]) != type(new_X[i]):
        #         X_list[y_all[i]] = new_X[i]
        #     else:
        #         a = X_list[y_all[i]]
        #         b = new_X[i]
        #         c = np.row_stack((a,b))
        #         X_list[y_all[i]] = c
        
        # means = []
        # covariances = []
        # weights = []
        # from sklearn.mixture import GaussianMixture
        # for i in range(len(X_list)):
        #     MOG = GaussianMixture(n_components=1, covariance_type='full').fit(X_list[i])
        #     means.append(MOG.means_[0])
        #     covariances.append(MOG.covariances_[0])
        #     weights.append(MOG.weights_[0])
        
        
        # MOG_time_G += (time.time() - end)
        # end = time.time()
        # # print(means[0])

        # ave_OLR = get_ave_OVR_new(means,covariances,weights)

        # MOG_time_O += (time.time() - end)

        # batch_time = 0.0
        # data_time = 0.0
        # MOG_time_G = 0.0
        # MOG_time_O = 0.0
        
        print(id+"  "+ str(batch_time)+"  "+str(data_time)+
            "  "+str(MOG_time_G)+"  "+str(MOG_time_O))
        # break

        print(id+"  "+ str(batch_time)+"  "+str(data_time)+
            "  "+str(MOG_time_G)+"  "+str(MOG_time_O), flush=True, file=File)


if __name__ == '__main__':
    main()