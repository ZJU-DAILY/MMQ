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
from image_folder import image_folder
from torchvision import transforms
import random
import os
import xlsxwriter
import re
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import datetime
import torchvision.models as models
from EMS_methods import *

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

def dataset(root):
    root = root + "/"
    images = None
    for i in range(1,536):
        bathPath = root + str(i)+"_bathroom.jpg"
        badPath = root + str(i)+"_bedroom.jpg"
        froPath = root + str(i)+"_frontal.jpg"
        kitPath = root + str(i)+"_kitchen.jpg"
        housePaths = [bathPath, badPath, froPath, kitPath]
        inputImages = []
        outputImage = np.zeros((64, 64, 3), dtype="uint8")
        for housePath in housePaths:
			# load the input image, resize it to be 32 32, and then
			# update the list of input images
            image = cv2.imread(housePath)
            image = cv2.resize(image, (16, 16, ))
            inputImages.append(image)
        outputImage[0:16, 0:16] = inputImages[0]
        outputImage[0:16, 16:32] = inputImages[1]
        outputImage[16:32, 16:32] = inputImages[2]
        outputImage[16:32, 0:16] = inputImages[3]
        outputImage = np.swapaxes(outputImage,0,2)
        outputImage = torch.Tensor(outputImage) / 255.0
        if type(images) == type(None):
            images = outputImage.unsqueeze(0)
        else:
            outputImage = outputImage.unsqueeze(0)
            images = torch.cat((images, outputImage),0)
    labels = []    
    for line in open(root + "HousesInfo.txt"):
        price = float(line.split()[-1])
        labels.append(price)
    maxPrice = max(labels)
    labels = torch.Tensor(labels) / maxPrice
    return images, labels

def main():
    if not torch.cuda.is_available():
        print("GPU is not available!")
        return
    else:
        print("GPU is available!")
        device = torch.device("cuda")

    root = "/home/yau/zmj/IJCAI21/regression/VGG11_source_models"
    File = open('/home/yau/zmj/IJCAI21/regression/MOG_rank_end.txt', 'w')
        
    net_list = get_all_files(root)

    b_x_all, b_y_all = dataset("/home/yau/zmj/IJCAI21/regression/Houses Dataset")
    b_x_all = b_x_all.split(32, 0)
    b_y_all = b_y_all.split(32, 0)
    
    for net_number in range(0, len(net_list)):
        print("root = ", root)
        id = net_list[net_number]
        print("current net = ", id)
        print("----------start----------")
        source = torch.load(root+"/"+id)
        
        myconvnet = models.vgg11_bn(pretrained=False, num_classes=source['class_num'])
        myconvnet.load_state_dict(source["state_dict"])
        myconvnet = myconvnet.to(device)
        # myconvnet = nn.DataParallel(myconvnet)

        myconvnet.eval()
        batch_time = 0.0
        data_time = 0.0
        MOG_time_G = 0.0
        MOG_time_O = 0.0

        batch_num = len(b_x_all)
        train_batch_num = round(batch_num * 0.9)
        output_all = None
        y_all = None
        end = time.time()
        for step in range(batch_num):
            b_x = b_x_all[step]
            b_y = b_y_all[step]
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            if step < train_batch_num:
                myconvnet.eval() 
                output = myconvnet(b_x)
                if type(output_all) == type(None):
                    output_all = output
                    y_all = b_y
                else:
                    output_all = torch.cat((output_all, output),0)
                    y_all = torch.cat((y_all, b_y),0)
        y_all = y_all.cpu()
        y_all = y_all.detach().numpy()
        output_all = output_all.cpu()
        output_all = output_all.detach().numpy()
        y_index = np.argsort(y_all)

        batch_time += (time.time() - end)
        end = time.time()

        output_all_sort = None
        for i, index in enumerate(y_index):
            new_X = output_all[index]
            if i == 0:
                output_all_sort = output_all[index]
            else:
                a = output_all_sort
                b = new_X
                c = np.row_stack((a,b))
                output_all_sort = c
        print(len(output_all_sort))
        X_list = np.split(output_all_sort, 10)

        means = []
        covariances = []
        weights = []
        from sklearn.mixture import GaussianMixture
            
        for i in range(len(X_list)):
            MOG = GaussianMixture(n_components=1, covariance_type='full').fit(X_list[i])
            means.append(MOG.means_[0])
            covariances.append(MOG.covariances_[0])
            weights.append(MOG.weights_[0])

        MOG_time_G += (time.time() - end)
        end = time.time()

        ave_OLR = get_ave_OVR_new_re(means,covariances,weights,2)

        MOG_time_O += (time.time() - end)

        # ave_OLR = EMS(X_list)

        # ave_OLR = EMS_re(X_list, 1)

        print(id+"  "+str(ave_OLR)+"  "+str(batch_time)+"  "+str(data_time)+
            "  "+str(MOG_time_G)+"  "+str(MOG_time_O))
            
        print("net_number = ",net_number)
        print("ave_OLR = ",ave_OLR)
        print("\n")

        # break

        print(id+"  "+str(ave_OLR)+"  "+str(batch_time)+"  "+str(data_time)+
            "  "+str(MOG_time_G)+"  "+str(MOG_time_O), flush=True, file=File)


if __name__ == '__main__':
    main()