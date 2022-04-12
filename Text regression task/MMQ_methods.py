import sys
from scipy.stats import multivariate_normal
import numpy as np
import math
import time

def get_OVR_new(means_i,means_j,covariances_i,covariances_j,weights_i,weights_j):
    Gi = multivariate_normal(mean = means_i, cov = covariances_i, allow_singular = True)
    Gj = multivariate_normal(mean = means_j, cov = covariances_j, allow_singular = True)
    
    i_m = means_i
    j_m = means_j
    pi = weights_i*Gi.pdf(i_m)
    pj = weights_j*Gj.pdf(j_m)
    pdfi = weights_j*Gj.pdf(i_m) + weights_i*Gi.pdf(i_m)
    pdfj = weights_i*Gi.pdf(j_m) + weights_j*Gj.pdf(j_m)
    
    ri = pi/pdfi
    rj = pj/pdfj

    return ((ri+rj)/2 - 0.5)*2.0

def get_ave_OVR_new(means,covariances,weights):
    all_OVR = 0.0
    num = 0
    for i in range(len(means)):
        mu1 = means[i]
        Sigma1 = covariances[i]
        pi1 = weights[i]
        for j in range(i+1, len(means)):
            mu2 = means[j]
            Sigma2 = covariances[j]
            pi2 = weights[j]
            OVR = get_OVR_new(mu1, mu2, Sigma1, Sigma2, pi1, pi2)
            all_OVR += OVR
            # print("all_OVR = ",all_OVR)
            # print("mu1 = {:.5f}; mu2 = {:.5f}; OVR = {:.5f}; all_OVR = {:.5f}".format(
            # mu1, mu2, OVR, all_OVR))
            num += 1
    # print("all_OVR = ",all_OVR)
    ave_OVR = all_OVR/num
    return ave_OVR

if __name__ == '__main__':
    m = 5.0
    step = 0.1
    num = 0
    while m>=0.0:
        olr = get_OVR_new(m,0.0,2.0,3.0,1.0,1.0)
        
        # x = np.arange(-10, 10, 0.01)
        # y1 = norm_pdf(x, 0.0, 2.0, 1.0)
        # y2 = norm_pdf(x, m, 3.0, 1.0)
        # plt.plot(x,y1,c='b',lw=0.3) 
        # plt.plot(x,y2,c='r',lw=0.3)
        # plt.plot(x,y1+y2,c='k',lw=0.3)
        # plt.savefig("test/"+str(1)+".png")
        # plt.close()

        m -= step
        print(olr)