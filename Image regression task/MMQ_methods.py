import sys
from scipy.stats import multivariate_normal
import numpy as np
import math

def get_OVR_new(m_i,m_j,cov_i,cov_j,weights_i,weights_j):
    alpha = 1
    dim = np.shape(cov_i)[0]
    covinv_i = np.linalg.inv(cov_i)
    covinv_j = np.linalg.inv(cov_j)
    while True:
        covdet_i = np.linalg.det(alpha * cov_i)
        covdet_j = np.linalg.det(alpha * cov_j)
        if covdet_j == 0 or covdet_i ==0:
            alpha *= 1000
            continue
        else:
            break
    xdiff_i = (m_i - m_j).reshape((1,dim))
    xdiff_j = (m_j - m_i).reshape((1,dim))
    eii = 1.0
    ejj = 1.0
    eij = np.exp(-0.5*xdiff_j.dot(covinv_i).dot(xdiff_j.T))[0][0]
    eji = np.exp(-0.5*xdiff_i.dot(covinv_j).dot(xdiff_i.T))[0][0]

    ci = np.power(np.abs(covdet_i),0.5)
    cj = np.power(np.abs(covdet_j),0.5)

    ri = cj*eii/(cj*eii+ci*eji) 
    rj = ci*ejj/(cj*eij+ci*ejj)

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
            print("all_OVR = ",all_OVR)
            # print("mu1 = {:.5f}; mu2 = {:.5f}; OVR = {:.5f}; all_OVR = {:.5f}".format(
            # mu1, mu2, OVR, all_OVR))
            num += 1
    # print("all_OVR = ",all_OVR)
    ave_OVR = all_OVR/num
    return ave_OVR

def get_ave_OVR_new_re(means,covariances,weights,n=2):
    all_OVR = 0.0
    num = 0
    for i in range(len(means)):
        mu1 = means[i]
        Sigma1 = covariances[i]
        pi1 = weights[i]
        for j in range(i+1, len(means)):
            alpha = j - i
            alpha = alpha**n
            mu2 = means[j]
            Sigma2 = covariances[j]
            pi2 = weights[j]
            OVR = get_OVR_new(mu1, mu2, Sigma1, Sigma2, pi1, pi2)
            OVR = OVR * alpha
            all_OVR += OVR
            print("all_OVR = ",all_OVR)
            num += alpha
    ave_OVR = all_OVR/num
    return ave_OVR

def EMS(X_list):
    means = []
    covariances = []
    weights = []
    from sklearn.mixture import GaussianMixture
        
    for i in range(len(X_list)):
        MOG = GaussianMixture(n_components=1, covariance_type='full').fit(X_list[i])
        means.append(MOG.means_[0])
        covariances.append(MOG.covariances_[0])
        weights.append(MOG.weights_[0])
    
    ave_OLR = get_ave_OVR_new(means,covariances,weights)

    return ave_OLR

def EMS_re(X_list, n):
    means = []
    covariances = []
    weights = []
    from sklearn.mixture import GaussianMixture
        
    for i in range(len(X_list)):
        MOG = GaussianMixture(n_components=1, covariance_type='full').fit(X_list[i])
        means.append(MOG.means_[0])
        covariances.append(MOG.covariances_[0])
        weights.append(MOG.weights_[0])
    
    ave_OLR = get_ave_OVR_new_re(means,covariances,weights, n)

    return ave_OLR


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