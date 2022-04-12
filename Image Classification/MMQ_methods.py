import sys
from scipy.stats import multivariate_normal
import numpy as np
import math
import scipy

def Gaussian(x,mean,cov):
    dim = np.shape(cov)[0]
    covdet = np.linalg.det(cov)
    covinv = np.linalg.inv(cov)
    xdiff = (x - mean).reshape((1,dim))
    prob = 1.0/(np.power(np.power(2*np.pi,dim)*np.abs(covdet),0.5))*\
            np.exp(-0.5*xdiff.dot(covinv).dot(xdiff.T))[0][0]
    return prob


def det(array:list) -> int:
	"""
	type array : List[List[float]]
	"""
	assert len(array) == len(array[0])
	if len(array) == 1:
		return array[0][0]
	s = 0
	for i in range(len(array)):
		A = [array[j][1:] for j in range(len(array)) if j != i]
		print(A)
		if i % 2:
			s -= array[i][0] * det(A)
		else:
			s += array[i][0] * det(A)
	return s

def get_OVR_new(m_i,m_j,cov_i,cov_j,weights_i,weights_j):
    alpha = 1
    dim = np.shape(cov_i)[0]
    covinv_i = np.linalg.inv(cov_i)
    covinv_j = np.linalg.inv(cov_j)
    xdiff_i = (m_i - m_j).reshape((1,dim))
    xdiff_j = (m_j - m_i).reshape((1,dim))
    eii = 1.0
    ejj = 1.0
    eij = np.exp(-0.5*xdiff_j.dot(covinv_i).dot(xdiff_j.T))[0][0]
    eji = np.exp(-0.5*xdiff_i.dot(covinv_j).dot(xdiff_i.T))[0][0]

    if eij==0 and eji==0:
        return 1.0

    while True:
        covdet_i = np.linalg.det(alpha * cov_i)
        covdet_j = np.linalg.det(alpha * cov_j)
        if covdet_j == 0 or covdet_i ==0:
            alpha *= 1000
            continue
        else:
            break
    if np.isinf(covdet_j) or np.isinf(covdet_i):
        lci = np.linalg.slogdet(cov_i)[1]
        lcj = np.linalg.slogdet(cov_j)[1]
        delta = abs(lci-lcj)
        if lci>lcj:
            lci = delta
            lcj = 0.0
        else:
            lcj = delta
            lci = 0.0
        covdet_i = np.exp(lci)
        covdet_j = np.exp(lcj)

    ci = np.power(np.abs(covdet_i),0.5)
    cj = np.power(np.abs(covdet_j),0.5)

    ri = cj*eii/(cj*eii+ci*eji) 
    rj = ci*ejj/(cj*eij+ci*ejj)

    return ((ri+rj)/2 - 0.5)*2.0

# def get_OVR_new(means_i,means_j,covariances_i,covariances_j,weights_i,weights_j):
#     Gi = multivariate_normal(mean = means_i, cov = covariances_i, allow_singular = True)
#     Gj = multivariate_normal(mean = means_j, cov = covariances_j, allow_singular = True)
    
#     i_m = means_i
#     j_m = means_j
#     # pi = weights_i*Gaussian(i_m,means_i,covariances_i) 
#     pi = Gi.pdf(i_m)
#     # pj = weights_j*Gaussian(j_m,means_j,covariances_j) 
#     pj = Gj.pdf(j_m)
#     pdfi = weights_j*Gj.pdf(i_m) + weights_i*Gi.pdf(i_m)
#     pdfj = weights_i*Gi.pdf(j_m) + weights_j*Gj.pdf(j_m)
#     # pdfi = weights_j*Gaussian(i_m,means_j,covariances_j) + weights_i*Gaussian(i_m,means_i,covariances_i)
#     # pdfj = weights_i*Gaussian(j_m,means_i,covariances_i) + weights_j*Gaussian(j_m,means_j,covariances_j)
    
#     ri = pi/pdfi
#     rj = pj/pdfj

#     return ((ri+rj)/2 - 0.5)*2.0

def get_ave_OVR_new(means,covariances,weights):
    all_OVR = 0.0
    num = 0
    flag = False
    for i in range(len(means)):
        mu1 = means[i]
        Sigma1 = covariances[i]
        pi1 = weights[i]
        for j in range(i+1, len(means)):
            mu2 = means[j]
            Sigma2 = covariances[j]
            pi2 = weights[j]
            OVR = get_OVR_new(mu1, mu2, Sigma1, Sigma2, pi1, pi2)
            if not np.isnan(OVR):
                all_OVR += OVR
                # print("all_OVR = ",all_OVR)
                # print("mu1 = {:.5f}; mu2 = {:.5f}; OVR = {:.5f}; all_OVR = {:.5f}".format(
                # mu1, mu2, OVR, all_OVR))
                num += 1
            else:
                print("n", end=" ")
                flag = True
                return 10.0
    # print("all_OVR = ",all_OVR)
    if num == 0:
        return 0.0
    ave_OVR = all_OVR/num
    if flag:
        ave_OVR += 10.0
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

def EMS_re(X_list):
    means = []
    covariances = []
    weights = []
    from sklearn.mixture import GaussianMixture
        
    for i in range(len(X_list)):
        MOG = GaussianMixture(n_components=1, covariance_type='full').fit(X_list[i])
        means.append(MOG.means_[0])
        covariances.append(MOG.covariances_[0])
        weights.append(MOG.weights_[0])
    
    ave_OLR = get_ave_OVR_new_re(means,covariances,weights)

    return ave_OLR

def EMS_weight_update(X_list):
    from sklearn.mixture import GaussianMixture
    ave_OLR_all = 0.0
    for i in range(len(X_list)):
        X_this = X_list[i]
        X_others = None
        for j in range(len(X_list)):
            if i == j:
                continue
            if type(X_others) == type(None):
                X_others = X_list[j]
            else:
                a = X_others
                b = X_list[j]
                c = np.row_stack((a,b))
                X_others = c

        MOG_this = GaussianMixture(n_components=1, covariance_type='full').fit(X_this)
        MOG_others = GaussianMixture(n_components=1, covariance_type='full').fit(X_others)

        means_this = MOG_this.means_
        covariances_this = MOG_this.covariances_
        weights_this = MOG_this.weights_

        means_others = MOG_others.means_
        covariances_others = MOG_others.covariances_
        weights_others = MOG_others.weights_/sum(MOG_others.weights_)

        all_OLR = 0.0
        num = 0.0
        mu1 = means_this[0]
        Sigma1 = covariances_this[0]
        pi1 = weights_this[0]
        for q in range(len(means_others)):
            mu2 = means_others[q]
            Sigma2 = covariances_others[q]
            pi2 = weights_others[q]
            alpha = pi1 * pi2
            OLR = alpha * get_OVR_new(mu1, mu2, Sigma1, Sigma2, 1.0, 1.0)
            all_OLR += OLR
            # print("all_OLR = ",all_OLR)
            num += alpha
        ave_OLR_all += all_OLR/num
        print("ave_OLR_all = ", ave_OLR_all)
    ave_OLR = ave_OLR_all/len(X_list)
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