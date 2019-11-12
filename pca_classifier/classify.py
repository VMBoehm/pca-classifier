import numpy as np
import scipy.linalg as lg
import numpy.linalg as nlg
import os
import pickle as pkl

def get_covariance(R,var,num,N=None,reg=True):
    '''
    get covarinace estimation for specific number of components
    num: number of components
    R: matrix of eigenvectors
    var: array of eigenvalues
    '''
    fl   = len(var)
    var_ = var[0:num]
    R    = R[0:num]
    
    if num<fl:
        sigma2 = np.mean(var[num::])
    else:
        sigma2 = 0.

    C_            = np.dot(R.T,np.dot(np.diag(var_), R))

    if reg:
        if np.any(N)==None:
            print('using internal estimate of recon error')
            C_+=np.eye(len(R.T))*sigma2
        else:
            C_+=N

    Cinv          = lg.inv(C_)
    sign ,logdetC = nlg.slogdet(C_)
    
    return Cinv, logdetC


def get_data_space_log_prob(data,logdet,Cinv,mean,vol=True):
    """
    logdet: ln det C
    Cinv  : C^-1
    data  : data
    mean  : mean(data)
    vol   : boolean, whether to include volume term
    """
    d    = len(data)
    data = data-mean
    Cinv_d = np.einsum('jk,...k->...j',Cinv,data, optimize=True)
    logprob = -0.5*np.einsum('ij,ij->i',data, Cinv_d, optimize=True)
    if vol:
        logprob+=(-0.5*logdet-0.5*d*np.log((2*np.pi)))

    return logprob

def get_latent_space_log_prob(data,cov,n_comp,vol=True):
    '''
    data : data
    cov  : instance of class Covariance 
    n_comp: int, number of components to keep in the pca
    vol : boolean, wether to include volume term
    '''
    z = cov.compress(data,n_comp)
    S = np.diag(cov.vars[0:n_comp])
    print(S)
    sSs = np.einsum('ij,jj,ij->i',z,S**(-1),z,optimize=True)
    logprob = -0.5*sSs
    if vol:
        _, logdet= nlg.slogdet(S)
        logprob+=(-0.5*logdet-0.5*n_comp*np.log(2*np.pi))
    return logprob

