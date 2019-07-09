import numpy as np
import scipy.linalg as lg
import numpy.linalg as nlg
import os
import pickle as pkl

def get_covariance(R,var,num):
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

    C_            = np.dot(R.T,np.dot(np.diag(var_), R))+np.eye(len(R.T))*sigma2
    Cinv          = lg.inv(C_)
    sign ,logdetC = nlg.slogdet(C_)
    
    return Cinv, logdetC


def get_log_prob(logdet,Cinv,data,mean):
    d    = len(data)
    data = data-mean
    Cinv_d = np.einsum('jk,...k->...j',Cinv,data, optimize=True)
    logprob = -0.5*logdet-0.5*d*np.log((2*np.pi))-0.5*np.einsum('ij,ij->i',data, Cinv_d, optimize=True)
    return logprob


def classify(data,labels,covs, num_classes, num, pca=False):
    acc = []
    nums= []
    for jj in range(num_classes):
        indices = np.where(labels==jj)[0]
        dd      = data[indices]
        logprob = []
        for ii,cov in enumerate(covs):
            if cov.masking:
                d_      = dd[:,cov.mask_in]
            else:
                d_      = np.ones_like(dd)*0.01
                d_[:,cov.mask_in] = dd[:,cov.mask_in]
            if num > d_.shape[1]:
                num = d_.shape[1]
            if pca:
                Cinv, logdetC = get_covariance(cov.pca_R,cov.pca_vars,num)
            else:
                Cinv, logdetC = get_covariance(cov.R,cov.vars,num)
            logprob_= get_log_prob(logdetC,Cinv,d_,cov.mean)
            logprob+=[logprob_]
        acc+=[len(np.where(np.argsort(np.asarray(logprob),axis=0)[-1]==jj)[0])/len(dd)]
    return np.asarray(acc)

def perform_classification(data, labels, modes, masks, num_classes,num_comp,inpath, outpath, pca=True, rerun=False):
    
    outfile = os.path.join(outpath,'results.pkl')
    if os.path.isfile(outfile) and rerun==False:
        results = pkl.load(open(outfile, 'rb'))
    else:
        results = {}
        if pca:
            results['pca']={}
        for mode in modes:
            results[mode] ={}

            for masking in masks:
                if masking:
                    label = 'masked'
                else:
                    label = 'inpainted'
                results[mode][label]={}
                if pca and mode=='ML':
                    results['pca'][label]={}
                covs=[]
                for ii in range(num_classes):
                    if masking:
                        filename = os.path.join(inpath,'cov_estimate_%s_%d_masked.pkl'%(mode,ii))
                    else:   
                        filename = os.path.join(inpath,'cov_estimate_%s_%d.pkl'%(mode,ii))
                    covs+=[pkl.load(open(filename, 'rb'))]
                accs=[]
                nums=[]
                for num in num_comp:
                    print(mode, label, num)
                    acc  = classify(data,labels,covs, num_classes, num=num)
                    accs+=[acc]
                results[mode][label]['accs']=np.asarray(accs)
                if pca:
                    if mode=='ML':
                        accs=[]
                        for num in num_comp:
                            print('pca', num)
                            acc = classify(x_test,targets_test,covs, num_classes, num=num, pca=True)
                            accs+=[acc]
                        results['pca'][label]['accs']=np.asarray(accs)
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                pkl.dump(results, open(outfile, 'wb'))
    return results
