from sklearn.covariance import LedoitWolf, EmpiricalCovariance, OAS
import scipy.linalg as lg
import numpy.linalg as nlg
from sklearn.decomposition import PCA
import os
import numpy as np
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


def load_covariance(path,dataset,mode):
    
    filename = os.path.join(path,'cov_estimate_%s_%s.pkl'%(dataset,mode))

    if os.path.isfile(filename):
        cov = pkl.load(open(filename,'rb'))
    else:   
        cov = None
        raise ValueError('%s doe not exist'%filename)

    return cov


class CovarianceEstimator():
    """
    class for covariance estimation and decomposition
    """

    def __init__(self,mode):
        """
        mode: shich covariance estimation method to use
        dataset: name of the dataset
        """
        assert(mode in ['ML','OAS', 'LW','NERCOME', 'TRUE'])
        self.mode = mode

    def decompose(self):
        """
        does a svd decompositiom
        """
        # do svd for numerical stability (ensuring var>=0)
        U,s,V     = lg.svd(self.cov)
        indices   = np.argsort(s)[::-1]
        self.vars = s[indices]
        self.R    = V[indices]
        if len(np.where(self.vars==0.)[0])>0:
            print('covariance estimate contains singular eigenvalues')
        return True

    def compress(self,data,n_comp):

        data      = data-self.mean
        comp_data = np.einsum('ij,kj->ki',self.R[:n_comp],data,optimize=True)

        return comp_data

    def decompress(self,comp_data):

        n_comp = comp_data.shape[-1]
        decomp_data = np.einsum('ij,ki->kj',self.R[:n_comp],comp_data,optimize=True)
        decomp_data+=self.mean

        return decomp_data

    def get_N(self,data,n_comp):
        recon = self.decompress(self.compress(data,n_comp))
        rms2  = np.mean((data-recon)**2,axis=0)
        N     = np.diag(rms2)
        return N

    def dist(self,cov1,cov2=None):
        """
        distance between two estimates used for the nercome estimator
        """
        if np.any(cov2 == None):
            cov2=self.cov
        A = cov1-cov2
        dist = np.trace(np.dot(A,A.T))
        return dist
    
    
    def nercome_estimator(self,data,splits=None,num_esti=None):

        nn      = len(data)
        ddim    = data.shape[1]
        if splits== None:
            if ddim < 200:
                splits = [0.33,0.4,0.45,0.5,0.55,0.66,0.7,0.75,0.8]
            else:
                splits = [0.66]
        if num_esti== None:
            num_esti  = min(nn//2,100)

        minQs      = -1
        best_split = 0.
        best_esti  = np.zeros((ddim,ddim))
        for split_frac in splits:
            print('nercome estimation with split %.2f, #samples %d'%(split_frac,num_esti))
            split    = np.int(split_frac*nn)
            cov      = np.zeros((ddim,ddim))
            cov_esti = np.zeros((ddim,ddim))
            for ii in range(num_esti):
                np.random.shuffle(data)
                data1 = data[0:split]
                data2 = data[split::]
                cov1     = EmpiricalCovariance().fit(data1).covariance_
                w1,v1    = lg.eigh(cov1)
                del cov1, w1
                cov2     = EmpiricalCovariance().fit(data2).covariance_
                diags    = np.diag(np.dot(np.dot(v1.T,cov2),v1))
                esti     = np.dot(np.dot(v1,np.diag(diags)),v1.T)
                cov+=cov2/num_esti
                del cov2
                cov_esti+=esti/num_esti

            Q = self.dist(cov_esti, cov)
            if minQs==-1 or Q<minQs:
                minQs=Q
                best_split=split_frac
                best_esti = cov_esti
 
        return best_esti
        

    def fit(self,data,dataset): 
        """
        data: array, data to fit cov on
        dataset: string, nametag of the data (e.g. 'mnist')
        """

        self.dataset = dataset
        self.mean    = np.expand_dims(data.mean(axis=0),0)

        if self.mode =='ML':
            self.cov = EmpiricalCovariance().fit(data).covariance_
        elif self.mode =='OAS':
            self.cov = OAS().fit(data).covariance_
        elif self.mode =='LW':
            self.cov = LedoitWolf().fit(data).covariance_
        elif self.mode =='NERCOME':
            self.cov = self.nercome_estimator(data)
        else: 
            raise ValueError

        return True
    
    def compute_logdet(self):
        #numerically unstable 
        sign ,self.logdetC = nlg.slogdet(self.cov)
        return True

    def compute_inverse(self):
        #full thing is often note invertible, use decomposition
        self.Cinv = lg.inv(self.cov)
        return True
    
    def save(self, path):

        if not os.path.exists(path):
            os.makedirs(path)
        
        self.filename = os.path.join(path,'cov_estimate_%s_%s.pkl'%(self.dataset,self.mode))
        pkl.dump(self, open(self.filename,'wb'))
        
        return self.filename
