from scipy.stats import ortho_group
import scipy.linalg as lg
import numpy.linalg as nlg
import numpy as np


def create_Gaussian_data(eigs, means, data_num, rand_state=4108):
    data_dim = len(eigs)
    R    = ortho_group.rvs(data_dim,random_state=rand_state)
    cov  = np.dot(np.dot(R,np.diag(eigs)),R.T)
    L    = lg.cholesky(cov,lower=True)
    data = np.random.randn((data_dim*data_num)).reshape((data_num, data_dim))
    data = np.einsum('ij,kj->ki',L,data)+means
    return data, cov
