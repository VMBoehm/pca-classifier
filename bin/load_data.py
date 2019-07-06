# functions to load different datasets (mnist, cifar10, random Gaussian data) 

import gzip, zipfile, tarfile
import os, shutil, re, string, urllib, fnmatch
import pickle as pkl
import numpy as np

def _download_mnist(dataset):
    """
    download mnist dataset if not present
    """
    origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
    print('Downloading data from %s' %origin)
    urllib.request.urlretrieve(origin, dataset)


def _download_cifar10(dataset):
    """
    download cifar10 dataset if not present
    """
    origin = ('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    
    print('Downloading data from %s' % origin)
    urllib.request.urlretrieve(origin,dataset)


def _get_datafolder_path():
    """
    returns data path
    """
    #where am I? return full path
    full_path = os.path.abspath('../')
    path = full_path +'/data'
    return path

def load_mnist(dataset=_get_datafolder_path()+'/mnist/mnist.pkl.gz'):
    """
    load mnist dataset
    """

    if not os.path.isfile(dataset):
        datasetfolder = os.path.dirname(dataset)
        if not os.path.exists(datasetfolder):
            print('creating ', datasetfolder)
            os.makedirs(datasetfolder)
        _download_mnist(dataset)

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pkl.load(f, encoding='latin1')
    f.close()
    x_train, targets_train = train_set[0], train_set[1]
    x_valid, targets_valid = valid_set[0], valid_set[1]
    x_test, targets_test = test_set[0], test_set[1]
    #omitting validation set for consistency
    return x_train, targets_train, x_test, targets_test


def load_cifar10(dataset=_get_datafolder_path()+'/cifar10/cifar-10-python.tar.gz'):
    """   
    load cifar10 dataset
    """

    datasetfolder = os.path.dirname(dataset)
    print(datasetfolder)
    if not os.path.isfile(dataset):
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_cifar10(dataset)
        with tarfile.open(dataset) as tar:
            tar.extractall(path=datasetfolder)
        
    train_x = np.empty((0,32*32*3))
    for i in range(5):
        batchName = os.path.join(datasetfolder,'cifar-10-batches-py/data_batch_{0}'.format(i + 1))
        with open(batchName, 'rb') as f:
            d = pkl.load(f, encoding='latin1')
            data = d['data']
            label= d['labels']
            train_x = np.vstack((train_x,data))
            try:
                train_y = np.append(train_y,np.asarray(label))
            except:
                train_y = np.asarray(label)
                
    batchName = os.path.join(datasetfolder,'cifar-10-batches-py/test_batch')
    with open(batchName, 'rb') as f:
        d = pkl.load(f, encoding='latin1')
        data = d['data']
        label= d['labels']
        test_x = data
        test_y = np.asarray(label)
                
    return train_x, train_y, test_x, test_y
