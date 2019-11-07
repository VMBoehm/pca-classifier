import numpy as np

def prepare_data(data, labels, num_classes):
    '''
    orderes data by labels and identifies zero variance pixels
    '''
    ordered_data = []
    masks_in     = []
    masks_out    = []
    for ii in range(num_classes):
        ind     = np.where(labels==ii)[0]
        d       = data[ind]
        mask_in = np.where(np.var(d, axis=0)>0.)[0]
        ordered_data+=[d]
        mask_out= np.where(np.var(d, axis=0)==0.)[0]
        masks_in+=[mask_in]
        masks_out+=[mask_out]
    
    return ordered_data, masks_in, masks_out
