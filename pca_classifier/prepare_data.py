import numpy as np

def identify_low_var_pixels(data_set,threshhold=1e-8):
  '''
  dataset: array, data in which to identify low variance pixels
  threshold: min variance that a pixel needs to exceed in order not to be masked
  '''
  var  = np.var(data_set,axis=0)
  mask = np.where(var>threshhold)
  return mask[0]

def mask_low_var_pixels(data_set1, data_set2=None, threshhold=1e-8):
  """
  dataset1: dataset to mask
  dataset2: dataset to infer mask from (if not given, defaults to dataset1)
  threshold: minimum variance value that a pixel needs to exceed in order to be kept
  """

  if np.any(data_set2)==None:
    data_set2=data_set1

  mask = identify_low_var_pixels(data_set2,threshhold)
  masked_data = data_set1[:,mask]

  return masked_data 
