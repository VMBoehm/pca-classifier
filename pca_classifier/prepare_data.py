import numpy as np

def identify_low_var_pixels(data_set,threshhold=1e-8):
  '''
  dataset: array, data in which to identify low variance pixels
  threshold: min variance that a pixel needs to exceed in order not to be masked
  '''
  var     = np.var(data_set,axis=0)
  mask_in = np.where(var>threshhold)
  mask_out= np.where(var<=threshhold)
  return mask_in[0], mask_out[0]

def mask_low_var_pixels(data_set1, data_set2=None, threshhold=1e-8):
  """
  dataset1: dataset to mask
  dataset2: dataset to infer mask from (if not given, defaults to dataset1)
  threshhold: minimum variance value that a pixel needs to exceed in order to be kept
  """

  if np.any(data_set2)==None:
    data_set2=data_set1

  mask_in, mask_out = identify_low_var_pixels(data_set2,threshhold)
  masked_data = data_set1[:,mask_in]

  return masked_data, None

def inpaint_low_var_pixels(data_set1, data_set2=None, threshhold=1e-8,inpaint_val=1e-1):
  """
  dataset1: dataset to mask
  dataset2: dataset to infer mask from (if not given, defaults to dataset1)
  threshhold: pixels with a variance lower than that get inpainted with white noise
  noise_level: noise level for inpainting, is constant so that it doesnt mess up our compression
  """
  if np.any(data_set2)==None:
    data_set2=data_set1

  mask_in,mask_out  = identify_low_var_pixels(data_set2,threshhold)
  shape             = data_set1[:,mask_out].shape
  noise             = np.ones(shape)*inpaint_val
  inpainted_data    = np.zeros(data_set1.shape)
  inpainted_data[:,mask_in] = data_set1[:,mask_in]
  inpainted_data[:,mask_out] = noise
  return inpainted_data, mask_out


