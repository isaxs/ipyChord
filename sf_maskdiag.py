#!/usr/bin/env python
from copy import deepcopy
import numpy as np
from scipy.ndimage import morphology
def sf_maskdiag(img,size=3):
    """
    This file is used to remove the diagonal blind area on the detector.
    Normally, we use it to mask the beam stop holder. In practice, it's
    also very useful to remove the bad spots on the detector.
    """
    res=deepcopy(img)
    tmp=deepcopy(img)
    #generate a ones matrix
    arr=np.ones(size,dtype=np.float)
    #make the diagonal matrix 
    mat=np.diag(arr)
    #get the anti-diagonal matrix
    antimat=np.fliplr(mat)
    tmp['map']=morphology.binary_erosion(img['map'],structure=mat)
    res['map']=morphology.binary_erosion(tmp['map'],structure=antimat)
    
    return res
