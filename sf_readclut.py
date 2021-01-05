#!/usr/bin/env python
from glob import glob
import numpy as np
import os
def sf_readclut(filename):
    """
    this file is used to load the color lookup table
    it only canbe used for lut05.dat file
    """
    path='/home/xuke/Dropbox/ipychord/'
    flist=glob(path+filename)
    #print(flist)
    fid=open(flist[0],'r')
    clt=np.loadtxt(fid,dtype=np.float)
    cltn=np.column_stack((clt[0],clt[1],clt[2]))
    fid.close()
    return cltn
