#!/usr/bin/env python
#this file is used to read the fio file
from numpy import loadtxt
def sf_fioread(filename,skips):
    #check the filename if there is an extension '.fio'
    if not '.fio' in filename:
       AssertionError('The fio file has  no extension .fio')
    fioid=open(filename,'r')
    pos,exp,ipetra=loadtxt(fioid,skiprows=skips,usecols=(0,2,3),unpack=True)
    return pos,exp,ipetra
