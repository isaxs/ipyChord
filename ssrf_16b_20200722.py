#!/usr/bin/env python
#This package is used to evaluate the data collected at SSRF on 20200722
##############################################################################
import fabio
import pyFAI
from mydict import mydict
from copy import deepcopy
from CDF_2D import *
import scipy.ndimage as ndimage
import numpy as np
import scipy.interpolate as interpolate
from glob import glob
from misc import cread_mc,cread,cplot,cwrite
from sf_show import sf_show
from wf_mapauto import wf_mapauto
import datetime
import time
from sf_show import sf_show
from convertToCartesianImage import convertToCartesianImage
from projection import sf_vp
from CDF_2D import *
from i0make import i0rbf,i0meridian,i0horizon
from iBorder import borderfitpolyN
from makegs import makegs
##############################################################################
#the code below is to evaluate the saxs pattern collected at ssrf on 2020-07-22.
##############################################################################
def calc_cdf(series,maxit=2):
    """
    Caculate the long spacing between the hard domains.
    """
    #get the file list
    flist=sorted(glob(series+'*harm.pkl'))
    #len
    flen=len(flist)
    for i in range(flen):
        #read the file
        res=pklread(flist[i])
        #cut the region of beamstop
        bswin=cutwin(res,140,140)
        print('##########Extrapolate into the beamstop#############')
        bswinf=i0horizon(bswin,row=100,order=4,keeporig=1)
        res=fillfit(res,bswinf)
        sf_show(res)
        #project it onto the s1-s3 plane
        res=fibproj(res)
        sf_show(res,win=1,auto=1,log=1)
        print('=====Start computation of interference function=====')
        gs=ference(res,maxit=maxit)
        sf_show(gs,win=2)
        print('==G(s) Done')
        print('=====Start computation of chord distribution====')
        cdf=chord(gs,4,fast=None,outputorig=1) #triple zeros padding
        print('The width and height of cdf is '+str(cdf['height']))
        cdf=cutwin(cdf,4096,4096)
        sf_show(cdf,log=1,win=3,neg=1)
        print('== CDF Done ==')

def calc_Q_relt(series):
    """
    Calculate the scattering invariant. 
    """
    #get the file list
    flist=sorted(glob(series+'*harm.pkl'))
    #number of files
    flen=len(flist)
    cur=np.zeros((flen,2),dtype=np.float)
    for i in range(flen):
        #read the pkl
        res=pklread(flist[i])
        print(flist[i])
        #cut the region of beamstop
        bswin=cutwin(res,140,140)
        print('##########Extrapolate into the beamstop#############')
        bswinf=i0horizon(bswin,row=100,order=4,keeporig=1)
        res=fillfit(res,bswinf)
        #calculate the Q
        vp=sf_vp(res)
        cplot(vp)
        #store the vp curve
        vpfn=flist[i][0:-8]+'vp.dat'
        cwrite(vp,vpfn)
        Q=2*np.trapz(vp[:,1],vp[:,0])
        curx=res['strainT']
        cur[i,0],cur[i,1]=curx,Q
        print(curx,Q)

    return cur
##############################################################################
def tmp_boxlen(series):
    """
    This func is used to correct the boxlen in the harm file.
    """
    flist=sorted(glob(series+'*harm.pkl'))
    flen=len(flist)
    for i in range(flen):
        harm=pklread(flist[i])
        harm['boxlen'][0]=harm['boxlen'][0]/10
        harm['boxlen'][1]=harm['boxlen'][0]
        pklwrite(harm,flist[i])
def tmp_vp(series):
    """
    This file is used to correct the boxlen in the harm file
    """
    flist=sorted(glob(series+'*_vp.dat'))
    flen=len(flist)
    for i in range(flen):
        vp=cread(flist[i])
        vp[:,0]=vp[:,0]/10
        cwrite(vp,flist[i])
def tmp_harm_svg():
    """
    This func is used to convert the harm pattern to.
    """
##############################################################################
def insert_strstr(series,ssfn):
    """
    This func is used to inser the stress-strain data into the hard file.
    """
    #file list
    flist=sorted(glob(series+'*harm.pkl'))
    flen=len(flist)
    #read the curve
    res=cread(ssfn)
    tmp=np.loadtxt(ssfn,dtype='float',unpack=True)

    for i in range(flen):
        #read the pattern
        harm=pklread(flist[i])
        print(flist[i])
        et=(i-1)*20+10
        if et < 0:
            et=0
        #get the stress-strain data
        harm['elapsedt']=et
        #true strain
        harm['strainT']=tmp[1][et]
        #true stress
        harm['stressT']=tmp[2][et]
        #convert the boxlen (\mum) to s
        harm['distance']=harm['distance']*1E6
        harm['boxlen'][0]=harm['boxlen'][0]*1E9
        harm['boxlen'][0]=harm['boxlen'][0]/(harm['distance']*\
                harm['wavelength'])
        harm['boxlen'][1]=harm['boxlen'][0]
        #print harm['elapsedt']
        #print harm['strainT']
        #print harm['stressT']
        #print harm['boxlen']
        pklwrite(harm,flist[i])
##############################################################################
def spat_normalize(series,bg,mask,N=2048,rotagl=-85,rds=750):
    """
    This func is used to normalize the saxs pattern.
    """
    #get the file list
    flist=sorted(glob(series+'*.pkl'))
    flen=len(flist)
    for i in range(flen):
        #the file
        pkl=pklread(flist[i])
        print(flist[i])
        pkl['map']=pkl['map'].astype(np.float)
        pkl=fillspots(pkl)
        #calculate the absorption factor emut
        emut=pkl['iexpt']*bg['ibeam']/(pkl['ibeam']*bg['iexpt'])
        pkl['emut']=emut
        print(emut)
        #subtract the background scattering
        pkl['map']=pkl['map']/emut-bg['map']
        pkl['map']=pkl['map']*mask['map']
        #paste the pattern on a large array
        arr=np.zeros((N,N),dtype=np.float)
        h,w=pkl['height'],pkl['width']
        arr[0:h,0:w]=pkl['map']
        pkl['map']=arr
        pkl['height'],pkl['width']=N,N
        
        #center the pattern
        cenx,ceny=pkl['center'][0],pkl['center'][1]
        shiftx,shifty=N/2-cenx,N/2-ceny
        pkl=shiftxy(pkl,[shifty,shiftx])
        pkl['center']=[N/2,N/2]
        #as the tensile machine is tilted about the equator by several degree
        #we need to tilt the detector by several degree.
        pkl=azimrot(pkl,rotagl)
        #sf_show(pkl,log=1)
        #harmonize the pattern
        harm=flipharmony(pkl)
        #mask the circle
        cen_x,cen_y=pkl['center'][0],pkl['center'][1]
        harm=killcircleout(harm,cen_x,cen_y,rds)
        harm=cutwin(harm,width=1500,height=1500)
        #store the harm file
        hfn=flist[i][:-4]+'_harm.pkl'
        print(hfn)
        pklwrite(harm,hfn)
        sf_show(harm)
def spat_readseries(fns,ionfn,period=20):
    """
    This file is used to read the saxs pattern series and the ion values.
    """
    #get the file list
    flist=sorted(glob(fns+'*.cbf'))
    flen=len(flist)
    #read the ion chamber file
    arr=ionchamber_read_20200722_strain(ionfn)
    for i in range(flen):
        cbf=scbfread(flist[i])
        count=i*20+10
        #match the count with the elapsed time in the arr
        pos=np.where(arr[:,0].astype(int) == count)
        #print pos
        #goto the specified lines
        cbf['ibeam']=arr[pos[0][0],1]
        cbf['iexpt']=arr[pos[0][0],2]
        #nfn
        nfn=flist[i][:-3]+'pkl'
        #print nfn
        #print flist[i],cbf['ibeam'],cbf['iexpt']
        pklwrite(cbf,nfn)
    
def spat_readstatic(fn,ionfn):
    """
    This func is used to read all the parameters into the pattern.
    """
    #get the file list
    cbf=scbfread(fn)
    #read the ion value
    ibeam,iexpt=ionchamber_read_20200722_static(ionfn)

    cbf['ibeam']=ibeam
    cbf['iexpt']=iexpt

    return cbf

def ionchamber_read_20200722_strain(ionfn):
    """
    This file is to read the ion chamber value during the straining.
    """
    fid=open(ionfn)
    text = fid.read()
    count = 0
    ibeamtmp,iexpttmp=0,0
    lines=text.splitlines()
    lines_len=len(lines)
    dt0=0
    arr=np.zeros((lines_len,3),dtype=np.float)
    for i in range(lines_len):
        Dat,Tim=lines[i].split()[0],lines[i].split()[1]
        dattim=Dat+' '+Tim
        dti=datetime.datetime.strptime(dattim,"%Y/%m/%d %H:%M:%S.%f")
        dti=(dti-datetime.datetime(1970,1,1)).total_seconds()
        #print dti
        if i==0:
            dt0=dti
        dt_elt=int(dti-dt0)
        
        #print dt.timestamp()
        ibeam=float(lines[i].split()[2])
        iexpt=float(lines[i].split()[3])
        #print dt_elt,ibeam,iexpt
        arr[i,0],arr[i,1],arr[i,2]=dt_elt,ibeam,iexpt

    return arr

def ionchamber_read_20200722_static(ionfn):
    """
    This file is used to read the ion chamber value of static samples.
    """
    fid=open(ionfn)
    text = fid.read()
    count = 0 
    ibeamtmp,iexpttmp=0,0
    for line in text.splitlines():
        #split the line
        #convert the 1st ion and 2nd ion
        ibeam=float(line.split()[2])
        iexpt=float(line.split()[3])

        ibeamtmp=ibeamtmp+ibeam
        iexpttmp=iexpttmp+iexpt
        count=count+1

    ibeam=ibeamtmp/count
    iexpt=iexpttmp/count

    return ibeam,iexpt

def scbfread(cbfn,dist=2028.4):
    """
    This file is to read the cbf or tif file collected with the saxs pattern.
    """
    #open the file with fabio
    img=fabio.open(cbfn)
    myd=mydict()
    myd['map']=img.data
    myd['distance']=dist
    myd['boxlen'][0]=0.000172
    myd['boxlen'][1]=0.000172
    myd['wavelength']=0.12398
    myd['title']=cbfn
    myd['beamline']='SSRF 16B'
    myd['filename']=cbfn
    myd['height']=img.dim2
    myd['width']=img.dim1
    myd['center']=[781,804]

    return myd
##############################################################################
#The procedures to evaluate the waxs patterns collected on 2020-07-22 at ssrf.
##############################################################################
def w_series_20200722(wpatt,bg,wmask,ponifn):
    """
    This func is used to evaluate the waxs pattern series.
    This func is normalized with respective the thickness of the sample during
    stretching. Thickness correction is applied here.
    """
    #get the series
    flist=sorted(glob(wpatt+'*'))
    flen=len(flist)
    #run all the files
    for i in range(flen):
        #get the file
        img=wcbfread2dict_20200722(flist[i],ponifn)
        img['map']=img['map']/0.65-bg['map']
        #read the poni
        ai=pyFAI.load(ponifn)
        destw,destaglbin=1000,720
        res2d=ai.integrate2d(img['map'],destw,destaglbin,unit="q_A^-1",\
                mask=wmask['map'],correctSolidAngle=True)
        Intensity2d,twotheta2d,azimchi2d=res2d
        #get the intensity
        img['map']=Intensity2d
        sf_show(img,log=1)
        #get the range of the polar image
        delta_tt=(twotheta2d[-1]-twotheta2d[0])/destw
        rorig=int(twotheta2d[0]/delta_tt)
        rmin,rmax=rorig,rorig+destw
        #azimuthal angle
        #print rmin,rmax
        azim_min=np.radians(azimchi2d[0]+360)
        azim_max=np.radians(azimchi2d[-1]+360)
        #print azim_min,azim_max
        #convert the data in the polar coordinate into cartesian coordinate
        cartarr,pset=convertToCartesianImage(Intensity2d,initialRadius=rmin,\
                finalRadius=rmax,initialAngle=azim_min,finalAngle=azim_max)
        #show the data
        print(cartarr.shape)
        #img['center']=cartarr.shape[0]/2,cartarr[1]
        img['map']=cartarr
        img['center']=[1000,1000]
        sf_show(img,log=1)

##############################################################################
def wcbfread2dict_20200722(fn,ponifn):
    """
    This func is used to read the tif file.
    """
    #read tif
    img=fabio.open(fn)
    #construct the dict
    myd=mydict()
    #include the map
    myd['map']=img.data
    #load the poni
    ai=wponiread_20200722(ponifn)
    #pixel size and 
    myd['distance']=ai.dist
    myd['boxlen'][0]=ai.pixel1
    myd['boxlen'][1]=ai.pixel2
    myd['wavelength']=ai.wavelength
    myd['title']=fn
    myd['beamline']='SSRF 16B'
    myd['filename']=fn
    w,h=img.data.shape[1],img.data.shape[0]
    myd['height']=h
    myd['width']=w

    return myd
###############################################################################
def wponiread_20200722(ponifn):
    """
    Read the poni file calibrated with standard 
    """
    ai=pyFAI.load(ponifn)
    return ai
