#!/usr/bin/env python
#
#*****************************************************************************
#i0horizon()
#i0meridian()
#i0rbf()
#i0Gauss()
#Gaufit1d()
#Gau1d()
#*****************************************************************************
from copy import deepcopy
import numpy as np
import scipy.interpolate as interpolate
import scipy.optimize as optimize
#******************************************************************************
def i0horizon(inset,row=6,order=4,Gau=None,keeporig=True):
    """
    Herein, we extrapolate the saxs data into the beam stop
    But herein, we only build several horizontal lines, after this the
    CDF_2D.i0extrapolate 
    The img should be cut from the image center.
    """
    #first we slice ahorizontal line through the center of image
    #now we assume that the Guenier law is valid,we can use the 4 order
    #of polynomial formula to fit this line
    imgs=deepcopy(inset)
    cen=inset['center'][1]
    half=int(row/2)
    for i in np.arange(cen-half+1,cen+half+1):
        liney=inset['map'][i,:]
        #extrac the x-axis
        linex=np.arange(inset['width'])
        #
        w=np.where(liney > 0.0)
        yfit=liney[w]
        xfit=linex[w]
        if Gau is not None:
           nlen=len(xfit)
           Imean=np.sum(xfit*yfit)/nlen
           sigm=np.sum(yfit*(xfit-Imean)**2)/nlen
           popt,pcov=optimize.curve_fit(Gaufit,xfit,yfit,p0=[1,Imean,sigm])
           ynew=Gaufit(linx,*popt)
        else:
           #4 order polyfit
           z=np.polyfit(xfit,yfit,order)
           p=np.poly1d(z)
           ynew=np.polyval(p,linex)
        if keeporig is True:
           ynew[w]=liney[w]
        imgs['map'][i,:]=ynew
    return imgs
#***************************************************************************
def i0meridian(inset,column=6,order=4,Gau=None,keeporig=True):
    """
    construct data for several columns. like what we did in i0horizon.
    """
    imgs=deepcopy(inset)
    cen=inset['center'][0]
    half=int(column/2)
    for i in np.arange(cen-half+1,cen+half+1):
        linx=np.arange(inset['height'])
        liny=inset['map'][:,i]
        #
        w=np.where(liny>0)
        xfit=linx[w]
        yfit=liny[w]
        if Gau is not None:
           nlen=len(xfit)
           Imean=np.sum(xfit*yfit)/nlen
           sigm=np.sum(yfit*(xfit-Imean)**2)/nlen
           popt,pcov=optimize.curve_fit(Gau1d,xfit,yfit,p0=[1,Imean,sigm])
           ynew=Gau1d(linx,*popt)
        else:
          z=np.polyfit(xfit,yfit,order)
          p=np.poly1d(z)
          ynew=np.polyval(p,linx)
        if keeporig is True:
           ynew[w]=liny[w]
        imgs['map'][:,i]=ynew
        
    return imgs
#***************************************************************************
def i0rbf(inset,func='multiquadric',eps=2,keeporig=True):
    """
    This function extrapolate the beamstop area with rbf.
    func may be:
    'multiquadric': sqrt((r/self.epsilon)**2 + 1)
    'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
    'gaussian': exp(-(r/self.epsilon)**2)
    'linear': r
    'cubic': r**3
    'quintic': r**5
    'thin_plate': r**2 * log(r)
    """
    imgs=deepcopy(inset)
    tmp=inset['map'] > 0.0
    w=np.where(tmp == 1)
    y=w[0]
    x=w[1]
    z=inset['map'][w]
    #now call the fitting functions
    rbfi=interpolate.Rbf(x,y,z,function=func,epsilon=eps)
    #construct the coordinate
    pos=np.where(np.zeros(inset['map'].shape)==0)
    xi=pos[1]
    yi=pos[0]
    zfit=np.reshape(rbfi(xi,yi),inset['map'].shape)
    if keeporig is True:
       zfit[w]=inset['map'][w]
    imgs['map']=zfit
    return imgs
#***************************************************************************
def i0Gauss(inset,keeporig=True):
    """
    Extrapolate the data into the beamstop with Gaussian fitting.
    """
    #duplicate the img
    imgs=deepcopy(inset)
    ppos=np.where(inset['map'] > 0.0)
    xp,yp=ppos[0].ravel(),ppos[1].ravel()
    arrp=inset['map'][ppos].ravel()
    #predefine the parameters
    p=[1.,1.,1,1.,1.]
    #get the fitting function
    arrn=Gaufit2d(imgs,xp,yp,arrp,p)
    imgs['map']=arrn
    if keeporig == True:
       imgs['map'][ppos]=inset['map'][ppos]
    return imgs
#***************************************************************************
def Gaufit2d(img,x,y,arr,p):
    """
    Fit and extrapolate the data into the beamstop region.
    """
    #fit the valid data first
    p_best,C,info,msg,success=optimize.leastsq(resi_Gau2d,p[:],\
                              args=(x,y,arr),full_output=1)
    #get the new index to construct the array
    arr0=np.zeros((img['height'],img['width']),dtype=np.float)
    pos0=np.where(arr0 == 0.0)
    xn,yn=pos0[0].ravel(),pos0[1].ravel()
    #print xn.shape,yn.shape
    arrn=Gau2d(p_best,xn,yn)
    #here we have to reshape the resulted array
    arrn=arrn.reshape(img['height'],img['width'])
    #print(arrn.shape)
    return arrn
#****************************************************************************
def resi_Gau2d(p,x,y,arr):
    """
    2D Gaussian function. Here, x and y should be .ravel() .
    """
    resi=Gau2d(p,x,y)-arr
    return resi
def Gau2d(p,x,y):
    """
    Gaussian 2d function. x and y should be .ravel() .
    """
    G_val=p[0]*np.exp(-1.0*(np.square(x-p[1])/(2.0*p[3]**2))-\
                      1.0*(np.square(y-p[2])/(2.0*p[4]**2)))
    return G_val
#****************************************************************************
def Gaufit1d(I1s,keeporig=True):
    """
    Fit the curve with Gaussian function.
    """
    #get the s and I
    s,I1=I1s[:,0],I1s[:,1]
    s=np.arange(s.shape[0])
    #get the valid data (positive)
    ppos=np.where(I1 > 0.0)
    I1p=I1[ppos]
    sp=s[ppos]
    coeff,varmat=optimize.curve_fit(Gau1d,sp,I1p,method='lm')
    #get the expected data
    ydata=Gau1d(s,coeff[0],coeff[1],coeff[2])
    #use the original data to replace the fitted data
    if keeporig == True:
       ydata[ppos]=I1[ppos]
    I1sn=deepcopy(I1s)
    I1sn[:,1]=ydata
    
    return I1sn

def Gau1d(x,a,x0,sig):
    """
    1D Gaussian function. 
    """
    #
    return a*np.exp(-(x-x0)**2/(2.*sig**2))
