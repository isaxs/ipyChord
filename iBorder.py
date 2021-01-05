#!/usr/bin/env python
#This module extrapolates the saxs data into the apron the pattern.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import scipy.signal as signal
import scipy.interpolate as interpolate
import numpy as np
import math
from sf_show import sf_show
from copy import deepcopy
from i0make import i0rbf
from CDF_2D import killcircleout,radius,flipharmony,cutwin,fillfit,fillspots
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def borderfitpolyN(imgin,npieces=5,ndata=256,linperc=10,sidelen=None,\
                    func='cubic',badband=15,medbox=5):
    """
    This func extrapolates the data into the apron of the saxs pattern.
    It cuts a piece from the saxs pattern and use the rbf to extrapolate, and 
    then cut another one until finishing all the pieces in the apron.
    """
    #generate  a mask of all the valid points
    m=deepcopy(imgin)
    imgout=deepcopy(imgin)
    #img=deepcopy(imgin)
    m['map']=imgin['map'] > 0.0
    radiuu=radius(m)
    radiuucut=radiuu-badband
    print('The radius of the valid area is '+str(radiuu))
    wh=int(radiuu*linperc/100/2)*2
    print('Image shell thickness of image radius: ', linperc,'%')
    print('The side-length of the rect is '+str(wh))
    
    if sidelen is not None:
       wh=sidelen
       print('The actual box length is '+str(wh))
    print('We are dividing one quadrant into '+str(npieces)+' pieces')
    
    #cut a small round pattern from the original pattern
    imgout=killcircleout(imgin,imgin['center'][0],imgin['center'][1],\
                           radiuucut)
    index=0
    print('Wait for a moment, please!')
    for i in range(-10,npieces+10,1):
        #print('We are working on the '+str(i)+'th piece of \
        #                        the second quadrant')
        #get the center of the window what we will cut off
        #radiuu*0.95 get more valid data
        xcen=imgout['center'][0]-radiuucut*math.cos(math.pi/2/npieces*i)
        ycen=imgout['center'][1]-radiuucut*math.sin(math.pi/2/npieces*i)
        #cut off the piece
        winpi=cutwin(imgout,wh,wh,xcen,ycen)
        winpi=fillspots(winpi,medbox)
        #sf_show(winpi,closewin=1,auto=1)
        #fit the data in the winpi
        #pick up some good points, fit these points, extrapolate the blind area
        pos=np.where(winpi['map']>0.)
        #convert the pos to str in order to put xy in one
        str0=pos[0].astype('str')
        str1=pos[1].astype('str')
        if str0.size < ndata:
            print('There are not enough points for selection!')
            raise ValueError('Please enlarge the window to cut more gooddata')
        dum=np.chararray(str0.size,itemsize=7)
        #combine the row and col NO. together
        for i in range(str0.size):
            dum[i]=str0[i]+'_'+str1[i]
        #select some random points
        #print('There are '+str(str0.size)+' valid pixels \
        #                 in the cut-off window')
        posnew=np.random.choice(dum,ndata,replace=False)
        #print('posnew size is '+str(posnew.size))
        #split the posnew xy in order to get the position z-value
        #tmp[i,0] is storing the row no., tmp[i,1] is storing col no.
        tmp=np.zeros((posnew.size,2),dtype=np.uint16)
        #split the positions
        for I in range(posnew.size):
            y,x=posnew[I].split('_')
            tmp[I,1],tmp[I,0]=int(y),int(x)
        #construct the data for rbf
        fdata=np.zeros(ndata,dtype=np.float)
        for k in range(ndata):
            fdata[k]=winpi['map'][tmp[k,1],tmp[k,0]]
        rbfi=interpolate.Rbf(tmp[:,0],tmp[:,1],fdata,function=func,epsilon=10)
        #construct all the coordinates
        fitcoor=np.zeros(winpi['map'].shape)
        npoints=winpi['map'].shape[0]*winpi['map'].shape[1]
        xyfit=np.zeros((npoints,2))
        xyfit[:,1]=np.where(fitcoor == 0)[0]
        xyfit[:,0]=np.where(fitcoor == 0)[1]
        zfit=np.reshape(rbfi(xyfit[:,0],xyfit[:,1]),winpi['map'].shape)
        zfit=zfit.astype(np.float)
        #extract the
        winpif=deepcopy(winpi)
        winpif['map']=zfit
        imgout=fillfit(imgout,winpif)
        
    return flipharmony(imgout)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def borderfitrbfslow(imgin,ndata=256,linperc=10,medibox=3,badband=15):
    """
    Fit a masked dict with a smooth surface. It's assumed that the
    invalid data are in the border of the picture
    Input:
    img, a dict, where invalid data are at zero intensity
    linperc, defines the depth of a shell (in percent of the picture 
    radius), in which the weighting function is unity
    medibox, defines the box length of a median filter which is applied
    to the input data before picking references points thus making the
    reference points more representative.
    Using the CBF function
    In this function, the randoms only cover a corner of the pattern.
    We use the flipharmony to fill another three corners 
    """
    if medibox < 2:
       medibox=2
    if medibox > 20:
       medibox=20
    if linperc < 0:
       linperc=0
    print('Image shell thickness of image radius: ', linperc,'%')
    #print 'boxlen of medin filter: ', medibox
    #generate  a mask of all the valid points
    m=deepcopy(imgin)
    #img=deepcopy(imgin)
    m['map']=imgin['map'] > 0.0
    radiuu=radius(m)
    m=killcircleout(m,m['center'][0],m['center'][1],radiuu-badband)
    imgout=killcircleout(imgin,imgin['center'][0],imgin['center'][1],\
                                                    radiuu-badband)
    #(Double) median filter smoothing of imgin makes reference 
    #pixles to be picked more representative
    #the default size of medfilt2d is (3,3)
    #the kernel_size should be odd
    imgout['map']=signal.medfilt2d(imgout['map'],kernel_size=medibox)
    #Now prevent from picking reference point from edges after
        
    #Pour ndata reference points for the fitting into two feed arrays
    #xydata, fdata
    #The parameter spread controls the spreading of the references
    #points over the region covered with valid data. If spread is small
    #the references points concentrate close to the outer border of the
    #image map
    spread=0.5
    xydata=np.zeros((ndata,2),dtype=np.int)
    fdata=np.zeros((ndata,1),dtype=np.double)
    index=0
    
    #A prerequisite for weighting:
    #before we start, we want to know the average radius of valid data
    #as measured from the center of the picture
    radiu=radius(imgout)
    linperc=float(linperc)
    #restrict input to values that make sense
    if linperc < 1:
       linperc=1
    if linperc > 100:
       linperc=100
    #compute the radius of dampened core
    progress=int( (1.0-linperc/100)*radiu)
    print('image radius = ',radiu,', radius of dampened core = ',progress)
    cx=-0.5+imgout['center'][0]
    cy=-0.5+imgout['center'][1]
    #repeat
    while index < ndata :
          #pick the reference points in polar coordinates (rho,phi)
          #Division by 6 makes, that with spread=1 the center of the
          #image can be a reference point as well (although with low
          #probability). At rdius*0.95 the probability to pick a reference
          #ponit is set at maximum
          #rho is picked from a Gaussian normal probability distribution
          rho=abs(np.random.normal(0,1,1))*radiu*spread+radiu*0.95
          #phi is picked from a uniform pro
          phi=3.1415926*np.random.random_sample()
          x=int(rho*math.cos(phi))#+imgin['center'][0])
          y=int(rho*math.sin(phi))#+imgin['center'][1])
          distcen= int(math.sqrt((x-cx)**2+(y-cy)**2)) > progress
          #print x,y
          if x > 0 and x < imgout['width']/2 and \
                y > 0 and y <imgout['height']/2 and distcen ==1 and \
                m['map'][y,x]==1 :
             #
             xydata[index,0]=x
             #print index
             xydata[index,1]=y
             fdata[index]=imgout['map'][y,x]
             m['map'][y,x]=0
             #print
             #print index,x,y,fdata[index]
             index += 1
    
    #call the smoother
    #print xydata.shape,fdata.shape 
    rbfi=interpolate.Rbf(xydata[:,0],xydata[:,1],fdata,epsilon=2)
    #free memory
    xydata=0
    fdata=0
    
    #generate the grid for the surface in the unusual way of
    #coordinate pairs arranged in a way, that the result of
    #Rbf can easily be reformed to be interrepted as a matrix.
    nfit=imgin['center'][0]*imgin['center'][1]
    xyfit=np.zeros((nfit,2),dtype=np.double)
    
    #the y-values
    tmp=np.zeros((imgin['center'][1],imgin['center'][0]))
    w=np.where(tmp==0)
    xyfit[:,0]=w[1]
    xyfit[:,1]=w[0]
    #get the rbfi
    di=rbfi(xyfit[:,0],xyfit[:,1])
    zfit=np.reshape(di,(imgin['center'][1],imgin['center'][0]))
    #free memory
    xyfit=0
      
    #reduce the single precision
    zfit=zfit.astype(float)
    #print img['map'].shape,type(img['map'][0,0])
    n= (m['map'] == 0.0)
    posi_zfit=zfit > 0
    zfit=zfit*n[0:imgin['center'][1],0:imgin['center'][0]]*posi_zfit
    #generate and fill result
    
    imgout['map'][0:imgin['center'][1],0:imgin['center'][0]]=zfit
    #free memory
    zfit=0
    masktmp=imgout['map'] >= 0.0
    imgout['map']=imgout['map']*masktmp
    imgout=flipharmony(imgout)
    if keeporig is True:
       posipos=np.where(imgin['map']>0.0)
       imgout['map'][posipos]=imgin['map'][posipos]
    return imgout

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def borderfitrbffast(imgin,ndata=256,linperc=10,badband=30):
    """
    This function is used to extrapolate the intensity at the boundary
    into until zero...as flat as possible.
    Use the rbf to fit and extrapolate the intensity.
    In this function, the random only cover a corner of the pattern
    We use flipharmony to fill another three corners.
    Restriction: this function must be run after cenfit and i0extrapolate
    and the pattern height and width has three digits or less than
    three digits.
    When the MEMORY ERROR appears, reduce the ndata
    """
    if linperc < 0:
       linperc=0
    print('Image shell thickness of image radius: ',linperc,' %')
    #generate all the valid points
    m=deepcopy(imgin)
    res=deepcopy(imgin)
    m['map']=imgin['map'] > 0.0
    mmask=deepcopy(imgin)
    mmask['map']=np.ones(imgin['map'].shape)
    radiuu = radius(m)
    #kill the circleout part of m
    m=killcircle(m,m['center'][0],m['center'][1],radiuu-badband)
    #get the working img
    imgkeep=deepcopy(imgin)
    mmask=killcircle(mmask,mmask['center'][0],mmask['center'][1],\
                              radiuu-badband)
    imgkeep=killcircleout(imgkeep,imgkeep['center'][0],\
                        imgkeep['center'][1],radiuu-badband)
    #now pick some points from outskirt of saxs pattern
    #get the upper left corner
    ulmask=np.vsplit(np.hsplit(m['map'],2)[0],2)[0]
    #set the working array to store the coordinate of choosen points
    xydata=np.zeros((ndata,2),dtype=np.int)
    fdata=np.zeros(ndata,dtype=np.float)
    index=0
    if np.sum(ulmask) > ndata:
       #find the good points
       pos=np.where(ulmask)
       #convert the pos to str in order to put xy in one
       str0=pos[0].astype('str')
       str1=pos[1].astype('str')
       dum=np.chararray(str0.size,itemsize=7)
       #combine the row and col NO. together
       for i in range(str0.size):
           dum[i]=str0[i]+'_'+str1[i]
       #select some random points
       posnew=np.random.choice(dum,ndata,replace=False)
       #split the posnew xy in order to get the position z-value
       #tmp[i,0] is storing the row no., tmp[i,1] is storing col no.
       tmp=np.zeros((posnew.size,2),dtype=np.float)
       #split the positions
       for I in range(posnew.size):
           y,x=posnew[I].split('_')
           tmp[I,1],tmp[I,0]=int(y),int(x)
       #put it back to the xyfit
       while index < ndata:
             xydata[index,0]=tmp[index,0]
             xydata[index,1]=tmp[index,1]
             fdata[index]=imgin['map'][tmp[index,1],tmp[index,0]]
             index =index+1
       #now call the fitting functions
       rbfi=interpolate.Rbf(xydata[:,0],xydata[:,1],fdata,epsilon=10)
       #generate the grid for the surface coordinate
       npoints=ulmask.shape[0]*ulmask.shape[1]
       xyfit=np.zeros((npoints,2))
       fit=np.zeros(ulmask.shape,dtype=np.double)
       #put the y into clo 1 and x into col 0
       ##np.where()
       #the returned position array arr[0] row number
       #                            arr[1] column number
       xyfit[:,1]=np.where(fit == 0)[0]
       xyfit[:,0]=np.where(fit == 0)[1]
       #Get the result in a 1D-vector and make it a matrix
       #transpose is no longer necessary
       zfit=np.reshape(rbfi(xyfit[:,0],xyfit[:,1]),ulmask.shape)
       zfit=zfit.astype(np.float)
       #herein, zfit is only the upper left corner
       zfitflipd=np.flipud(zfit)
       pattleft=np.vstack((zfit,zfitflipd))
       pattright=np.fliplr(pattleft)
       patt=np.hstack((pattleft,pattright))
       pattoutskirt=patt*mmask['map']
       res['map']=imgkeep['map']+pattoutskirt
       masktmp=res['map'] >= 0.0
       res['map']=res['map']*masktmp
       res=fillspots(res,3)
       if keeporig is True:
          posipos=np.where(imgin['map']>0.0)
          res['map'][posipos]=imgin['map'][posipos]
       return res
    else:
       print('Please enlarge the apron area')
       print('You can enlarge the badband')
