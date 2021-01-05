#!/usr/bin/env python
#this file is the module of all the relative functions of 
#chord distribution function,including pkl pattern read, write,
#and show when we compute the cdf.
#usage: import CDF_2D 
#its function can be used like CDF_2D.funx()
#the doc of each function is attached in the function
#no mouse operation in this module
#herein, funx() contains:
############################################################################
#pklread()
#pklwrite()
#cutwin()
#cutwinhalfup()
#cutwinhalfdown()
#blow()
#shrink()
#fillfit()
#fibproj()
#ference()
#logscale()
#flatback()
#base512()
#patdist()
#dist()
#borderfit()
#borderzero()
#flipharmony()
#filpudharmony()
#lowpass()
#smooth()
#chord()
#prochord()
#lowpass1d()
#fourier1d()
#widen()
#fourier2d()
#scat2fou()
#inside()
#radius()
#zweight()
#cycexpand()
#i0extrapolate()
#polyfit2d()
#polyval2d()
#blindsize()
#tolog()
#erode()
#dilate()
#opening()
#closing()
#diamatmask()
#antidiamatmask()
#cenfit()
#pilholeboundaries()
#cenfitoff()
#fillspots()
#killcircle()
#killcircleout()
#shiftxy()
#azimrot()
##############################################################
#these package must be loaded
import _pickle as pickle
import scipy.ndimage.filters as filters
import scipy.ndimage.interpolation as interpolation
import scipy.ndimage.morphology as morphology
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import scipy.signal as signal
import scipy.interpolate as interpolate
import zlib
from copy import deepcopy
import itertools
import scipy.ndimage.measurements as measurements
import pyfftw
from sf_show import sf_show
##############################################################
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def pklread(imgname):
    """
    pklread(imgname). This function is used to read the pkl file stored on
    the hard disk.
    imgname: it should be the basestring.
    res: the returned python dictionary including many domains.
    """
    if not isinstance(imgname,basestring):
       raise AssertionError('pklread:imgname is not string')
    if not '.' in imgname:
       imgname=imgname.strip()+'.pkl'
    res=open(imgname,"rb")
    res=zlib.decompress(res.read())
    res=pickle.loads(res)
    return res
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def pklwrite(img,pklname):
    """
    pklwrite(img,pklname). This function used to write the python dictionary
    with many domains on the hard disk. The pkl file is stored in binary 
    format, and it's compressed by the zlib. It uses the HIGHEST_PROROCAL
    The default compression is level 6. 
    """
    #herein,img is a dictionary
    if not isinstance(img,dict):
       raise AssertionError('pklwrite:img is not dict')
    if pklname.split('.')[-1] == 'pkl':
       pklname=pklname
    else:
       pklname=pklname+'.pkl'
    fw=open(pklname,'wb')
    #the default compression level is 6.
    imgcomp=zlib.compress(pickle.dumps(img,pickle.HIGHEST_PROTOCOL))
    fw.write(imgcomp)
    fw.close()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cutwin(img,width=128,height=128,xcen=None,ycen=None):
    """
    Function for cutting a window out of a big image
    return a smaller image
    img   a dictionary with 2d array.
    width width of the output window
    height height of the output window
    xcen central column of the window with respect to the input window
    ycen central row of the window with respect to the input window
         (count from the bottom)
    RESTRICTION:
    width and height must be even interger numbers
    """
    if not isinstance(img,dict):
       raise TypeError('CDF_2D.cutwin:img is not dict')
    if xcen is None:
       xcen=img['center'][0]
    if ycen is None:
       ycen=img['center'][1]
    wh=int(width/2)
    hh=int(height/2)
    if wh < 3 or hh < 3:
       raise ValueError('CDF_2D.cutwin: Desired window is too small')
    width=wh*2
    height=hh*2
    if xcen-wh < 0 or xcen+wh > img['width'] or ycen-hh < 0 or \
                                       ycen+hh > img['height']:
       raise ValueError('CDF_2D.cutwin: Desired window exceeds map of \
                         the input image')
    #make the result img
    win=deepcopy(img)
    #
    #Fill in the window of the input map
    win['map']=img['map'][ycen-hh:ycen+hh,xcen-wh:xcen+wh]
    win['center'][0]=img['center'][0]-xcen+wh
    win['center'][1]=img['center'][1]-ycen+hh
    win['width']=width
    win['height']=height
    
    return win
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cutwinhalfup(img,width=128,height=128,xcen=None,ycen=None):
    """
    Thif function is used to cut the central-up part from a image
    about its center.
    cutwin(width,height,cenx,ceny)
    Herein, the height is the double size of what we get.
    """
    if not isinstance(img,dict):
       raise TypeError('CDF_2D.cutwin:img is not dict')
    if xcen is None:
       xcen=img['center'][0]
    if ycen is None:
       ycen=img['center'][1]
    wh=int(width/2)
    hh=int(height/2)
    if wh < 3 or hh < 3:
       raise ValueError('CDF_2D.cutwin: Desired window is too small')
    width=wh*2
    height=hh*2
    if xcen-wh < 0 or xcen+wh > img['width'] or ycen-hh < 0 or \
                                       ycen+hh > img['height']:
       raise ValueError('CDF_2D.cutwin: Desired window exceeds map of \
                         the input image')
    #make the result img
    win=deepcopy(img)
    #
    #Fill in the window of the input map
    win['map']=img['map'][ycen-hh:ycen,xcen-wh:xcen+wh]
    win['center'][0]=img['center'][0]-xcen+wh
    win['center'][1]=img['center'][1]-ycen+hh
    win['width']=width
    win['height']=height
    
    return win
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cutwinhalfdown(img,width=128,height=128,xcen=None,ycen=None):
    """
    This function is used to cut the halfdown window.
    """
    if not isinstance(img,dict):
       raise TypeError('CDF_2D.cutwin:img is not dict')
    if xcen is None:
       xcen=img['center'][0]
    if ycen is None:
       ycen=img['center'][1]
    wh=int(width/2)
    hh=int(height/2)
    if wh < 3 or hh < 3:
       raise ValueError('CDF_2D.cutwin: Desired window is too small')
    width=wh*2
    height=hh*2
    if xcen-wh < 0 or xcen+wh > img['width'] or ycen-hh < 0 or \
                                       ycen+hh > img['height']:
       raise ValueError('CDF_2D.cutwin: Desired window exceeds map of \
                         the input image')
    #make the result img
    win=deepcopy(img)
    #
    #Fill in the window of the input map
    win['map']=img['map'][ycen:ycen+hh,xcen-wh:xcen+wh]
    win['center'][0]=img['center'][0]-xcen+wh
    win['center'][1]=img['center'][1]-ycen+hh
    win['width']=width
    win['height']=height
    
    return win
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def blow(img,times=2,prefilt=False):
    """
    Enlarge the image, so that width and height doubled
    The pixel size will be reduced to half
    """
    
    res=deepcopy(img)
    res['map']=interpolation.zoom(img['map'],times,order=0,\
                                           prefilter=prefilt)
    res['boxlen'][0]=img['boxlen'][0]/times
    res['boxlen'][1]=img['boxlen'][1]/times
    res['width']=img['width']*times
    res['height']=img['height']*times
    res['center'][0]=res['width']/times
    res['center'][1]=res['height']/times
    return res
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def shrink(img,prefilt=False):
    """
    The image is compressed by a factor of 2 in both x and y direction.
    The pixel size of returned img is 2 times of input img.
    """
    #check the type of img
    if not isinstance(img,dict):
       raise TypeError('CDF_2D.shrink: img is not dict')
    #dupliate the image
    newimg=deepcopy(img)
    #use the zoom function to enlarge the 2d array.
    newimg['map']=interpolation.zoom(newimg['map'],0.5,order=0,\
                                           prefilter=prefilt)
    newimg['boxlen'][0]=img['boxlen'][0]*2
    newimg['boxlen'][1]=img['boxlen'][1]*2
    newimg['center'][0]=img['center'][0]/2
    newimg['center'][1]=img['center'][1]/2
    newimg['width']=img['width']/2
    newimg['height']=img['height']/2

    return newimg
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fillfit(img,extra):
    """
    Fill blind areas of a scattering pattern, which is supposed to
    result from a 2D extrapolation carried out either by 
    borderfit or cenfit
    Input:
    img, a dict, which contains the image with the blind spot.
    extra, a dict, which contains data, which shall be inserted into
    the blind area
    The alignment of both pixelmaps is carried out via the data in 
    img['center'] and extra['center']
    Output:
    imgfill, a dict, which contains extrapolated data, where in the 
    original image there was 'no ntensity'
    If the extrapolated map contains negative data, the imgfill.map
    will be set to zero where this occurs.
    """
    if not isinstance(img,dict):
       raise TypeError('CDF_2D.fillfit: img is not dict')
    if not isinstance(extra,dict):
       raise TypeError('CDF_2D.fillfit: extra is not dict')
    #computer the vector of the movement for extra
    mv=[0,0]
    mv[0]=img['center'][0]-extra['center'][0]
    mv[1]=img['center'][1]-extra['center'][1]
    #test, if the moved map of extra is inside the map of img
    if mv[0] <0 or mv[1] < 0 or \
       extra['width']+mv[0] > img['width'] or \
       extra['height']+mv[1] > img['height']:
       raise ValueError('CDF_2D.fillfit: extra not inside the img')
    #cut the window covered by extra out of img
    ux=mv[0]+extra['width']
    uy=mv[1]+extra['height']
    exwin=img['map'][mv[1]:uy,mv[0]:ux]
    #generate a mask describe the blind area
    m= (exwin == 0)
    
    #exlarge the blind area by some pixels
    #dilation box 3x3 enlarges by 1 pixel, 2x2 by 2 
    dila=np.ones((5,5))
    m=morphology.binary_dilation(m,dila)
    #compose the results in the window
    exwin=exwin*(1-m)+extra['map']*m
    #generate the resulted img
    imgfill=deepcopy(img)
    #paste in the window exwin
    imgfill['map'][mv[1]:uy,mv[0]:ux]=exwin
    
    return imgfill 
      
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fibproj(img):
    """
    Project a scattering pattern with fiber symmetry (which must have its
    center defined) onto the s1-s3-plane. This is good to do before a 
    Fourier transformation in order to guarantee that the resulting pattern
    (like a 2D-correlation function, e.g.) really is a section
    Input:
    img, a scattering pattern with fiber symmetry and its center defined
    in its center
    Outpot:
    A scattering pattern projected according to the intergral covering a
    cylindrical definition body
    "Intergrate_{s12=s1}^{s12=s12max} (I(s12,s3) ds2)"
    which can be rewritten using s2=Sqrt(s12^2 - s1^2 ) and its derivate
    with respect to the s12 yielding:
    primg['map'](s1,s3)=
    $\int_{s1}^{s12max} |frac {s12}{\sqrt{s12^2-s1^2}} img['map'](s12,s3)
    ds12 $
    Intergration is carried out using the tangent rule
    The apron extrapolation can be done before this fiber projection.
    """
    if not isinstance(img,dict):
       raise TypeError('CDF_2D.fibproj: img is not dict')
    if img['width']/2 != img['center'][0] or img['height']/2 !=\
                        img['center'][1]:
       print("CDF_2D.fibproj: img is not symetric or not properly centered")
    #preset result
    primg=deepcopy(img)
    #set minimum to zero
    primg['map']=img['map']-img['map'].min()
    #make a simple data array in double precision
    data=deepcopy(signal.medfilt2d(np.double(primg['map'])))
    #print data.min(),data.max(),data.mean()
    #***begin projection*************
    #We need a matrix describing the s12 values (which is s12)
    
    #scatterers paradigms. The center of the pattern s in the corner
    #shared by the four central pixles
    #Thus the horizontal coordinate gets 0.5 extra
    half=primg['center'][0]
    hor=np.roll(np.arange(primg['width'])+0.5,half) 
    #The left wing of hor is not yet reflecting the distance of the 
    #pixel from the center
    hor[0:half]=primg['width']-hor[0:half]
    hor=np.double(hor)
    #make a unit column vector
    ver=np.ones((primg['height']),dtype=np.double)
    #generate the s12 by matrix product
    ####Attention: the order of data
    s12=ver.reshape(len(ver),1)*hor.reshape(1,len(hor))
    #we have got a same matrix as in the s12=hor#ver
    #and we call the column s12[:,NO.] row s12[NO.,:]
    #variable substituation needs s12 squared
    s12q=np.square(s12)
    s12=np.double(s12)
    s12q=np.double(s12q)
    #we will work on the left side only (s12 is symmetric)
    #number of columns on the left side
    columns=primg['center'][0]
    i=1
    #Initiate constants
    truecen=primg['center'][0]-0.5
    #compute the intergral for every column on the left hand side
    #NO.=column will not be extracted in python
    while i < columns:
        #for the i-th column (coordinate: s1) there are i columns 
        #left of it to the process
        #cut the intensity matrix
        #print 'We are processing column: ',0,':',i-2
        is12=data[:,0:i]
        #print i,is12.mean()
        #compute the variable substituation matrix "ds2"
        delt=s12q[:,0:i] -(i-truecen)**2
        #print i, delt.min()
        delt=np.true_divide(s12[:,0:i],np.sqrt(delt))
        #print delt.max(),delt.min()
        ##becomes s12^2-s1^2
        ##becomes s12/sqrt(s12^2-s1^2)
        #delt goes with 1/sin(\phi) being ablge
        #between s12-direction and s1-axis, and this poses
        #a problem to the numerical integration to be carried out

        #multiply intensities by line element "ds2"
        is12=np.multiply(is12,delt)
        #print is12.max(),is12.min()
        #print is12.shape 
        #Sum is12 over its first dimension (y) into projected column
        #(this is the exact implementation of the tangent rule up to
        #the upper bound corresponding to (i-1-center[0]*boxlen[0])
        #because scatterer's paradigm is that wonderfully convenient).
        #In order to avoid the singularit the intergral range is 1.5
        #pixels shorter than the intended overall length . we will take
        #care for that in the last statement of the loop.
        summe=np.sum(is12,axis=1)
        primg['map'][:,i]=summe+1.5*is12[:,i-1]
        #endfor project over all the columns on the left
        i += 1
    #copy the result from the left to right side
    col=columns
    newR=np.fliplr(primg['map'])
    primg['map'][:,col:2*col]=newR[:,col:2*col]
    #end projection    

    return primg

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def ference(img,maxit=1,power=2,noise=0.1,backf=0.002,outsize=512,\
              prefilt=False):
    """
    Make an 'interference function', which can be converted into
    a chord distribution function by FFT
    Input:
    img, a dict
    maxit, maximal number of iterations (in conjunction with/iterate
    default: 1
    pow, The power of s, the image is initially multiplicated with
    (default: 2) It's normally set to 2. indicating the Laplacian 
    operator in its reciprocal space representation.
    noise, [0..1] limiting spatial frequency for noise filtering 
    expressed in units of the relative Nyquist frequency
    default: 0.1
    backf [0..1] background filter frequency in units of relative signal
    band width (default: 0.0002)
    outsize [a power of 2] determines the size (and thus the accuracy)
    of the result, which will have outsize x outsize pixels.
    (default:512, works with computers which have little RAM)
    apron which has to be extraolated before the ference function.
    output, A 2D interference function
    Restriction:
    It should only be called with scattering patterns which are projected
    on s1-s3-plane, the center of which is filled. The data should cover
    approxiamtely a square region in reciprocal space. An exactly 
    reciprocal space will be cut before starting to work.
    """
    if not isinstance(img,dict):
       raise TypeError('CDF_2D.ference: img is not dict')
    if img['map'][img['center'][1],img['center'][0]] <= 0:
       raise ValueError('central blind spot,\
              plz extrapolate before using CDF_2D.ference')
    #the default parameter
    #maxit=1, don't iterate, backf=0.002,it must be extremely small
    #outsize=512, it's only needed to increase outsize,if the CDF 
    #has far-ranging correlations
    #parameter plausibility checks
    if outsize > 4096:
       outsize=4096
    if outsize < 64:
       outsize=64
    #round to the next power of 2
    outsize=2**int(math.log(outsize)/math.log(2.0)+0.5)
    #multiplication power to 2 -> Lalacian
    if power < 0:
       power=2
    #at least the first step of iteration
    if maxit < 1:
       maxit=1
    print('Image I(s1,s3) will be multiplicated by s^', power)
    print('Results will have', outsize, 'x', outsize, 'pixels.')
    print('Background cut-off (fraction of bandwidth):',backf)
    print('Noise cut-off (fraction of bandwidth): ',noise)
    
    #convert from fraction of bandwidth to absolute spatial frequency
    backf=outsize*backf
    
    #Change the gridding of the image to make the edge length a power
    #of 2 with square pixels 
    ab=base512(img,outsize)
    
    #generate the power of s
    s=interpolation.shift(patdist(outsize),(outsize/2,outsize/2),\
                           mode='reflect',order=0,prefilter=prefilt)
        
    #prepare the powlaw for Laplacian
    powlaw=s**power 
        
    #background now is a 2D map
    #u=flatback(ab)
    #m=ab
    #m['map']= ab['map'] > 0
    #ab['map']=(ab['map']-u)*m['map']
    
    #Laplacian: power law transformation
    ab['map']=ab['map']*powlaw
    
    #prepare for iterative roughness subtraction
    limit=10* max([abs(ab['map'].max()),abs(ab['map'].min())])
       
    #Background subtraction and Hanning window
    ab=flipharmony(ab)
    abf=flipharmony(lowpass(ab,backf,isprepared=1))
    ab['map']=(ab['map']-abf['map'])*np.outer(np.hanning(outsize),\
                                                np.hanning(outsize))
    #ab['map']=np.real(ab['map'])
    if maxit >= 2:
       print('Iterative roughness removal. Limit: ', limit)
       print('**Be careful. Must be well justified!***')
       #iterate to remove roughness
       I=1
       while (abs(np.sum(ab['map'])) > limit) and (I < maxit):
             abf=flipharmony(lowpass(ab,backf,isprepared=1))
             ab['map']=(ab['map']-abf['map'])*np.outer(np.hanning(\
                                         outsize),np.hanning(outsize))
             I=I+1
             print('Repeating subtraction of bg scattering. No. ',I)
    ab['map']=np.real(ab['map'])
    #filter noise from the result
    smoo=smooth(ab,noise)
    smoo['map']=np.real(smoo['map'])
    ab=flipharmony(smoo)
    return ab

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def logscale(img,ulev):
    """
    Does a proper logarithmic scaling of the positive
    values of img obeying the upper level ulev. When
    called with an inverted img, the negative values can
    be scaled in another run with the same ulev and, finally,
    both subimages can be combined to yield a smooth surface
    """
    #mask to show only positive function values
    res=deepcopy(img)
    array=res['map']
    #get the index of number that is less and equal to zero
    ind=np.where(array <= 0.0)
    #all the number less and equal to zero is replaced by trumin
    if len(ind) is 0:
       #assign the maximal value of array to the ind
       array[ind]=array.max()
       #get the trumin of array
       trumin=array.min()
       #assign the minimal value
       arrar[ind]=trumin
    #normalize res to give a nice color scale
    array=array-array.min()
    trumax=array.max()
    array=100*array/ulev
    #compute the logarithm
    array=np.log(array+1)
    #scale the image if needed
    scalfac=np.log(100*trumax/ulev)/np.log(100)
    array=array/np.log(100)*scalfac
    #indicate invalid values
    if len(ind) > 0:
       array[ind]=0.0
    #return the array
    res['map']=array
    return res

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def flatback(img,prefilt=False):
    """
    img, is the square image of a harmonized scattering pattern
    The function returns a 2D background matrix that has to be subtrated
    from the image in order to make the apron zone as flat as possible
    before spatial frequency filtering is assumed to start. It confirms
    to the general assumption that the diffuse background is explained
    in even powers of s. Here the expansion is a0+b2*s_{s12}^2+c2*s_3^2
    """
    limit=0.9
    #make an image containing s values (not normalized)
    base=img['width']/2
    s=interpolation.shift(patdist(2*base),(base,base),\
                        mode='reflect',order=0,prefilter=prefilt)
    #get the number of point which are outside the inscribed circle
    #and contain valid data. They shall be considered for the purpose
    #of background determination.
    plist=np.where(s > base*limit)
    if len(plist[0]) < 100:
       raise ValueError('CDF_2D.flatback: Not enough valid point found\
                         for extrapolation')
    #generate the lists of the x and y values in the point list
    #to be used in the 2D regression process
    xlist=plist[1]-0.5*img['width']+0.5
    ylist=plist[0]-0.5*img['height']+0.5
    #compute the element of the matrix that forms the linear equation
    #system
    #n    sx2    sy2     ...  sv
    #sx2  sx4                 svx2
    #sy2  sx2y2  sy4          svy2

    n=len(plist[0])
    sx2=np.sum(xlist**2)
    sy2=np.sum(ylist**2)
    sx4=np.sum(xlist**4)
    sy4=np.sum(ylist**4)
    sx2sy2=np.sum(xlist**2*ylist**2)
    sv=np.sum(img['map'][plist])
    svx2=np.sum(img['map'][plist]*xlist**2)
    svy2=np.sum(img['map'][plist]*ylist**2)
    #enter elements to 1D array
    homomat=np.array([n,sx2,sy2,sx2,sx4,sx2sy2,sy2,sx2sy2,sy4])
    #giveit 2d look
    homomat=np.reshape(homomat,(3,3))
    #prepare for solving the system using Cramer's rule
    #The determinant of the homogeneous part
    homodet=np.linalg.det(homomat)
    #solve for the constant coefficient (leftmost in matrix)
    mat=homomat.copy()
    mat[0,0]= sv
    mat[1,0]=svx2
    mat[2,0]=svy2
    c0=np.linalg.det(mat)/homodet
    #solve for the  coefficient of the x-direction
    mat=homomat.copy()
    mat[0,1]=sv 
    mat[1,1]=svx2
    mat[2,1]=svy2
    cx=np.linalg.det(mat)/homodet
    #solve the coefficient of the y-direction
    mat=homomat.copy()
    mat[0,2]=sv
    mat[1,2]=svx2
    mat[2,2]=svy2
    cy=np.linalg.det(mat)/homodet
    
    #print c0,cx,cy
    
    #generate x and y
    #width and height of the image and the resulting map
    w=img['width'] 
    x=np.reshape(range(w*w),(w,w))-0.5*w+0.5
    y=np.true_divide(np.reshape(range(w*w),(w,w)),w) - 0.5*w+0.5
    #generate the resulting background map
    res=x**2*cx+y**2*cy+c0
    return res
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def base512(img,outsize=512,cut=None,prefilt=False):
    """
    Prepare for Fourier operations
    INPUT:
    img, a dict
    outsize, a different output edge length may be given by the
    experienced. It defaults to 512.
    cut, Change the procedure from congridding the
    rectangular input matrix on a square to
    cutting the maximum square matrix from the
    supplied image map. See: PROCEDURE!
    IF called with /cut: (former procedure)
    The input image is prepared by CUTTING to the
    maximum square of pixels as they are supplied.
    Then this square is  "congridded" to an 
    outsize  x  outsize  definition range.
    If called without /cut:
    The COMPLETE rectangular matrix is congridded to a
    square matrix of outsize x outsize pixels.
    This changes the size of the base pixel stored in
    prep.boxlen - So from this parameter "square pixels"
    may later be generated for the purpose of user-friendly
    display.
    
    """
    if not isinstance(img,dict):
       raise TypeError('CDF_2D.base512: img is not dict')
    #
    if cut is not None:
       #The old and complicated business of cutting
       exwidth = img['width']*img['boxlen'][0]
       exheight = img['height']*img['boxlen'][1]
       #smallest extension becomes possible range
       if exwidth > exheight :
          inrange=exheight
       else:
          inrange = exwidth
       #cut a quadratic image in external coordinates
       wishwidth = int( inrange/img['boxlen'][0] )
       wishheight = int( inrange/img['boxlen'][1] )
       tile = cutwin( img, wishwidth, wishheight )
    else:
       tile=deepcopy(img)
    #resample to a power of 2 in order to fulfil the
    #requirements of fast Fourier-transform
    #Herein,we use the user-defined function "zoom"
    zoomtimes=float(outsize)/float(img['width'])    
    resamp=interpolation.zoom(tile['map'],zoomtimes,order=0,\
                                   prefilter=prefilt)
    prep=deepcopy(img)
    prep['map']=resamp
    prep['boxlen'][0] = tile['boxlen'][0]*tile['width']/outsize
    prep['boxlen'][1] = tile['boxlen'][0]*tile['height']/outsize
    prep['center'][0]=outsize/2
    prep['center'][1]=outsize/2
    prep['width']=outsize
    prep['height']=outsize
    
    return prep
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def patdist(el):
    """
    INPUT: <el> Edge length of an 2D array
    el must be an even, integer number.
    OUTPUT: A 2D-array similar to the one generated by the
    DIST() function, but there the distance is measured
    "from the lower left CORNER of the edge pixel"
    instead of the distance from its CENTER. The latter
    is the natural definition for scattering applications.
    Using the image processing paradigm on my scattering
    images generates sharp indentations running along the
    axes.
    
    """
    elh = int(el/2)
    elm = elh
    eli = elh*2
    d = np.square(dist(eli)) + 0.5
    #Each element of corr is filled with the sum of its indices.
    corr = np.reshape(range(elh**2),(elh,elh))
    corr = (corr -np.mod(corr,elh))/elh +np.mod(corr,elh)
    corr = np.fix(corr)
    
    ref = d[0:elm,0:elm] + corr
    d[0:elm,0:elm] = ref
    d[0:elm,elh:eli]= np.rot90(ref,k=3)
    d[elh:eli,elh:eli] =np.rot90(ref,k=2)
    d[elh:eli,0:elm]=np.rot90(ref,k=1)
    
    d = np.sqrt(d)
    return d
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def dist(n):
    #generates a square array in which each element equals
    #the Euclidean distance from the nearest -corner.
    dist=np.zeros((n,n),dtype=np.double)
    i=0
    while i < n:
        x=i
        if i >= n/2:
           x=n-i-1
        j=0
        while j < n:
            y=j
            if j >= n/2:
               y=n-j-1
            dist[i,j]=np.sqrt(x**2+y**2)
            j += 1
        i += 1
    return dist

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def borderfit(imgin,ndata=256,linperc=10,medibox=3,badband=15,\
               func='multiquadric',keeporig=True):
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
    rbfi=interpolate.Rbf(xydata[:,0],xydata[:,1],fdata,function=func,epsilon=2)
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
def borderzero(imgin,ndata=256,linperc=10,badband=30,\
                func='multiquadric',keeporig=True):
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
       rbfi=interpolate.Rbf(xydata[:,0],xydata[:,1],fdata,\
                             function=func,epsilon=10)
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
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def harmony(img):
    """
    This function is used to harmonize the saxs pattern in order to
    average the four quadrants.
    """
    pass
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def flipharmony(img):
    """
    This function is used to fill the blind spots or blank bars
    of pilatus detector by symmetric reflection
    Attention: this function can only be used in the positive values as 
    the negative values will be set as zero 
    """
    #create two new image
    harm=deepcopy(img)
    tmp=deepcopy(img)
    #get the mask of image
    mask0=img['map'] !=  0.0
    mask1=np.fliplr(mask0)
    mask2=np.flipud(mask0)
    mask3=np.flipud(np.fliplr(mask0))
    #get the flip array
    arr0=tmp['map'].astype(np.float)
    arr1=np.fliplr(tmp['map'])
    arr2=np.flipud(tmp['map'])
    arr3=np.flipud(np.fliplr(tmp['map']))
    #compute the harmony array
    mask=mask0.astype(int)+mask1.astype(int)+mask2.astype(int)+\
                                              mask3.astype(int)
    #extract the bad points
    mask=mask+(mask==0).astype(int)
    arr=arr0+arr1+arr2+arr3
    ha=arr/mask
    harm['map']=ha
    return harm
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def flipudharmony(img):
    """
    This function is used to flip the matrix upside down to harmonize
    the pattern. Now we need to keep "The upper part is perfect" in
    mind. We flip the upper part down as the button part.
    We must be vary cautious. 
    Attention this function can only be used in the saxs with good upper half.i
    It doesnot average the pixels.
    """
    #create two new images
    harm=deepcopy(img)
    #get the mask of image
    tmp=np.vsplit(img['map'],2)
    #get the upper side
    tmp_up=tmp[0]
    #flip the upper side down
    tmp_down=np.flipud(tmp_up)
    #combine the upper and down side of matrix
    tmp=np.vstack((tmp_up,tmp_down))
    harm['map']=tmp
        
    return harm
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def lowpass(img,freq=30,order=1,isprepared=None,prefilt=False):
    """
    Image smoothing by lowpass filtering in the frequency domain
    herein, Butterworth filter
    Input:
    img, a dict
    freq, the spatial cut-off frequency
    order, word the order of the Butterworth filter
    /isprepared, If set, the programmer has prepared a 2^n by 2^n image
    It will not be resized during the procedure.
    Pro:
    The input image is prepare by cutting to a square maximum square
    area, which is 'congrided' to a 512 by 512  definition range. 
    After that the Butterworth filter is applied.
    """
    if isprepared is not None:
       outsize=img['width']
       filt=deepcopy(img)
    else:
       outsize=512
       filt=base512(img,outsize)
    
    #freq=0.07*outsize
    print('Frequency of CDF_2D.lowpass: ',freq)
    if freq <= 0:
       print('Freq is too low')
    if freq > outsize-1:
       print('Freq is too high')
    if order < 1:
       print('Order too low')
    if order > 31:
       print('Order too high')
    #long integer can only be 2**31
    ordd=2**int(order)
    #create the euclidean distance matrix 
    H=interpolation.shift(dist(outsize),(outsize/2,outsize/2),\
                       mode='reflect',order=0,prefilter=prefilt)
    #performing fft transform
    tmp=np.fft.fft2(filt['map'])
    #print tmp0.shape,type(tmp0)
    #shift the frequency
    #print tmp.shape
    #normalize
    #divide by element number of the array, normalization
    #tmp=tmp/(tmp.shape[0]*tmp.shape[1])
    tmp=np.fft.fftshift(tmp)
    #make the mask
    M=deepcopy(H)
    #mask the valid frequency
    M= H > freq
    #Get the masked H
    H=M*H
    #apply the convolution
    tmp=tmp*(1/(1+(H/freq)**ordd))
    #shift back
    tmp=np.fft.ifftshift(tmp)
    filt['map']=np.fft.ifft2(tmp)
    filt['map']=np.real(filt['map'])
    
    return filt

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def smooth(img,cut=0.1):
    """
    Smooth a scattering pattern by digital filtering
    Input:
    img, the measured 2D scattering pattern 
    cut, the cut-off Nyquist frequency (fraction of the interval in
    which the Fourier-transformed image is defined) Default: 0.1
    Procedure:
    use of DIGITAL_FILTER and low_pass filter to smooth the pattern
    """
    #pat=deepcopy(img)
    cut=abs(cut)
    if cut < 0.0005 :
       cut=0.0005
    if cut > 1.0:
       cut=1.0
    #generate the 1D digital low pass filter (flow=0) with
    #cut-off frequency cut, Gibbs suppressor -50 dB and filter order
    # F0=10, (makes 2*fo+1 coefficients in filter)
    fo=10
    filterr=signal.firwin(fo,cut)
    #make it 2d for the purpose of operating on images
    filtt=filterr.reshape(fo,1)
    filte=filtt * filterr
    #print filte 
    #Cyclic boundary extension of the image in order to push boundary
    #effects out of the relevant image map
    imgi=cycexpand(img,fo,fo)
    #compute the smoothed result (undefined in an apron which is n+1)
    #pixels wide
    imgi['map']=np.real(signal.convolve2d(imgi['map'],\
                              filte,mode='same')/np.sum(filte))
    #cut back to relevant and unaffected region
    pat = cutwin(imgi,img['height'],img['width'])
    
    return pat
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def chord(img,padding=2,fast=None,outputorig=None):
    """
    Image transformation from reciprocal to real space
    gr=chord(gs,padding)
    Input:
    gs, an image of the 2D interference function
    padding, an integer giving the number of zero paddings default:2
    The higher the zero padding, the higher the spatial resolution of
    CDF
    Output: A 2D chord distribution
    Remarks: It is a feature of Fourier transform theory that one does
    not increase the spatial resolution of the CDF by providing highly
    resolved G(s) functions. Instead, higher resolution of G(s) extends
    the range in r-space, in which the CDF can be analyzed. Only in rare
    cases this is really needed.
    If one intends to get higher spatial resolution in the CDF it is only
    required to increase the number of the zero paddings HERE. This is 
    a common request, because the user normally requires a fine gridding
    of the CDF, because the gridding defines, how precise structural 
    parameters can be determined.
    """
    outsize=img['width']
    if padding <= 0:
       padding=2
    gf=deepcopy(img)
    #the frequency of the following repeated statement ('zero padding')
    #controls the spatial resolution. It costs memory and time. It is
    #a common practice to use it two to four times.
    for I in range(padding):
          gf=widen(gf,100,100)
    #sf_show(gf)
    gf=fourier2d(gf,noscale=1,noparadigm=1,fast=fast)
    #cut back the image to a reasonable size
    if padding > 2:
       outsize=(padding-1)*outsize
    if outputorig is not None:
       gf['map']=-1.0*gf['map']
       return gf
    else:
       gf=cutwin(gf,outsize,outsize)
       #print gf
       #the chord distribution is the negative Fourier of the interference
       #function.
       gf['map']=-1.0*gf['map']
       return gf
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def prochord(img,zerpad=3,backf=0.3,dphi=1,short=None,prefilt=False):
    """
    Function for use with python. It starts from a scattering
    pattern, which is projected on to the fibre plane (s1,s3)
    and performs projection on to an inclined line, subtracts
    background, does 1D-Fourier-transform, stores the result in
    an (r,phi)-matrix and remaps to cartesian coordinates
    for to present the result.
    gr = prochord(pro, zerpad, backf, delphi,short=1)
    pro   An image of the projected intensity which should
          have been produced from fibproj and run through
          base512
    zerpad [0..6] The number of interval doublings by zero padding
           in the input of the 1D Fourier transform. Default: 3
    backf  (0.3) The frequency of the background low pass filter
    delphi angular increment in the polar chart (controls accuracy with
           patterns having a lot of peaks and the computing time)
           default: 1
    short  If this keyword is set, the r-range will NOT be SQRT(2)-times
           longer than half of the original image edge length.The implicit
           prolongation provides the possibility to interpolate a full
           square of image data without loss of information.
    OUTPUT:
          A 2D chord distribution in polar coordinates
    """
    #planar polar coordinate system
    outsize = img['width'] # Width of the input image
    rpoints = int(outsize/2) # Number of points in radial direction
    ppoints = int(90/dphi)+1   # Number of points in phi direction
    dr      = 1.0/(img['boxlen'][0]*rpoints*2**(zerpad+1))
    
    radback = (backf >= 0)
    #run length of the radial ray in the polar diagram
    rrun = int(rpoints*1.4142136 + 0.5)
    if short is not None:
       rrun = rpoints
    #Pol is made to hold the g(r,phi) function
    pol = np.zeros((ppoints,rrun),dtype=np.float)
    #HannWin is the Hann window operator
    hannwin = np.hanning(outsize)
    #s is the "s-value" of the intensity
    s = 0.5 + range(rpoints)
    ss = s**2
    #loop over all inclinations and fill Pol
    for iphi in range(ppoints):
        #tilted image
        #rotate about centre of matrix
        tim=interpolate.rotate(img['map'],iphi*dphi,\
                            reshape=False,prefilter=prefilt)
        #project the tilted data on to the horizontal direction
        curve = np.sum( tim, axis=0 )
        #slight disharmony must be removed. Harmon is only 
        #the right side of curve.
        harmon = ( np.fliplr(curve[0:rpoints]) +\
                                    curve[rpoints:outsize] )/2.0
        #now we have the projection, which must be multiplied by s^2
        #and low pass filtered before fourier transformation
        #multiply by s^2
        harmon = harmon * ss
        if radback :
           harmon =(harmon-lowpass1(harmon,outsize,backf,1))*hannwin
        #Accumulate *negative* Fourier transform in polar coordinates
        re = -1.0*fourier1d( harmon, rpoints, zerpad )
        pol[iphi,:] = re[0:rrun]
        
    #dr is as well the increment in the anticipated cartesian matrix
    res=deepcopy(img)
    res['map'] = pol
    res['boxlen'] = [dr,dphi]
    res['center'] = [0,0]
    
    return res

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fourier1d(right, lenn, mul):
    """
    FFT of the 1d curve
    """
    #prepare input array. One doubling is simply for to mirror "RIGHT"
    ilen = lenn * 2**(mul+1)
    midl = int( ilen/2 )
    inarr = np.zeros(ilen,dtype=np.float)
    right = inarr[midl:midl+lenn]
    #reverse of "right" is left side
    inarr[midl-lenn:midl] = np.fliplr(right)
    #Shift origin to the corners (programmed this way for
    #clarity instead of combining with the two preceeding statements
    inarr = np.roll( inarr, midl )
    #drop the imaginary part
    res = np.real( np.fft.fft( inarr ) )
    #normalization in order to gain invertibility
    res = res/math.sqrt(1.0*ilen)
    #Extract the full result vector now directly
    res = res[0:ilen/2]
    return res
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def lowpass1d(curve, freq=30, order=1):
    """
    Low pass of the 1d curve.
    """
    order = 2**int(order)
    lenn=len(curve)
    h = np.arange(lenn/2)
    r = np.zeros( lenn, dtype=np.int)
    r[0:lenn/2] = h[::-1]
    r[lenn/2:lenn] = h
    #low pass
    res=np.fft.ifft(np.fft.fft(curve)*(1/(1 + (r/freq)**order)))
    res=np.real(res)
    return res
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def widen(img,xperc=50,yperc=50,absolute=None):
    """
    Widen a dict in one (or two) directions by defining a larger
    picture map and placing the image into its center
    wid=widen(img,xperc,yperc)
    wid=widen(img,newwidth,newheight,'absolute')
    img, a dict
    xperc, a positive percentage for widening the map
    yperc, a positive percentage for rising the map
    Restriction:
    xperc and yperc must be positive and less than 300
    """
    wide=deepcopy(img)
    
    if absolute is not None:
       if xperc < img['width']:
          raise ValueError('New width too small')
       if yperc < img['height']:
          raise ValueError('New height too small')
       nhw=int(xperc/2)
       nhh=int(yperc/2)
       #print 'Widening to absolute map size'
    else:
       if xperc > 300:
          xperc=300
       if yperc > 300:
          yperc=300
       if xperc < 0:
          xperc=0
       if yperc < 0:
          yperc=0
       #Compute new half width and new half height
       nhw=int(0.5*img['width']*(1.0+xperc/100.0))
       nhh=int(0.5*img['height']*(1.0+yperc/100.0))
    
    width=nhw*2
    height=nhh*2
    #print width,height
    ohw=int(img['width']/2)
    ohh=int(img['height']/2)
    oldcenx=img['center'][0]
    oldceny=img['center'][1] 
    
    wide['map']=np.zeros((height,width),dtype=np.double)
    wide['map'][nhh-ohh:nhh+ohh,nhw-ohw:nhw+ohw]=img['map'][0:2*ohh,\
                                                             0:2*ohw]
    wide['center'][0]=oldcenx+nhw-ohw
    wide['center'][1]=oldceny+nhh-ohh
    wide['width']=width
    wide['height']=height
    #print wide['center'],img['center']
    return wide

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fourier2d(img,noscale=None,noparadigm=None,fast=None):
    """
    This function is used to do the 2D Fourier transform.
    """
    filt=deepcopy(img)
    if noscale is not None:
       filt=deepcopy(img)
    else:
       filt=base512(img,512)
    #**FFT works with its center in the center of the pixel (0,0)
    if noparadigm is None:
       filt=scat2fou(filt)
    #shift the array
    filt['map']=np.fft.fftshift(filt['map'])
    if fast is None:
       #FFT transform
       filt['map']=np.fft.fft2(filt['map'])
    else:
       print('***We are using the pyfftw to do the FFT')
       pyfftw.interfaces.cache.enable()
       pyfftw.interfaces.cache.set_keepalive_time(5)
       filt['map']=pyfftw.n_byte_align(filt['map'],16)
       filt['map']=pyfftw.interfaces.numpy_fft.fft2(filt['map'],\
            overwrite_input=True,threads=4)
       #threads is got from ncores
       #import multiprocessing
       #ncores=multiprocessing.cpu_count()
       #print type(filt['map']),filt['map'].shape
    #Normalization in order to gain some invertibility
    filt['map']=filt['map'] *1.0/(filt['width']*filt['height'])
    #shift back
    filt['map']=np.fft.ifftshift(filt['map'])
    filt['map']=np.real(filt['map'])
    #realign the pixel center to scattering notion
    if noparadigm is None:
       filt=scat2fou(filt,inverse=1)
    #bookkeeping: pixelwidth in output according to the symmetric
    #definition of the Fourier kernel: 2*PI*x*s
    #This relation has additionally been checked by transforming
    #an isotropic scattering pattern with a sharp Debye ring
    filt['boxlen'][0]=1/(filt['width']*filt['boxlen'][0])
    filt['boxlen'][1]=1/(filt['height']*filt['boxlen'][1])
    return filt
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def scat2fou(img,inverse=None,prefilt=False):
    """
    It transforms from the scattering paradigm of gridding to the 
    fast Fourier transformation paradigm of gridding
    Input:
    gs, normally an image of 2D interference function
    Output:
    gsf, an image, in which the gridding is with respect to the center
    of the pixels with the cetral pixel ("zero") doubled in order to 
    meet the specifications of the fast fourier transformation
    """
    res=deepcopy(img)
    cx=img['width']/2
    cy=img['height']/2
    tx=img['width']
    ty=img['height']
    
    if inverse is not None:
       #compress from the double zeros to single zero gridding
       helpp=res['map']
       res['map'][:,cx-1:tx-1]=helpp[:,cx:tx]
       helpp=res['map']
       res['map'][cy-1:ty-1,:]=helpp[cy:ty,:]
       #double the gridding
       helpp=interpolation.zoom(res['map'],2,order=0,prefilter=prefilt)
       #shift back  a pixel
       helpp=interpolation.shift(helpp,(2,2),mode='reflect',\
                                   prefilter=prefilt,order=0)
       #rebin to original size
       zoomtimes=float(helpp.shape[0])/float(res['height'])
       res['map']=interpolation.zoom(helpp,zoomtimes,order=0,\
                                           prefilter=prefilt)
       #heal the rebin disharmony
       res=flipharmony(res)
    else:
       #double the gridding
       helpp=interpolation.zoom(img['map'],2,order=0,prefilter=prefilt)
       #shift half a pixel
       helpp=interpolation.shift(helpp,(-1,-1),mode='reflect',order=0,\
                                         prefilter=prefilt)
       #again shrink the map to half its size by averaging
       helpp=interpolation.zoom(helpp,0.5,order=0,prefilter=prefilt)
       #now we have an image ,in which the data are with respect to
       #the center of the pixel and the zero pixel is at
       #helpp(cy-1,cx-1). The highest row and column are filled with
       #scratch. DFT asks to have the zero rows and columns to be 
       #present in double. Moreover, rebining does not preserve the
       #harmony of the image map. So we will restore this 
       res=flipharmony(res)
    
    return res
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def inside(x,y,mask):
    """
    Inside could have been programmed as a local function of borderfit
    if th programming language would be able to handle such constructs.
    It checks, if the coordinate pair x,y is within the valid area of a 
    mask, which is a dict.
    """
    insi = (x >= 0) and (y >= 0) and (x < mask['width']) and \
              (y < mask['height'])
    #a valid pixel
    if insi:
       inside= mask['map'][y,x]
    return inside

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
def radius(img):
    """
    radius could have been programmed.
    radius determines an average radius of the valid data
    measured from the img['center']
    """
    #harvest the indices of bad pixels, i.e. where we have no data
    bad=np.where(img['map'] == 0)
    #determine, how many bad pixels we have in img
    many=len(bad[0])
    #compute the distances from the center for all bad points
    baddist=np.zeros(many,dtype=np.double)
    cx=img['center'][0]-0.5
    cy=img['center'][1]-0.5
    baddist[:]=np.fix(np.sqrt((bad[1][:]-cx)**2+(bad[0][:]-cy)**2))
    #make a histogram of bad point distances from the img['center']
    histo=np.histogram(baddist,bins=range(many))
    #let us determine an average distance of the edge from the center of
    #of the image (threshold 1/2), named radiu
    m=int(histo[0].max()/2)
    radiu=0
    while histo[0][radiu] < m :
          radiu +=1
    
    return radiu
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def zweight(img,x,y,corerad):
    """
    Performs no weighting, if the distance of (x,y) from (cx,cy) is larger
    than corerad. But inside the core, i.e. if the distance dist is smaller
    than corerad, the z value is dampened by the sigmodal damping function
    sin**2(dist)
    """
    cx=img['center'][0]-0.5
    cy=img['center'][1]-0.5
    dist=int(math.sqrt((x-cx)**2+(y-cy)**2))
    if dist > corerad:
       f=1.0
    else:
       f=math.sin(dist*math.pi/(corerad*2.0))**2
    zw=f*img['map'][x,y]
    return zw

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cycexpand(img,framew,frameh):
    """
    Expands an image by a frame and fills the frame obeying cyclic
    boundary conditions, such an image can be filtered (median,boxcar,
    convolution) without boundary effects inside the old region of
    definition, provided the requested frame is wide and high enough
    Input:
    img,  a dict, the image which shall be expanded. The image should
    not have a blind apron region. If this should be the case, the
    apron should be filled by borderfit and fillfit
    framew, width of the frame strip (in pixel) which shalle be
    generated and filled on each side of the original image
    frameh, height of the frame strip (in pixels) which shall be
    genrated and filled above and below the original image
    Output:
    The extended image obeying cyclic boundary conditions
    The image should have no blind spots
    """
    if framew < 1:
       framew=1
    if framew > img['width']:
       framew = img['width']
    if frameh < 1:
       frameh =1
    if frameh > img['height']:
       frameh =img['height']
    res=widen(img,img['width']+2*framew,img['height']+2*frameh,'absolute')
    #convient shortcuts
    resw1=res['width']
    resh1=res['height']
    imgw1=img['width']
    imgh1=img['height']
    reswf=res['width']-framew
    reshf=res['height']-frameh
    imgwf=img['width']-framew
    imghf=img['height']-frameh
    #fill blank frame by patching with submatrices of img obeying
    #cyclic boundary conditions
    #
    #right upper "square"
    res['map'][reshf:resh1,reswf:resw1]=img['map'][0:frameh,0:framew]
    #left upper square
    res['map'][reshf:resh1,0:framew]=img['map'][0:frameh,imgwf:imgw1]
    #left lower suqare
    res['map'][0:frameh,0:framew]=img['map'][imghf:imgh1,imgwf:imgw1]
    #right lower square
    res['map'][0:frameh,reswf:resw1]=img['map'][imghf:imgh1,0:framew]
    #right strip
    res['map'][frameh:imgh1+frameh,reswf:resw1]=img['map'][:,0:framew]
    #top strip
    res['map'][reshf:resh1,framew:imgw1+framew]=img['map'][0:frameh,:]
    #left strip
    res['map'][frameh:imgh1+frameh,0:framew]=img['map'][:,imgwf:imgw1]
    #bottom strip
    res['map'][0:frameh,framew:imgw1+framew]=img['map'][imghf:imgh1,:]

    return res
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def i0extrapolate(img,order=4,keeporig=True):
    """
    This function is used to extrapolate the 2D data into the beam
    stop. It uses the polynomial of 4 order to fit the apron outside
    the beamstop. Then it use the same formula to extrapolate the data
    into the blind area.
    In principal, the medianfilter2d will induce the smaller value
    of the circle of the beamstop hole,so it's better not to use the
    medianfilter2d before the extrapolation.
    """
    inset=deepcopy(img)
    inset['map']=inset['map'] > 0.0
    w=np.where(inset['map'] ==1)
    #w[0] is the row and w[1] is the col
    y=w[0]
    x=w[1]
    z=img['map'][w]
    
    #fit the 2d array
    m=polyfit2d(x,y,z,order=order)
    
    #Evaluate it on the same grid
    tmp=np.zeros((img['width'],img['height']),dtype=np.int)
    w0=np.where(tmp==0)
    xx=w0[1]
    yy=w0[0]
    zz=polyval2d(xx,yy,m)
    #
    inset['map']=np.reshape(zz,(img['width'],\
                         img['height'])).astype(np.float)
    if keeporig is True:
       masktmp=inset['map'] > 0.0
       inset['map']=inset['map'] * masktmp
       inset['map'][w]=img['map'][w]
    else:
       masktmp=inset['map'] > 0.0
       inset['map'] = inset['map'] * masktmp
    return inset
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def polyfit2d(x,y,z,order=3):
    """
    Fit the 2d surface with polynomial.
    Example:
    i,j=2,2
    We can export the formula like this,
    a+ay+ay^2+ax+axy+axy^2+ax^2+ax^2y+ax^2y^2.
    """
    ncols=(order+1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        #print 'i is ',i,' j is ',j
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m
def polyval2d(x, y, m):
    """
    Determine the value of fit 2d polynomial.
    """
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        #print 'i is ',i,' j is ',j
        z += a * x**i * y**j
    return z
   
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def blindsize(img):
    """
    determine the size of the blind area in the center of a scattering
    pattern
    Input:
    img , a dict which contains blind central blind spot
    bl a dict which contains the border coordinates of the smallest
    rectangle which encloses 'all' blind spots ot the img
    """
    mask=img['map'] <= 1.0
    if np.sum(mask) < 4:
       raise ValueError('Blind area too small---nothing done')
    #Determine the boundary indecies for the map thta includes
    #all the valid pixels--Data in map are stored row after row
    blindspots=np.where(mask == 1)
    blindxmin=blindspots[0].min()
    blindxmax=blindspots[0].max()
    blindymin=blindspots[1].min()
    blindymax=blindspots[1].max()
    
    bl=deepcopy(img)
    bl['width']=blindxmax-blindxmin+1
    bl['height']=blindymax-blindymin+1
    bl['center'][0]=int((blindxmin+blindxmax)/2)+1
    bl['center'][1]=int((blindymin+blindymax)/2)+1
    
    return bl

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def tolog(img,stretch=None):
    """
    scales an image to logarithmic "intensity"
    Works with images that have zero or
    negative values as well.
    stretches both positive and negative
    sub-image to full scale. This is useful,
    if the image has little correlation but
    the correlation shall be demonstrated.
    The image is split in positive and negative values
    that are processed separately and put together. The
    normalization is critical for a good visual effect.
    I have chosen two decades.
    """
    res=deepcopy(img)
    positive = np.where(img['map'] >= 0)
    negative = np.where(img['map'] < 0)
    pm = img['map'].max()
    nm = -1*img['map'].min()
    if pm== nm:
       return res
    
    if stretch is None:
       if pm < nm :
          pm=nm
       nm=pm
    upimg =logscale( img, pm )
    imgi = deepcopy(img)
    imgi['map'] = -1*imgi['map']
    dnimg =logscale( imgi, nm )
    dnimg['map'] = -1*dnimg['map']
    # because upimg is 0 where dnimg is not and vice versa:
    imgi['map'] = pm*(upimg['map'] + dnimg['map'])
    
    return imgi

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++      
def erode(img,erosize=3):
    """
    Erodes a noisy mask. A mask is a picture, which 
    only contains values of 0 (data not valid) and 1 (data valid)
    This is just the "erosion" part of sharpen. It is useful,
    if one intends to remove the edge of a "GoodData/BadData-mask",
    in order to mark the edge as bad. May be called repeatedly.
    Mask processing. Normally called after a sequence like
    INPUT:
    mask should be an dict, where the map contains only
    "zeros" or "ones". The filling of the primary beam area will
    only work, if the primary beam stop is orientated horizontally
    if erosize is not given, the size of the minimum surviving
    island is a 3 by 3 matrix. erosize is the edgelength of the
    erosion matrix
    The main function of the procedure is the erosion of areas
    in the picture, which contain less than four "true" values.
    In a second scan of the pic.map a primary beam region is
    searched and filled with "true" values.
    """
    eros=np.ones((erosize,erosize),dtype=np.int8)
    res=deepcopy(img)
    res['map']=morphology.binary_erosion(img['map'],eros)
    return res

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def dilate(img,erosize=3):
    """
    dilates a noisy mask. A mask is a picture, which
    only contains values of 0 (data not valid) and 1 (data valid)
    
    """
    eros=np.ones((erosize,erosize),dtype=np.int8)
    #dilate the mask
    res=deepcopy(img)
    res['map']=morphology.binary_dilation(img['map'],eros)
    return res

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def opening(img,erosize=3):
    """
    implements the morphological opening operator
    which removes 'little black spots' from a mask
    ('Closing' would remove 'small black spots').
    
    """
    res=deepcopy(img)
    tmp=erode(img,erosize)
    res=dilate(tmp,erosize)
    return res
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def closing(img,erosize=3):
    """
    implements the morphological closing operator,
    which removes 'little black spots' from a mask
    ('Ouverture' would remove 'little white spots'.
    It would be programmed by firstly calling the
     eroder and secondly the dilator
    """
    res=deepcopy(img)
    tmp=dilate(img,erosize)
    res=erode(tmp,erosize)
    return res
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def diamatmask(img,size=3):
    """
    This function is used to mask the diagonal streak.
    Its mask matrix is diagonal matrix
    """
    res=deepcopy(img)
    arr=np.ones(size,dtype=np.float)
    mat=np.diag(arr)
    res['map']=morphology.binary_erosion(img['map'],structure=mat)
    return res
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def antidiamatmask(img,size=3):
    """
    This function is used to mask the diagonal streak.
    The mask matrix is anti-diagonal matrix.
    """
    res=deepcopy(img)
    arr=np.ones(size,dtype=np.float)
    mat=np.fliplr(np.diag(arr))
    res['map']=morphology.binary_erosion(img['map'],structure=mat)
    return res
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cenfit(imgin,ndata=256):
    """
    Fit a masked dict with a smooth surface. It is assumed that
    the invalid data are in the center of the picture and that 
    those data are surrounded by valid data
    The radial basis function is used to extrapolate it into the 
    beam stop.
    Now in this function, we have assumed that the width and the
    height only has 3 digits or less.
    If you supply a 4 digits, we need to modify the function.
    """
    #generate a mask of all the valid points
    m=deepcopy(imgin)
    m['map']=imgin['map'] > 0.0
    #Median filter of the input data in order to pick representative
    #points
    imgin['map']=signal.medfilt2d(signal.medfilt2d(imgin['map'],\
                               kernel_size=3),kernel_size=3)
    m=erode(m,3)
    
    #Pour ndata reference points for the fitting into two
    #feed array xydata, fdata. 
    #The parameters spread controls the spreading of the 
    #reference points over the region covered with valid 
    #data. If spread is small, the reference points concentrate
    #close to the central border of the 'blind spot' caused by the
    #primary beam stop
    #the default number of selecting points is 256.
    #ndata=256
    xydata=np.zeros((ndata,2),dtype=np.int)
    fdata=np.zeros(ndata,dtype=np.double)
    index=0
    #now we use the np.random.choice to get points in a fast way
    if np.sum(m['map']) > ndata:
       #find the good points
       pos=np.where(m['map'])
       #convert the pos to str in order to put xy in one
       str0=pos[0].astype('str')
       str1=pos[1].astype('str')
       dum=np.chararray(str0.size,itemsize=7)
       for i in range(str0.size):
           dum[i]=str0[i]+'_'+str1[i]
       #select some random points
       posnew=np.random.choice(dum,ndata,replace=False)
       #split the posnew xy in order to get the position z-value
       #tmp[i,0] is storing the row no., tmp[i,1] is storing col no.
       tmp=np.zeros((posnew.size,2),dtype=np.float)
       for I in range(posnew.size):
           y,x=posnew[I].split('_')
           tmp[I,1],tmp[I,0]=int(y),int(x)
       #put the x-value in column 0
       #put the y-value in column 1
       while index < ndata:
             xydata[index,0]=tmp[index,0]
             xydata[index,1]=tmp[index,1]
             fdata[index]=imgin['map'][tmp[index,1],tmp[index,0]]
             index =index+1
       #free memory
       m=0
       #call the smoother 
       rbfi=interpolate.Rbf(xydata[:,0],xydata[:,1],fdata,epsilon=2)
       xydata=0
       fdata=0
       #generate the grid for the surface
       npoints=imgin['height']*imgin['width']
       xyfit=np.zeros((npoints,2))
       #generate the matrix that has the same shape as imgin
       fit=np.zeros((imgin['height'],imgin['width']),dtype=np.double)
       #put the y into clo 1 and x into col 0
       #np.where()
       #the returned position array arr[0] row number
       #                            arr[1] column number
       xyfit[:,1]=np.where(fit == 0)[0]
       xyfit[:,0]=np.where(fit == 0)[1]
       #Get the result in a 1D-vector and make it a matrix
       #transpose is no longer necessary
       zfit=np.reshape(rbfi(xyfit[:,0],xyfit[:,1]),\
                             (imgin['height'],imgin['width']))
       xyfit=0
       fitsurf=deepcopy(imgin)
       fitsurf['map']=zfit.astype(np.float)
       zfit=0
       return fitsurf
    else:
       print('Please draw a larger area')
       print('There are not enough valid points for extrapolation')
       
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fillcen(img,extra):
    """
    Function fills the central blind spot of a scattering pattern
    with data from another image, which is supposed to be the result
    of a 2D extrapolation carried out by cenfit
    """
    mv[0]=img['center']-extra['center'][0]
    mv[1]=img['center']-extra['center'][1]
    #Test 
    itfitsnot= mv[0] < 0 or mv[1] < 0 or \
               extra['width']+mv[0] > img['width'] or\
               extra['height']+mv[1] > img['height']
    if itfitsnot:
       raise ValueError('Extra us not inside the img')
    #cut the window covered by extra out of img
    ux=mv[0]+extra['width']-1
    uy=mv[1]+extra['height']-1
    exwin=img['map'][mv[1]:uy,mv[0]:ux]
    #Here generate a mask describing the blind area
    m=exwin == 0.0
    #compose the result im the window
    exwin=exwin+extra['map']*m
    #generate the resulting structure
    imgfill=deepcopy(img)
    #and pass exwin in the window exwin
    imfill['map'][mv[1]:uy,mv[0]:ux]=exwin
    
    return imgfill

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def pilholeboundaries(img):
    """
    Determine the cheese-hole of harmonized pilatus saxs pattern. 
    """
    #copy in order not to touch the input image
    res=deepcopy(img)
    #mask good/bad pixel of scattering pattern 
    m=deepcopy(img)
    m['map']=m['map'] > 0
    #try to fill inlets, as well average radius of the circle
    #in which the valid pixels reside
    r=int(math.sqrt(np.sum(m['map'])/math.pi))
    #artificially make every pixel valid that is outside of
    #this circle in order to close the important inlets
    mt=deepcopy(m)
    mt['map']=1
    mt=killcircle(mt,mt['center'][0],mt['center'][1],r)
    mq=deepcopy(m)
    mq['map']=m['map']*mt['map']
    #label the holes
    holelabel,holenums=measurements.label(mq['map'])
    print('There are ',holenums,'holes in the pattern')
    #find the holes in a labeled array
    holes=measurements.find_objects(holelabel)
    #limit the hole size
    #small_holes=[hole for hole in holes if 20< mq['map'][hole].size<2000]
    for hole in holes:
        a,b,c,d=hole[0].start,hole[0].stop,hole[1].start,hole[1].stop
        #use the radial basis function to fill the holes
        #with valid intensity
        print('We are processing hole ',a,':',b,',',c,':',d)
        holewindow=res['map'][a:b,c:d]
        actualwindow=res['map'][a:b,c:d]
        actualwindow=actualwindow*holewindow
        w0=np.where(actualwindow > 0)
        wval=actualwindow[w0]
        irbf=interpolate.Rbf(w0[1],w0[0],wval,epsilon=2)
        w1=np.where(np.zeros((b-a,d-c)) == 0)
        zval=irbf(w1[1],w1[0])
        zval=np.reshape(zval,(b-a,d-c))
        res['map'][a:b,c:d]=zval
        hole=[]
    return res
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def oddd(i):
    """
    Make the i to odd.
    """
    res=int(i/2)*2 != i
    return res
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cenfitoff(imgin,ndata=None):
    """
    Fit a masked dict with a smooth surface. It is assumed that the
    invalid data are in the center of the picture and those data are
    surrounded by valid data
    """
    #generate a mask of all the valid points
    m=deepcopy(imgin)
    m['map']=imgin['map'] > 0.0
    alldata=np.sum(m['map'])
        
    #median filter of the input data in order to pick 
    #representative points
    img=deepcopy(imgin)
    img['map']=np.medfilt2(np.medfilt2(imgin['map'],\
                              kernel_size=3),kernel_size=3)
    m=erode(m,3)
    #Pour ndata reference poinmts for the fitting into two
    #feed arrays xydata, fdata
    #The parameter spread controls the spreading of the
    #reference points over the region covered with valid
    #data. If spread is small, the reference points
    #concentrate close to the central border of the 
    #blind spot caused by the primary beam stop
    spread=0.33
    if ndata is None:
       ndata=256
       if ndata > alldata/2 :
          ndata=int(alldata/2)
        
          xydata=np.zeros((ndata,2),dtype=np.int8)
          fdata=np.zeros(ndata,dtype=np.double)
          index = -1
          breakdown=0
          while index < ndata-1 :
                x=int(np.random.normal(0,1,1)*6*\
                                     img['width']*spread)
                if x > 0 and x < img['width']:
                   y=int(np.random.normal(0,1,1)*6\
                                     *img['height']*spread)
                   if y > 0 and y < img['height']:
                      #now the random position is inside the map
                      #is it on a valid pixel as well
                      if m['map'][y,x]:
                         index += 1
                         xydata[index,0]=x
                         xydata[index,1]=y
                         fdata[index]=img['map'][y,x]
                         m['map'][y,x]=0

    m=0
    rbfi=interpolate.Rbf(xydata[:,0],xydata[:,1],ndata,epsilon=2)
    xydata=0
    fdata=0
    #generate the grid 
    nfit=img['width']*img['height']
    xyfit=np.zeros((nfit,2),dtype=np.double)
    
    xyfit[:,1]=np.where(xyfit == 0)[1]
    xyfit[:,0]=np.where(xyfit == 0)[0]
    zfit=np.reshape(rbfi(xyfit[:,0],xyfit[:,1]),img['height'],\
                                      img['width'])
    xyfit=0
    fitsurf=deepcopy(img)
    fitsurf['map']=float(zfit)
    zfit=0
   
    return fitsurf
   
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fillspots(img,spotsize=3):
    """
    remove tiny blind spots in a scattering pattern
    where the intensity is zero.
    """
    #Make a copy of the scattering pattern
    res = deepcopy(img)
    # Apply a median filter that will probably fill
    #empty pixels with a reasonable value from nearby
    res['map']=signal.medfilt2d(img['map'],kernel_size=spotsize)
        
    return res
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def killcircle(img,cenx,ceny,radius):
    """
    Kill the unvalid data in the circle.
    """
    imgmask=deepcopy(img)
    y,x=np.ogrid[-ceny:img['height']-ceny,-cenx:img['width']-cenx]
    #print len(y),len(x)
    y,x=y+0.5,x+0.5
    mask=np.where(x*x+y*y <= radius**2)
    arr=np.ones((img['height'],img['width']))
    arr[mask]=0
    imgmask['map']=imgmask['map']*arr
    return imgmask
 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def killcircleout(img,cenx,ceny,radius):
    """
    It is used to mask the image.
    This function mask the value outside the defined circle as
    0
    """ 
    imgmask=deepcopy(img)
    y,x=np.ogrid[-ceny:img['height']-ceny,-cenx:img['width']-cenx]
    y,x=y+0.5,x+0.5
    #mask the circle
    mask=np.where(x*x+y*y <= radius**2)
    arr=np.zeros((img['height'],img['width']))
    arr[mask]=1
    #print arr.shape
    imgmask['map']=imgmask['map']*arr
    return imgmask
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def shiftxy(ima,xy=[0,0]):
    """
    Shift the img by delx and dely.
    """
    img=deepcopy(ima)
    img['map']=interpolation.shift(ima['map'],xy,order=0,prefilter=0)
    img['center'][0]=ima['center'][0]+xy[1]
    img['center'][1]=ima['center'][0]+xy[0]
    return img
def azimrot(ima,degazm):
    """
    Rotate the pattern azimuthally by degazm.
    """
    img=deepcopy(ima)
    img['map']=interpolation.rotate(ima['map'],degazm,reshape=0,order=0,\
                    prefilter=0)
    return img
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
