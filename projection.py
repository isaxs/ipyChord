#!/usr/bin/env python
#this file is use to store the projection-like functions
######################################################################
#sf_vp()
#sf_hp()
#sf_rp()
#fib2iso1()
#fiblorentz()
#azimavgcur()
#azimI2theta()
#azimcurve()
#azimhalo()
#azimavgimg()
#waxs_distcali()
#polar()
#rect()
#sf_vs()
#sf_hs()
#####################################################################
import numpy as np
import scipy.ndimage.interpolation as interpolation
from CDF_2D import dist
from copy import deepcopy
from scipy import signal
from misc import cendist,curv_rmnan
import math
from sf_show import sf_show
#####################################################################
def sf_vp(ima,begcol=0):
    """
    Project an aligned (preferably harmonized)  scattering
    pattern on the vertical direction
    Input:
    img, the harmonized image
    begcol,  an optional positive integer value 
    designating the first column which
    should be taken into account for the
    projection. begcol must be smaller
    than img['center'][1].
    Output:
    curve = np.zeros((ndata,2)) a 2D array.
    curve(i,0) holds the x-values computed from img.BoxLen
    curve(i,1) holds the y-values computed by projecting the img['map']
               under consideration of the cylindrical
               of the scattering pattern.
    Hint: The number of data, ndata, in the curve
          can be obtained by the c.shape function.
    RESTRICTIONS:
    img['map'] should not contain negative values
    PROCEDURE:
    The lower left quadrant of the image is processed,
    beginning at (begcol,0) until (img['center'][0]-1,img['center'][1]-1))
    It is rotated into upper right position (we do not take this
    quadrant directly, since the finger of the beamstop is supposed
    to have been there). Then the projection is carried out.
    """
    img=deepcopy(ima)
    #check the begcol
    if begcol < 0:
       raise ValueError('begcol less than zero')
    if begcol >= img['center'][0]:
       raise ValueError('begcol beyond img center line')
    #make quadrant matrix for projection
    qmat = img['map'][ 0:img['center'][1], begcol:img['center'][0] ]
    #rotate qmat upside down, so that zero x- and y- indices
    #correspond to the "s-values" in the vicinity of the origin
    qmat=np.fliplr(np.flipud(qmat))
    #make the vector holding the 2*pi*s-values with which to multiply
    #each column of qmat before summing (because of the fiber
    #symmetry of the pattern)
    #mulv is the real s-value
    columns = img['center'][0] - begcol
    mulv = ( np.arange( columns ) + 0.5 ) * 2*np.pi*img['boxlen'][0]
    #Compute integrand in qmat
    for i in range(columns):
        qmat[ :, i ] = qmat[:,i]*mulv[i]
    #Make the curve to hold the result
    curve=np.zeros((img['center'][1],2),dtype=np.double)
    #Fill in the projected intensity
    #sum over 0th dimension
    curve[:,1] = np.sum( qmat,axis=1 )*img['boxlen'][0]
    #Fill in the "s-values" (or millimeters in vertical direction
    #on the image plate)
    curve[:,0] = (np.arange( img['center'][1] ) + 0.5) * img['boxlen'][1]
    return curve
    
#*******************************************************************
def sf_hp(ima,begrow=0):
    """
    Project an aligned (preferably harmonized)  scattering
    pattern on the vertical direction.
    Input:
    img, the harmonized image
    begrow,  an optional positive integer value
    designating the first column which
    should be taken into account for the
    projection. begcol must be smaller
    than img['center'][0].
    
    """
    img=deepcopy(ima)
    #check the begcol
    if begrow < 0:
       raise ValueError('begrow less than zero')
    if begrow >= img['center'][1]:
       raise ValueError('begrow beyond img center line')
    #make quadrant matrix for projection
    qmat = img['map'][0:img['center'][1],begrow:img['center'][0]]
    #rotate qmat upside down, so that zero x- and y- indices
    #correspond to the "s-values" in the vicinity of the origin
    qmat=np.fliplr(np.flipud(qmat))
    #Make the resulting curve
    curve=np.zeros((img['center'][0],2),dtype=np.double)
    #Fill in the projected intensity
    curve[:,1] = np.sum( qmat,axis=0 )
    #Fill in the "s-values" (or millimeters in vertical direction
    #on the image plate)
    curve[:,0] = (np.arange( img['center'][1] ) + 0.5) * img['boxlen'][1]
    
    return curve
#*********************************************************************
def sf_rp(img):
    """
    Projection of a pattern or a reflection on the radial direction
    Function for use with python. Performs an azimuthal
    integration resulting in a curve in radial direction
    Very similar to azimuthal averaging - but here the integral
    is not normalised with respect to the points having the same
    distance, but with respect to the circumference of the actual
    circle. This is, in fact, a radial projection sf_rp
    OUTPUT:
    curve = np.zeros(ndata,2) a 2D array.
            curve(i,0) holds the x-values computed from img['boxlen']
            curve(i,1) holds the y-values computed by averaging
    """
    if img['width'] != img['height']:
       raise ValueError('only for square maps!')
    if img['center'][0] != img['width']/2 or \
              img['center'][1] != img['height']/2:
       raise ValueError('only for centered image maps!')
    if img['boxlen'][0] != img['boxlen'][1]:
       raise ValueError('only for square pixels!')
    
    # under these premises it's easy to build a distance array
    dia2 = img['width']
    cen  = img['width']/2
    #no no no
    spot = interpolation.shift( dist(dia2),(cen,cen),order=0,prefilter=False)
    #Now let's make it integer
    intspot = np.reshape( np.arange(dia2**2),(dia2, dia2))
    spot=spot+0.5
    intspot = np.fix( spot )
    npoints = int(intspot.max())
    #In IntSpot there is now the index in the later curve that
    #corresponds to the position of the pixel.
    #
    #Make the resulting curve
    curve = np.zeros( ( npoints,2),dtype=np.float )
    #Fill it with data
    for I in range(npoints):
        # Radius and defaults
        R = I + 1
        curve[I,0] = R*img['boxlen'][1] # x value
        curve[I,1] = 0.0                # no data yet
        #Collect all indices, which correspond to the current radius
        indcoll = np.where( intspot == R )
        #Put these pixels into azimmap
        azimmap = img['map'][indcoll]
        #There may be pixels with the value zero. This means that the
        #data are invalid. Collect only the valid ones!
        goodcoll = np.where( azimmap > 0.0 )
        #print goodcoll
        #Now there may be no valid data at all. Check this!
        if goodcoll != () :
           #filter out the good ones
           goodmap = azimmap[goodcoll]
           #... and compute the result
           #Integral of intensities in the azimuthal scan
           # per circumference 2*PI*R
           curve[I,1] = np.sum( goodmap ) / R
           
    return curve
#*********************************************************************
def fib2iso1(img,lc=True):
    """
    Isotropise a pattern with fiber symmetry.
    Performs a fibre pattern integration followed by an azimuthal
    integral with increasing distance from the centre and thus 
    generates a curve which corresponds to the isotropised and "Lorentz-
    corrected" scattering curve. Thus the scattering curve that
    would have been measured is obtained after division by 4*Pi*s**2
    INPUT: img
    OUTPUT:
    curve = np.zeros((ndata,2),dtype=np.float) a 2D array. curve[i,0] 
    holds the x-values computed from img['boxlen']
    curve[i,1] holds the y-values computed by averaging
    """
    if img['width'] != img['height']:
       raise ValueError('only for square maps!')
    if img['center'][0] != img['width']/2 or \
            img['center'][1] != img['height']/2:
       raise ValueError('only for centered image maps!')
    if img['boxlen'][0] != img['boxlen'][1]:
       raise ValueError('only for square pixels!')

    # under these premises it's easy to build a distance array
    dia2 = img['width']
    cen  = img['width']/2
    spot = interpolation.shift( dist(dia2),(cen,cen),order=0,prefilter=False)
    #Now let's make it integer
    intspot = np.reshape(range(dia2**2),(dia2, dia2))
    spot=spot+0.5
    intspot = np.fix( spot )
    npoints = int(intspot.max())
    #print npoints
    #In IntSpot there is now the index in the later curve that
    #corresponds to the position of the pixel.
    #
    #Make the resulting curve
    curve = np.zeros( ( npoints,2),dtype=np.float )
    #Perform fibre integration in s_12-plane on a copy of
    #the image still omitting the factor Pi (remember: every
    #pixel is representative of the HALF of a circle only.
    imgi = deepcopy(img)
    truecen = img['center'][0] - 0.5
    # treat every column as a whole
    #for i in range(imgi['width']-1):
    #    imgi['map'][:,i]=imgi['map'][:,i]*abs(i-truecen)
    row,col=np.indices((img['height'],img['width']))
    #shift the array along the left-right direction
    s12col=np.absolute(col-truecen)*img['boxlen'][0]
    imgi['map']=imgi['map']*s12col
    #Fill the resulting curve with data by
    #performing azimuthal integration about \psi, the angle
    #with respect to the fibre axis as a function of increasing radius
    for I in range(npoints-1):
        #Radius and defaults
        R = I + 1
        curve[I,0] = R*imgi['boxlen'][1]  # x value
        curve[I,1] = 0.0                  # no data
        #Collect all indices, which correspond to the current radius
        indcoll = np.where( intspot == R )
        #Put these pixels into azimmap
        azimmap = imgi['map'][indcoll]
        #There may be pixels with the value zero. This means that the
        #data are invalid. Collect only the valid ones!
        goodcoll = np.where( azimmap > 0.0 )
        
        #Now there may be no valid data at all. Check this!
        if goodcoll != () :
           #filter out the good ones
           goodmap = azimmap[goodcoll]
           #and compute the results
           #Intensity is average of all the
           #good points in the azimuthal scan
           #times 2*Pi*curve(0,i) times the factor Pi
           #that was omitted above for speed
           curve[I,1] = 2*np.pi**2*curve[I,0]*goodmap.mean()
           #for test purpose: invert Lorentz correction and compare
           #to the result of sf_azimavg for isotropic patterns.
           if lc is not True:
              curve[I,1] = curve[I,1]/(2*np.pi*curve[I,0]**2)
    
    #shift the curve half-pixel left
    curve[:,0]=curve[:,0]-img['boxlen'][0]/2
    #remove the nan point in the curve
    #curve=curv_rmnan(curve)
    #return the Lorentz corrected isotropic intensity
    return curve
#*********************************************************************
def fiblorentz(img):
    """
    It multiplies. every column of the image by its distance 
    from image axis. Thus it does something like the "Lorentz 
    correction" does in the case of isotropic data, but now 
    for data with fiber. symmetry: Every value of the function
    is integrated along its revolutional circle.
    """
    resimg= deepcopy(img)
    columns = img['width']-1
    #Scatterers paradigm:
    center  = img['center'][0] - 0.5
    # Do the "correction"
    for I in range(columns):
        resimg['map'][:,I] = resimg['map'][:,I]*abs(I-center)
    return resimg
#*******************************************************************
def azimavgcur(img,waxs=None):
    """
    Performs an azimuthal average with increasing distance from 
    the center.
    It generates a curve.
    waxs is the distance from sample to detector. unit: pixel
    The default wavelength is 0.154nm
    """
    if img['width'] != img['height']:
       raise ValueError('only for square maps!')
    if img['center'][0] != img['width']/2 or \
         img['center'][1] != img['height']/2:
       raise ValueError('only for centered image maps!')
    if img['boxlen'][0] != img['boxlen'][1]:
       raise ValueError('only for square pixels!')
    #
    #check  the position of nan
    nanpos=np.argwhere(np.isnan(img['map']))
    img['map'][nanpos]=0.
    #build a distance array
    spot=cendist(img['width'])
    Intspot=np.fix(spot+0.5)
    npoints=int(Intspot.max())
    
    #prepare a curve
    cur=np.zeros((npoints,2),dtype=np.float)
    bl=img['boxlen'][0]
    #fill it with data
    for i in range(npoints):
        #get the radius
        R=i+1
        #get the position of radius
        pos=np.where(Intspot == R)
        #get the value of radius
        Rval=img['map'][pos]
        #if R > 200:
        #   img['map'][pos]=1.0
        #   print len(pos[0])
        #   sf_show(img,log=1,block=1)
        #mask the value
        goodpos=np.where(Rval > 0.0)
        #print len(goodpos[0])
        #if the value is NaN, fill it with 0
        Rvalgood=Rval[goodpos]
        Rvalgoodmean=np.abs(Rvalgood.mean())
        #print Rvalmean
        if math.isnan(Rvalgoodmean) :
           Rvalgoodmean=0.0
        if waxs is not None:
           cur[i,0]=(2.0/0.154)*np.sin(np.arctan((i+0.5)/waxs)/2.0)
           cur[i,1]=Rvalgoodmean
        else:
           cur[i,0],cur[i,1]=(i+0.5)*bl,Rvalgoodmean
        
    return cur
#**********************************************************************
def azimI2theta(img,dist):
    """
    This func returns the curve of I vs 2theat from the harmonized 2D
    WAXS pattern. dist is the distance form sample to detector. The returned
    curve has the un-equidistancial 2theta. The unit of dist is mm.
    """
    if img['width'] != img['height']:
       raise ValueError('only for square maps!')
    if img['center'][0] != img['width']/2 or \
         img['center'][1] != img['height']/2:
       raise ValueError('only for centered image maps!')
    if img['boxlen'][0] != img['boxlen'][1]:
       raise ValueError('only for square pixels!')
    #
    #check  the position of nan
    nanpos=np.argwhere(np.isnan(img['map']))
    img['map'][nanpos]=0.
    #build a distance array
    spot=cendist(img['width'])
    Intspot=np.fix(spot+0.5)
    npoints=int(Intspot.max())
    
    #prepare a curve
    cur=np.zeros((npoints,2),dtype=np.float)
    bl=img['boxlen'][0]
    dist_pix=dist*1000/(bl*1000000)
    #fill it with data
    for i in range(npoints):
        #get the radius
        R=i+1
        #get the position of radius
        pos=np.where(Intspot == R)
        #get the value of radius
        Rval=img['map'][pos]
        #mask the value
        goodpos=np.where(Rval > 0.0)
        #if the value is NaN, fill it with 0
        Rval=Rval[goodpos]
        Rvalmean=np.abs(Rval.mean())
        #print Rvalmean
        if math.isnan(Rvalmean) :
           Rvalmean=0.0
        cur[i,0]=np.arctan((i+0.5)/dist_pix)*57.2958
        cur[i,1]=Rvalmean
        
    return cur
#**********************************************************************
def azimcurve(img,radius,interactive=None):
    """
    Extract an azimuthal curve from anisotropic patterns
    radius-->If /interactive is not set, a positive radius must be given
    and the azimuthal curve through this radius is returned
    /interactive If this keyword is set, the radius is chosen
    interactively. The user inputs a circle (sf_killoutcircle), 
    and inside this circle the maximum defines the radius of the 
    azimuthal curve that is extracted. The chosen radius is printed.
    OUTPUT:
    a curve that can be plotted (sf_cplot) or written in ASCII format
    to a file (sf_cwrite) for documentation
    """
    if img['width'] != img['height']:
       raise ValueError('only for square maps!')
    if img['center'][0] != img['width']/2 or \
           img['center'][1] != img['height']/2:
       raise ValueError('only for centered image maps!')
    if img['boxlen'][0] != img['boxlen'][1]:
       raise ValueError('only for square pixels!')
    
    if interactive is not None:
       dum=deepcopy(img)
       print('Encircle a maximum that defines the radius!')
       dum=sf_killcircleout(dum)
       dummax = dum['map'].max()
       mcoll = np.where( dum['map'] == dummax )
       xl = mcoll[0] - img['center'][0]
       #compute y values - with respect to the image center
       yl = mcoll[1] - img['center'][1]
       #Convert to polar coordinates
       pcoll = polar(xl,yl,deg=1)
       r = np.fix( pcoll[:,1].avg() + 0.5 )
       print('Radius determined: ', r)
    else:
       if radius < 2:
          print('radius too small or negative')
       r=radius

    #under these premises it's easy to build a distance array
    dia2 = img['width']
    cen  = img['width']/2
    spot = interpolation.shift(dist(dia2),(cen,cen),order=0,prefilter=False)
    #Now let's make it intege
    intspot = np.reshape( range(dia2**2),(dia2, dia2))
    spot=spot+0.5
    intspot = np.fix( spot )
    c = 0 # trivial result
    #Collect all indices, which correspond to the current radius
    #np is the number of points in the collection of indices
    indcoll = np.where(intspot==r)
    #Put the intensities of these pixels into an intensity collection
    icoll = img['map'](indcoll)
    #how many intensities are above valid (i.e. above 0)?
    goodcoll = np.where( icoll > 0.0)
    
    #Skip, if there are not enough valid data in the ring
    if ng > 5:
       # compute x values - with respect to the image center
       xl = np.mod(indcoll[:],img['width'])-img['center'][0]
       #compute y values - with respect to the image center
       yl = np.fix(indcoll[:]/img['width']) - img['center'][1]
       #Convert to polar coordinates
       pcoll = polar(xl,yl,deg=1)
       #Sort the pollist in the order of ascending angles
       sortangles = sorted( pcoll[:,0] )
       #The intensity data along the circle
       c = np.zeros((len(indcoll[0]),2),dtype=np.float)
       c[:,0] = pcoll[sortangles[:],0]
       c[:,1] = icoll[sortangles[:]]
    else:
       print('+++ Trivial curve: not enough data!')

    return c
#**********************************************************************
def azimhalo(img, smooth=None):
    """
    Compute isotropic halo of WAXS pattern
    Collects data in azimuthal rings
    and puts the minimum found in every azimuthal ring in order to
    generate the isotropic fraction in a scattering pattern
    /smooth :  the isotropic background is taken from the smoothed
               instead of from the de-noised image
    """
    if img['width'] != img['height']:
       raise ValueError('only for square maps!')
    if img['center'][0] != img['width']/2 or \
         img['center'][1] != img['height']/2:
       raise ValueError('only for centered image maps!')
    if img['boxlen'][0] != img['boxlen'][1]:
       raise ValueError('only for square pixels!')

    #
    dia2 = img['width']
    cen  = img['width']/2
    #but we do not need the distances from the corners, but
    #the distances from the centre of the image
    spot = interpolation.shift(dist(dia2),(cen,cen),order=0,prefilter=False)
    
    # Now let's make it integer
    intspot = np.reshape( range(dia2**2),(dia2, dia2))
    spot=spot+0.5
    intspot = np.fix( spot )
    npoints=intspot.max()
    #make the rsults
    azimg = deepcopy(img)
    #Pick data from a filtered map
    if smooth is not None:
       imfil = sf_smooth( img )
    else:
       imfil = deepcopy(img)
       imfil['map'] = signal.medfilt2d( img['map'], kernel_size=7 ) 
       # only the noiseless minimum
    for I in range(npoint-1):
        #Radius and defaults
        R = I + 1
        #Collect all indices, which correspond to the current radius
        indcoll = np.where( intspot == R )
        #Put these pixels into azimmap
        azimmap = imfil['map'][indcoll] # From the filtered image
        #There may be pixels with the value zero. This means that the
        #data are invalid. Collect only the valid ones!
        goodcoll = np.where( azimmap == 0.0 )
        # Now there may be no valid data at all. Check this!
        if goodcoll != ():
           #filter out the good ones
           goodmap = azimmap[goodcoll]
           #and compute the result
           #Intensity is minimum of all the
           #good points in the azimuthal scan
           #put them in azimg at the all the indices collected in IndColl
           azimg['map'][indcoll] =  goodmap.min()
           
    return azimg
#*********************************************************************
def azimavgimg(img):
    """
    Azimuthal average with increasing distance from the center for 
    isotropic pattern. The output here is an azimuthally averaged image.
    """
    if img['boxlen'][0] != img['boxlen'][1] :
       raise ValueError('Only for square pixels')
    if img['center'][0] != img['width']/2 :
       raise ValueError('Only for centered pattern')
    if img['height'] != img['width']:
       raise ValueError('Only for square maps')
    #build a distance array
    spot=cendist(img['width'])
    Intspot=np.fix(spot+0.5)
    npoints=int(Intspot.max())
    
    #Make the resulting image
    azimg=deepcopy(img)
    #fill it with data
    for i in range(npoints):
        #get the radius
        R=i+1
        #get the position of radius
        pos=np.where(Intspot == R)
        #get the value of radius
        Rval=img['map'][pos]
        #mask the value
        goodpos=np.where(Rval > 0.0)
        #if the value is NaN, fill it with 0
        Rval=Rval[goodpos]
        Rvalmean=np.abs(Rval.mean())
        #print Rvalmean
        if math.isnan(Rvalmean) :
           Rvalmean=0.0
        azimg['map'][pos]=Rvalmean
    return azimg
#*****************************************************************************
def waxs_distcali(wax,dist):
    """
    This func is to determine the distance from the sample to detector.
    """
    #get the waxs curve, but the unit of x is pixel
    wcur=azimavgcur(wax,waxs=1)
    #keep the original data
    aequcurv=deepcopy(wcur)
    #get the length of curve
    clen=wcur.shape[0]
    for i in range(clen):
        aequcurv[i,0]=math.sin(0.5*math.atan((i+0.5)/dist))*2.0/0.154
    return aequcurv    
#*****************************************************************************
def polar(x, y, deg=0):
    """
    Convert from rectangular (x,y) to polar (r,w)
    r = sqrt(x^2 + y^2)
    w = arctan(y/x) = [-\pi,\pi] = [-180,180]
    # radian if deg=0; degree if deg=1
    """
    if deg:
       return np.hypot(x, y), 180.0 * np.atan2(y, x) /np.pi
    else:
       return np.hypot(x, y), np.atan2(y, x)
#********************************************************************
def rect(r, w, deg=0):		
    """
    # radian if deg=0; degree if deg=1
    Convert from polar (r,w) to rectangular (x,y)
    x = r cos(w)
    y = r sin(w)
    """
    if deg:
       w = np.pi * w / 180.0
    return r * np.cos(w), r * np.sin(w)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sf_hs(img,cut=None,half=None):
    """
    Cuts a scattering pattern in a row and returns the resulting
    curve. 
    INPUT:
       img  a dict, which is aligned and preferably harmonized.
            The img['center'] field has to be set correctly.
       cutrow an optional positive integer value
            designating the row which holds the data
            of the section. If cutrow is not supplied,
            the picture is cut in the central row,  img['center'][1].
       cut is the value deviated from the center of the pattern.
           input=math.fabs(cut-img['center'][1]).
    OUTPUT:
        curve = np.zeros((ndata,2)) a 2D array.
        curve[i,0] holds the x-values computed from img.BoxLen
        curve[i,1] holds the y-values computed by projecting the img['map']
    
    """
    #Make the resulting curve
    if cut is not None:
       cutrow=int(math.fabs(cut-img['center'][1]))
    else:
       cutrow=img['center'][1]
    #preset the curve
    curve = np.zeros((img['width'],2),dtype=np.float)
    #Fill in the intensity from the row
    curve[:,1] = img['map'][cutrow,:]
    #Fill in the "s-values" (or millimeters in vertical direction
    #on the image plate)
    curve[:,0]=(1.0*np.arange(img['width'])+0.5-img['center'][0])\
                     *img['boxlen'][0]
    if half is not None:
       halflen=len(curve[:,1])/2
       x=curve[:,0][halflen::]
       y=curve[:,1][halflen::]
       curve = np.zeros((img['width']/2,2),dtype=np.float)
       curve[:,0]=x
       curve[:,1]=y
    return curve
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sf_vs(img,cut=None,halfshift=None,half=None):
    """
    Cuts a scattering pattern in a row and returns the resulting
    curve. 
    INPUT:
       img  a dict, which is aligned and preferably harmonized.
            The img['center'] field has to be set correctly.
       cutcol an optional positive integer value
            designating the col which holds the data
            of the section. If cutcol is not supplied,
            the picture is cut in the central col,  img['center'][1].
    OUTPUT:
       curve = np.zeros((ndata,2)) a 2D array.
       curve[i,0] holds the x-values computed from img['boxlen']
       curve[i,1] holds the y-values computed by sliceing the img['map']
    """
    if cut is not None:
       cutcol=int(math.fabs(cut-img['center'][0]))
    else:
       cutcol=img['center'][0]
    #Make the resulting curve
    curve = np.zeros((img['width'],2),dtype=np.float)
    #Fill in the intensity from the row
    curve[:,1] = img['map'][:,cutcol]
    #Fill in the "s-values" (or millimeters in vertical direction
    #on the image plate)
    curve[:,0]=(1.0*np.arange(img['height'])+0.5-img['center'][1])\
                      *img['boxlen'][1]
    if halfshift is not None:
       #Required for data that should be processed by a
       #Fourier transform routine
       csav = deepcopy(curve)
       curve = np.zeros((img['height']-1,2))
       for i in range(img['height']-1):
           curve[i,1]=(csav[i,1]+csav[i+1,1])*0.5
       curve[:,0]=(np.arange(img['height']-1)+1 -\
                            img['center'][1]) * img['boxlen'][1]
    if half is not None:
       halflen=len(curve[:,1])/2
       x=curve[:,0][halflen::]
       y=curve[:,1][halflen::]
       curve = np.zeros((img['width']/2,2),dtype=np.float)
       curve[:,0]=x
       curve[:,1]=y
    return curve
