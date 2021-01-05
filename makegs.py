#!/usr/bin/env python
from CDF_2D import pklread,fibproj,ference,chord,cutwin,tolog,pklwrite
from CDF_2D import fillspots
from sf_show import sf_show
from misc import sg_smooth2d
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def makegs(datei,maxit=2,fast=None):
    """
    this file is used to show the main procedures of converting the 
    harm img to cdf image
    This is a batch file which does the processing from a complete
    harmonized scattering pattern (all pixels valid) up to the chord
    distribution. It is intended to be used within a batch of processor
    which will process a complete series of data (naming convention:
    CDF_2D.allharm2cdf)
    maxit,it means the times of the background scattering subtraction
          from the fiber projected saxs pattern. 
    """
    print('****Processing****file ',datei, ' ****')
    a=pklread(datei)
    #---------------------------------------------------------
    #the harmonized images are already filled in the center, but
    #the apron in the border still carries undefined "zero-values"
    #
    #in the most simple case, we assume that all the relevant scattering
    #has already decayed in a quadratic zone that is inside the 'circular'
    #vacuum tube (rule of thumb: 3x shortest long period position).
    # In this
    #case:simply cut the image down instead of carrying around a very long
    #border zone, e.g.:
    #a=sf_cutwin(aa,740,740)
    #
    #if you decide to consider the following case, then put a comment
    #before the line above, and remove all the '##' from the following
    #lines
    #
    #on the other hand, we consider that we should have more of the tail
    #of the background in the image, we will have to extrapolate into the 
    #apron zone that is still empty.
    #we anticipate that the files here are still big (1200*1200),but they
    #will be shrinked to 512*512 in makegs() anyway. So we can speed up,
    #if we shtink already here.
    #shrink the image
    #a=shrink(aa)
    #
    #The images look quite good: The valid scattering data have almost
    #vanished to zero at the border of the valid zone, and the invalid 
    #apron is not very big. So we decide to fill the narrow apron zone.
    #
    #On the other hand, if the apron zone is too narrow, we would first
    #'CDF_2D.widen apron zone so that it becomes wide(e.g. if there is 
    #a reflection close to the edge of the vacuum tube)
    #
    #Generate a fit of the apron zone recursively until the data in the 
    #corner do no longer fall negative
    
    #apply a very faint smoothing
    a=fillspots(a,5)
    #a=sg_smooth2d(a,21,1)
    #
    #sf_show(tolog(a),win=0,auto=1)
    print('====Start fiber diagram projection')
    a=fibproj(a)
    print(a['filename'],' has been projected onto a 2d plane')
    sf_show(a,win=1,auto=1,log=1)
    #
    print('=====Start computation of interference function=====')
    gs=ference(a,maxit=maxit)
    #
    #save interference function G(s) and show it
    #print type(gs)#,gs['filename']+'_gsf'
    gs['filename']=gs['title']+'_gsf.pkl'
    #pklwrite(gs,gs['filename'])
    sf_show(gs,win=2,auto=1)
    #sf_show(tolog(tolog(low)))
    print('==G(s) Done')
    
    print('=====Start computation of chord distribution====')
    cdf=chord(gs,4,fast=fast,outputorig=1) #triple zeros padding
    cdf['filename']=cdf['title']+'_CDF.pkl'
    print('The width and height of cdf is '+str(cdf['height']))
    cdf=cutwin(cdf,4096,4096)
    #pklwrite(cdf,cdf['filename'])
    sf_show(cdf,log=1,win=3,auto=1,neg=1)
    print('== CDF Done ==')
