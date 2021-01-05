#!/usr/bin/env python
import numpy as np
from copy import deepcopy
from sf_circlenew import sf_circlenew
from sf_show import sf_show
def sf_killcircle(img,log=None,ss=None,cm=None):
    """
    This function is used to kill the circle from the image.
    """
    imgmask=deepcopy(img)
    if ss is None:
       sf_show(imgmask,log=log,cm=cm,block=False)
       ss=sf_circlenew()
       y,x=np.ogrid[-ss[1]:img['height']-ss[1],\
                       -ss[0]:img['width']-ss[0]]
       mask=np.where(x*x+y*y <= ss[2]**2)
       arr=np.ones((img['height'],img['width']))
       arr[mask]=0
       imgmask['map']=img['map']*arr
       
    else:
       y,x=np.ogrid[-ss[1]:img['height']-ss[1],\
                       -ss[0]:img['width']-ss[0]]
       #print len(y),len(x)
       mask=np.where(x*x+y*y <= ss[2]**2)
       #print mask[0,0]
       arr=np.ones((img['height'],img['width']))
       arr[mask]=0
       imgmask['map']=img['map']*arr
    
    mask=deepcopy(img)
    mask['map']=arr
    return imgmask,mask
