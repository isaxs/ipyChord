#!/usr/bin/env python
import numpy as np
from copy import deepcopy
from sf_circle import sf_circle
def sf_killcircleout(img,ss=None):
    """
    This function is used to kill the outside part of the circle.
    """
    imgmask=deepcopy(img)
    if ss is None:
       ss=sf_circle(img)
       y,x=np.ogrid[-ss[1]:img['height']-ss[1],\
                       -ss[0]:img['width']-ss[0]]
       mask=np.where(x*x+y*y <= ss[2]**2)
       arr=np.zeros((img['height'],img['width']))
       arr[mask]=1
       imgmask['map']=img['map']*arr
       tmp=deepcopy(img)
       tmp['map']=arr
    else:
       y,x=np.ogrid[-ss[1]:img['height']-ss[1],\
                       -ss[0]:img['width']-ss[0]]
       #print len(y),len(x)
       mask=np.where(x*x+y*y <= ss[2]**2)
       #print mask[0,0]
       arr=np.zeros((img['height'],img['width']))
       arr[mask]=1
       imgmask['map']=img['map']*arr
       tmp=deepcopy(img)
       tmp['map']=arr
    return imgmask,tmp
