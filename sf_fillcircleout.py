#!/usr/bin/env python
import numpy as np
from copy import deepcopy
from sf_circle import sf_circle
def sf_fillcircleout(img,ss=None):
    """
    This function is used to fill the area outside the circle
    """
    imgmask=deepcopy(img)
    if ss is None:
       ss=sf_circle(img)
       #print ss
       y,x=np.ogrid[-ss[1]:img['height']-ss[1],\
                       -ss[0]:img['width']-ss[0]]
       mask=np.where(x*x+y*y <= ss[2]**2)
       arr=np.zeros((img['height'],img['width']))
       arr[mask]=1
       #keep the circle inside as before
       imgmask['map']=imgmask['map']*arr
       newarr=np.ones((img['height'],img['width']))
       newarr[mask]=0
       #make outside as 1
       #print 'Done'
       imgmask['map']=imgmask['map']+newarr
    else:
       y,x=np.ogrid[-ss[1]:img['height']-ss[1],\
                       -ss[0]:img['width']-ss[0]]
       #print len(y),len(x)
       mask=np.where(x*x+y*y <= ss[2]**2)
       #print mask[0,0]
       arr=np.zeros((img['height'],img['width']))
       arr[mask]=1
       imgmask['map']=imgmask['map']*arr
       newarr=np.ones((img['height'],img['width']))
       newarr[mask]=0
       imgmask['map']=imgmask['map']+newarr
    return imgmask
