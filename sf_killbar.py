#!/usr/bin/env python
import cv2
from copy import deepcopy
from matplotlib.colors import ListedColormap
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
from sf_bar import sf_bar
from sf_show import sf_show
def sf_killbar(img,log=None,polygon=None,cm=None):
    """
    This function is used to kill the bar-shaped blind area
    and set the relative area as False
    
    global fig,ax,ss,res,cmapp,statusL,statusM,axv,axh,polygondraw
    res=deepcopy(img)
    res['map']=np.double(res['map'])
    w,h=figure.figaspect(img['map'])
    fig=plt.figure(img['filename'],figsize=(1.2*w,1.2*h))
    ax=fig.add_axes([0,0,1,1])
    #load the color lookup table from  a pv-wave text file
    lut=sf_readclut('lut05.dat')
    clut=np.double(lut)/255
    cmapp=ListedColormap(zip(clut[0],clut[1],clut[2]),N=256)
    if grey is not None:
       cmapp=cm.Greys_r
    if log is None:
       ax.imshow(img['map'],interpolation='nearest',cmap=cmapp)
    else:
       ax.imshow(tolog(img)['map'],interpolation='nearest',cmap=cmapp)
    #set some parameters to control the property of image.
    plt.rcParams['toolbar']='None'
    plt.axis('off')
    statusL,statusM=None,None
    ss=[]
    print 'Drawing a polygon, you should select at least 3 points'
    
    def onmouse(event):
        global fig,ax,ss,res,cmapp,statusL,statusM,axh,axv,polygondraw
        if statusL==None:
           if event.button==1:
              x,y=np.around(event.xdata),np.around(event.ydata)
              print 'Coordinate of current point', x, y
              ss.append([x,y])
              axv=ax.axvline(np.around(event.xdata))
              axh=ax.axhline(np.around(event.ydata))
              fig.canvas.draw()
              #remove the vertical and horizontal lines
              axh.remove()
              axv.remove()
              #print ss
              statusL=None
           elif event.button==2:
              #polygon=np.array([ss],dtype=np.int32)
              #cv2.fillPoly(res['map'],polygon,0)
              #ax.imshow(res['map'],interpolation='nearest',cmap=cmapp)
              polygondraw=plt.Polygon(ss,fill=None,edgecolor='b')
              ax.add_patch(polygondraw)
              #show the patch
              fig.canvas.draw()
              print 'Press the middle button to remove the bad bar'
              print 'Press the right button to quit and return the res'
              statusL=1
              statusM=1
           elif event.button==3:
              plt.close()
              #returen res
        elif statusM==1:
           if event.button==2:
              res=deepcopy(img)
              #ax.imshow(res['map'],interpolation='nearest',cmap=cmapp)
              #show the original pattern
              polygondraw.remove()
              fig.canvas.draw()
              statusL=None
              ss=[]
              print 'Press the left button to draw a new bar'
           elif event.button==3:
              plt.close()

    cid=fig.canvas.mpl_connect('button_press_event',onmouse)
    plt.show(block=True)
    #remove the polygon area
    tmp=np.ones((res['height'],res['width']))
    polygon=np.array([ss],dtype=np.int32)
    cv2.fillPoly(tmp,polygon,0.0)
    mask=deepcopy(img)
    mask['map']=tmp
    res['map']=res['map']*tmp
    
    return res,mask
    """
    imgmask=deepcopy(img)
    if polygon is None:
       sf_show(imgmask,log=log,cm=cm,block=False)
       polygon=sf_bar()
       arr=np.ones((imgmask['height'],imgmask['width']))
       cv2.fillPoly(arr,polygon,0)
       imgmask['map']=img['map']*arr
    else:
       arr=np.ones((imgmask['height'],imgmask['width']))
       cv2.fillPoly(arr,polygon,0)
       imgmask['map']=img['map']*arr

    mask=deepcopy(img)
    mask['map']=arr
    return imgmask,mask
#def sf_readclut(filename):
#    """
#    this file is used to load the color lookup table
#    it only canbe used for lut05.dat file
#    """
#    fid=open(filename,'r')
#    clt=np.loadtxt(fid,dtype=int)
#    fid.close()
#    return clt
