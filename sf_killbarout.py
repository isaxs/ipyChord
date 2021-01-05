#!/usr/bin/env python
import cv2
from copy import deepcopy
from CDF_2D import tolog
from matplotlib.colors import ListedColormap
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from sf_show import sf_show
from sf_readclut import sf_readclut
def sf_killbarout(img,log=None,grey=None):
    """
    This function is used to kill the outside of bar-shaped blind area
    and set the relative area as True or False
    """
    global fig,ax,ss,tmp,cmapp,statusL,statusM,axv,axh,polygondraw,polygon
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
    plt.rcParams['toolbar']='None'
    plt.axis('off')
    statusL,statusM=None,None
    ss=[]
    def onmouse(event):
        global fig,ax,ss,tmp,cmapp,statusL,statusM,axh,axv,polygondraw,\
               polygon
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
              polygon=np.array([ss],dtype=np.int32)
              #cv2.fillPoly(tmp['map'],polygon,0)
              #ax.imshow(tmp['map'],interpolation='nearest',cmap=cmapp)
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
              #tmp=deepcopy(img)
              #ax.imshow(tmp['map'],interpolation='nearest',cmap=cmapp)
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
    res=deepcopy(img)
    tmp=deepcopy(img)
    #get an zero array
    tmp['map']=np.zeros((res['height'],res['width']),dtype=np.float)
    cv2.fillPoly(tmp['map'],polygon,1.0)
    res['map']=res['map'] * (1.0-tmp['map'])
    #print 'Press the right button to close the window'
    #sf_show(res)
    return res,tmp

#def sf_readclut(filename):
#    """
#    this file is used to load the color lookup table
#    it only canbe used for lut05.dat file
#    """
#    fid=open(filename,'r')
#    clt=np.loadtxt(fid,dtype=int)
#    fid.close()
#    return clt
