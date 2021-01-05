#!/usr/bin/env python
import numpy as np
import math
import scipy.ndimage.interpolation as interpolation
import scipy.misc as misc
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from CDF_2D import tolog
from copy import deepcopy
from sf_show import sf_show
from sf_readclut import sf_readclut
def sf_imgrot(img,cc=[],grey=None,cross=None,zoom=1.4):
    """
    This function is used to rotate the image about its center
    It will not broaden the image
    """
    global fig, statusL,rotpoint,ax,ss,preharm,cmapp
    preharm=deepcopy(img)
    if cc == []:
       w,h=figure.figaspect(img['map'])
       fig=plt.figure(img['filename'],figsize=(zoom*w,zoom*h))
       ax=fig.add_axes([0,0,1,1])
       #load the color lookup table from  a pv-wave text file
       lut=sf_readclut('lut05.dat')
       clut=np.double(lut)/255
       cmapp=ListedColormap(zip(clut[0],clut[1],clut[2]),N=256)
       if grey is not None:
          cmapp=cm.Greys_r
       ax.imshow(preharm['map'],\
                         interpolation='nearest',cmap=cmapp)
       if cross is not None:
          ax.axhline(preharm['width']/2-0.5)
          ax.axvline(preharm['height']/2-0.5)
       plt.rcParams['toolbar']='None'
       plt.axis('off')
       statusL=None
       ss=[0.0]
       
       def onmouserot(event):
           global fig,statusL,rotpoint,ax,ss,preharm,cmapp
           if statusL==None:
              if event.button==1:
                 rotpoint=[event.xdata,event.ydata]
                 rotpoint[0]=rotpoint[0]-preharm['center'][0]
                 rotpoint[1]=rotpoint[1]-preharm['center'][1]
                 rotangle=math.degrees(math.atan2(-rotpoint[1],\
                                                        rotpoint[0]))
                 preharm['map']=interpolation.rotate(preharm['map'],\
                         rotangle,reshape=False,order=0,prefilter=False)
                 ax.imshow(preharm['map'],\
                                    interpolation='nearest',cmap=cmapp)
                 if cross is not None:
                    ax.axhline(preharm['width']/2-0.5)
                    ax.axvline(preharm['height']/2-0.5)
                 fig.canvas.draw()
                 varangle=rotangle
                 ss[0]=ss[0]+varangle
                 #print ss
                 print 'Rotate angle: ', varangle
                 statusL=1
              elif event.button==3:
                 plt.close()
           elif statusL==1:
              if event.button==1:
                 rotpoint=[event.xdata,event.ydata]
                 rotpoint[0]=rotpoint[0]-preharm['center'][0]
                 rotpoint[1]=rotpoint[1]-preharm['center'][1]
                 rotangle=math.degrees(math.atan2(-rotpoint[1],\
                                                      rotpoint[0]))
                 preharm['map']=interpolation.rotate(preharm['map'],\
                             rotangle,reshape=False,order=0,prefilter=False)
                 ax.imshow(preharm['map'],\
                                 interpolation='nearest',cmap=cmapp)
                 if cross is not None:
                    ax.axhline(preharm['width']/2-0.5)
                    ax.axvline(preharm['height']/2-0.5)
                 fig.canvas.draw()
                 varangle=rotangle
                 ss[0]=ss[0]+varangle
                 #print ss
                 print 'Rotate angle: ', varangle
              elif event.button==3:
                 statusL=None
                 plt.close()                 
       fig.canvas.mpl_connect('button_press_event',onmouserot)
       plt.show(block=True)
       ss[0]=round(ss[0])
       return preharm,ss
    else:
       preharm['map']=interpolation.rotate(preharm['map'],ss[0],\
                             reshape=False,order=0,prefilter=False)
       w,h=figure.figaspect(img['map'])
       fig=plt.figure(img['filename'],figsize=(zoom*w,zoom*h))
       ax=fig.add_axes([0,0,1,1])
       #load the color lookup table from  a pv-wave text file
       lut=sf_readclut('lut05.dat')
       clut=np.double(lut)/255
       cmapp=ListedColormap(zip(clut[0],clut[1],clut[2]),N=256)
       if grey is not None:
          cmapp=cm.Greys_r
       ax.imshow(preharm['map'],\
                         interpolation='nearest',cmap=cmapp)
       if cross is not None:
          ax.axhline(preharm['width']/2-0.5)
          ax.axvline(preharm['height']/2-0.5)
       plt.rcParams['toolbar']='None'
       plt.axis('off')
       statusL=None
       
       def onmouserot(event):
           global fig,statusL,rotpoint,ax,ss,preharm,cmapp
           if statusL==None:
              if event.button==1:
                 rotpoint=[event.xdata,event.ydata]
                 rotpoint[0]=rotpoint[0]-preharm['center'][0]
                 rotpoint[1]=rotpoint[1]-preharm['center'][1]
                 rotangle=math.degrees(math.atan2(-rotpoint[1],\
                                                        rotpoint[0]))
                 preharm['map']=interpolation.rotate(preharm['map'],\
                                rotangle,reshape=False,order=0,prefilter=False)
                 ax.imshow(preharm['map'],\
                                    interpolation='nearest',cmap=cmapp)
                 if cross is not None:
                    ax.axhline(preharm['width']/2-0.5)
                    ax.axvline(preharm['height']/2-0.5)
                 fig.canvas.draw()
                 varangle=rotangle
                 ss[0]=ss[0]+varangle
                 #print ss
                 print 'Rotate angle: ', varangle
                 statusL=1
              elif event.button==3:
                 plt.close()
           elif statusL==1:
              if event.button==1:
                 rotpoint=[event.xdata,event.ydata]
                 rotpoint[0]=rotpoint[0]-preharm['center'][0]
                 rotpoint[1]=rotpoint[1]-preharm['center'][1]
                 rotangle=math.degrees(math.atan2(-rotpoint[1],\
                                                      rotpoint[0]))
                 preharm['map']=interpolation.rotate(preharm['map'],\
                             rotangle,reshape=False,order=0,prefilter=False)
                 ax.imshow(preharm['map'],\
                                 interpolation='nearest',cmap=cmapp)
                 if cross is not None:
                    ax.axhline(preharm['width']/2-0.5)
                    ax.axvline(preharm['height']/2-0.5)
                 fig.canvas.draw()
                 varangle=rotangle
                 ss[0]=ss[0]+varangle
                 #print ss
                 print 'Rotate angle: ', varangle
              elif event.button==3:
                 statusL=None
                 plt.close()                
       fig.canvas.mpl_connect('button_press_event',onmouserot)
       plt.show(block=True)
       ss[0]=round(ss[0])
       return preharm,ss
       

#def sf_readclut(filename):
#    """
#    this file is used to load the color lookup table
#    it only canbe used for lut05.dat file
#    """
#    fid=open(filename,'r')
#    clt=np.loadtxt(fid,dtype=int)
#    fid.close()
#    return clt

