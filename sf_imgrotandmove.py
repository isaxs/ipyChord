#!/usr/bin/env python
import numpy as np
import math
import scipy.ndimage.interpolation as interpolation
import scipy.misc as misc
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from CDF_2D import tolog
from copy import deepcopy
from sf_show import sf_show
from sf_readclut import sf_readclut
def sf_imgrotandmove(img,cc=None,log=None,cm=None,clip=True):
    global fig, statusL,statusR,rotpoint,refpoint,ax,ss,\
           origpoint,preharm,shiftdist,cmapp
    preharm=deepcopy(img)
    if clip == True:
       preharm['map'] = img['map'] * (img['map'] >= 0.0 )
    if cc == []:
       w,h=figure.figaspect(img['map'])
       fig=plt.figure(img['filename'],figsize=(w,h))
       ax=fig.add_axes([0,0,1,1])
       #load the color lookup table from  a pv-wave text file
       lut=sf_readclut('lut05.dat')
       clut=np.double(lut)/255
       cmapp=ListedColormap(zip(clut[0],clut[1],clut[2]),N=256)
       if cm is not None:
          cmapp=plt.get_cmap(name=cm.strip())
       if log is not None:
          ax.imshow(tolog(preharm)['map'],\
                          interpolation='nearest',cmap=cmapp)
       else:
          ax.imshow(preharm['map'],interpolation='nearest',cmap=cmapp)
       #ax.plot(img['center'][0],img['center'][1],"y+")
       ax.axhline(img['width']/2-0.5)
       ax.axvline(img['height']/2-0.5)
       circ0=plt.Circle((img['center'][0],img['center'][1]),radius=\
                         img['width']/4,fill=False,edgecolor='b')
       circ1=plt.Circle((img['center'][0],img['center'][1]),radius=\
                         img['width']/8,fill=False,edgecolor='b')
       ax.add_patch(circ0)
       ax.add_patch(circ1)
       plt.rcParams['toolbar']='None'
       plt.axis('off')
       statusL=None
       statusR=None
       ss=[0,0,0]
       
       def onmouserot(event):
           global fig, statusL,rotpoint,statusR,origpoint,\
                   refpoint,ax,ss,preharm,shiftdist,cmapp
           if statusL==None:
              if event.button==1:
                 rotpoint=[event.xdata,event.ydata]
                 rotpoint[0]=rotpoint[0]-preharm['center'][0]
                 rotpoint[1]=rotpoint[1]-preharm['center'][1]
                 rotangle=math.degrees(math.atan2(-rotpoint[1],\
                                                     rotpoint[0]))
                 preharm['map']=interpolation.rotate(preharm['map'],\
                           rotangle,reshape=False,prefilter=False,order=0)
                 #tmp=deepcopy(preharm)
                 if log is not None:
                    ax.imshow(tolog(preharm)['map'],\
                                  interpolation='nearest',cmap=cmapp)
                 else:
                    ax.imshow(preharm['map'],interpolation='nearest',\
                                         cmap=cmapp)
                 fig.canvas.draw()
                 varangle=rotangle
                 ss[2]=ss[2]+varangle
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
                            rotangle,reshape=False,prefilter=False,order=0)
                 #tmp=deepcopy(preharm)
                 if log is not None:
                    ax.imshow(tolog(preharm)['map'],\
                               interpolation='nearest',cmap=cmapp)
                 else:
                    ax.imshow(preharm['map'],\
                               interpolation='nearest',cmap=cmapp)
                 fig.canvas.draw()
                 varangle=rotangle
                 ss[2]=ss[2]+varangle
                 print 'Rotate angle: ', varangle
              elif event.button==3:
                 statusL=None
                 plt.close()
       def onmousemove(event):
           global fig, statusL,rotpoint,statusR,origpoint,\
                  refpoint,ax,ss,preharm,shiftdist,cmapp
           if statusR==None:
              if event.button==2:
                 origpoint=(event.xdata,event.ydata)
                 #cross=ax.plot(origpoint[0],origpoint[1],"r+",linewidth=5)
                 #fig.canvas.draw()
                 #cross[-1].remove()
                 print 'The center of SAXS is at ', np.around(origpoint)
                 statusR=1
           elif statusR==1:
                 if event.button==2:
                    refpoint=(event.xdata,event.ydata)
                    #cross=ax.plot(refpoint[0],refpoint[1],"r+",\
                    #                                   linewidth=5)
                    shiftdist=(np.around(refpoint[1]-origpoint[1]),\
                                np.around(refpoint[0]-origpoint[0]))
                    print 'Shift distance: ',shiftdist 
                    preharm['map']=interpolation.shift(preharm['map'],\
                                        shiftdist,prefilter=False,order=0)
                    #tmp=tolog(tolog(preharm))
                    if log is not None:
                       ax.imshow(tolog(preharm)['map'],\
                              interpolation='nearest',cmap=cmapp)
                    else:
                       ax.imshow(preharm['map'],\
                              interpolation='nearest',cmap=cmapp)
                    fig.canvas.draw()
                    #cross[-1].remove()
                    print 'The center of IMAGE is at ', np.around(refpoint)
                    varshiftdist0=shiftdist[0]
                    varshiftdist1=shiftdist[1]
                    ss[0]=ss[0]+varshiftdist0
                    ss[1]=ss[1]+varshiftdist1
                    print 'Shift distance: ',varshiftdist0,varshiftdist1
                    statusR=None
                 elif event.button==3:
                    statusR=None
                    plt.close()
       fig.canvas.mpl_connect('button_press_event',onmousemove)
       fig.canvas.mpl_connect('button_press_event',onmouserot)
       plt.show(block=True)
       return preharm,ss
    else:
       preharm['map']=interpolation.shift(preharm['map'],(cc[0],cc[1]),\
                                              order=0,prefilter=False)
       preharm['map']=interpolation.rotate(preharm['map'],cc[2],\
                                  reshape=False,prefilter=False,order=0)
       w,h=figure.figaspect(img['map'])
       fig=plt.figure(img['filename'],figsize=(1.2*w,1.2*h))
       ax=fig.add_axes([0,0,1,1])
       #load the color lookup table from  a pv-wave text file
       lut=sf_readclut('lut05.dat')
       clut=np.double(lut)/255
       cmapp=ListedColormap(zip(clut[0],clut[1],clut[2]),N=256)
       if cm is not None:
          cmapp=plt.get_cmap(name=cm.strip())
       if log is not None:
          ax.imshow(tolog(preharm)['map'],\
                          interpolation='nearest',cmap=cmapp)
       else:
          ax.imshow(preharm['map'],interpolation='nearest',cmap=cmapp)
       #ax.plot(img['center'][0],img['center'][1],"y+")
       ax.axhline(img['width']/2-0.5)
       ax.axvline(img['height']/2-0.5)
       circ0=plt.Circle((img['center'][0],img['center'][1]),radius=\
                         img['width']/4,fill=False,edgecolor='b')
       circ1=plt.Circle((img['center'][0],img['center'][1]),radius=\
                         img['width']/8,fill=False,edgecolor='b')
       ax.add_patch(circ0)
       ax.add_patch(circ1)
       plt.rcParams['toolbar']='None'
       plt.axis('off')
       statusL=None
       statusR=None
       
       def onmouserot(event):
           global fig, statusL,rotpoint,statusR,origpoint,\
                   refpoint,ax,ss,preharm,shiftdist,cmapp
           if statusL==None:
              if event.button==1:
                 rotpoint=[event.xdata,event.ydata]
                 rotpoint[0]=rotpoint[0]-preharm['center'][0]
                 rotpoint[1]=rotpoint[1]-preharm['center'][1]
                 rotangle=math.degrees(math.atan2(-rotpoint[1],\
                                                         rotpoint[0]))
                 preharm['map']=interpolation.rotate(preharm['map'],\
                            rotangle,reshape=False,prefilter=False,order=0)
                 #tmp=deepcopy(preharm)
                 if log is not None:
                    ax.imshow(tolog(preharm)['map'],\
                                  interpolation='nearest',cmap=cmapp)
                 else:
                    ax.imshow(preharm['map'],interpolation='nearest',\
                                         cmap=cmapp)
                 fig.canvas.draw()
                 varangle=rotangle
                 cc[2]=cc[2]+varangle
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
                              rotangle,reshape=False,prefilter=False,order=0)
                 #tmp=deepcopy(preharm)
                 if log is not None:
                    ax.imshow(tolog(preharm)['map'],\
                               interpolation='nearest',cmap=cmapp)
                 else:
                    ax.imshow(preharm['map'],\
                               interpolation='nearest',cmap=cmapp)
                 fig.canvas.draw()
                 varangle=rotangle
                 cc[2]=cc[2]+varangle
                 print 'Rotate angle: ', varangle
              elif event.button==3:
                 statusL=None
                 plt.close()
       def onmousemove(event):
           global fig, statusL,rotpoint,statusR,origpoint,\
                  refpoint,ax,ss,preharm,shiftdist,cmapp
           if statusR==None:
              if event.button==2:
                 origpoint=(event.xdata,event.ydata)
                 #cross=ax.plot(origpoint[0],origpoint[1],"r+",linewidth=5)
                 #fig.canvas.draw()
                 #cross[-1].remove()
                 print 'The center of SAXS is at ', np.around(origpoint)
                 statusR=1
           elif statusR==1:
                 if event.button==2:
                    refpoint=(event.xdata,event.ydata)
                    #cross=ax.plot(refpoint[0],refpoint[1],"r+",\
                    #                                   linewidth=5)
                    shiftdist=(np.around(refpoint[1]-origpoint[1]),\
                                np.around(refpoint[0]-origpoint[0]))
                    print 'Shift distance: ',shiftdist 
                    preharm['map']=interpolation.shift(preharm['map'],\
                                        shiftdist,order=0,prefilter=False)
                    #tmp=tolog(tolog(preharm))
                    if log is not None:
                       ax.imshow(tolog(preharm)['map'],\
                              interpolation='nearest',cmap=cmapp)
                    else:
                       ax.imshow(preharm['map'],\
                              interpolation='nearest',cmap=cmapp)
                    fig.canvas.draw()
                    #cross[-1].remove()
                    print 'The center of IMAGE is at ', np.around(refpoint)
                    varshiftdist0=shiftdist[0]
                    varshiftdist1=shiftdist[1]
                    cc[0]=cc[0]+varshiftdist0
                    cc[1]=cc[1]+varshiftdist1
                    print 'Shift distance: ',varshiftdist0,varshiftdist1
                    statusR=None
                 elif event.button==3:
                    statusR=None
                    plt.close()
       fig.canvas.mpl_connect('button_press_event',onmousemove)
       fig.canvas.mpl_connect('button_press_event',onmouserot)
       plt.show(block=True)
       return preharm,cc

