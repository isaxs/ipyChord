#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib import patches
import numpy as np
from matplotlib.colors import ListedColormap
from CDF_2D import tolog
from sf_readclut import sf_readclut
def sf_curvbox():
    global fig,status,orig,refpoint,w,h,ax,axv,axh,ss
    """
    Attention: ss[0,0,0,0]
    ss[0],ss[1] is the left-upper corner.
    ss[2],ss[3] is the width and height of rect.
    The following lines are used in sf_box 
    w,h=figure.figaspect(img['map'])
    fig=plt.figure(img['filename'],figsize=(1.4*w,1.4*h))
    ax=fig.add_axes([0,0,1,1])
    #load the color lookup table from  a pv-wave text file
    lut=sf_readclut('lut05.dat')
    clut=np.double(lut)/255
    cmapp=ListedColormap(zip(clut[0],clut[1],clut[2]),N=256)
    if log is not None:
       ax.imshow(tolog(img)['map'],interpolation='nearest',cmap=cmapp)
    else:
       ax.imshow(img['map'],interpolation='nearest',cmap=cmapp)
    plt.rcParams['toolbar']='None'
    plt.axis('off')
    """
    fig=plt.gcf()
    ax=plt.gca()
    status=None
    orig=None
    refpoint=None
    w,h=None,None
    ss=[0.0,0.0,0.0,0.0]
    #print ax
    def onbuttonpress(event):
        #print '***Draw circle***'
        #print event.button
        global status, orig, refpoint, w, h, fig,ax,axv,axh,ss
        #plt.clf()
        if status==None:
           if event.button == 1:
              #print 'button',event.button,' is working'
              ss[0],ss[1] = event.xdata,event.ydata
              #print ss
              #cross=ax.plot(center[0],center[1],"r+",linewidth=5)
              axv=ax.axvline(ss[0])
              axh=ax.axhline(ss[1])
              #print ax
              fig.canvas.draw()
              #cross[-1].remove()
              axv.remove()
              axh.remove()
              print 'The origin of box is at ',ss[0],ss[1]
              status=1
           elif event.button == 3:
              plt.close()
        elif status==1 :
           if event.button==1 :
              #print status
              ss[0],ss[1] = event.xdata,event.ydata
              #cross=ax.plot(center[0],center[1],"r+",linewidth=5)
              axv=ax.axvline(ss[0])
              axh=ax.axhline(ss[1])
              fig.canvas.draw()
              #cross[-1].remove()
              axv.remove()
              axh.remove()
              print 'The origin of box is at ',ss[0],ss[1]
           elif event.button==2 :
              refpoint=(event.xdata,event.ydata)
              print 'The refpoint is at ',refpoint
              w, h = (refpoint[0]-ss[0]),(refpoint[1]-ss[1])
              print 'The width and height of box are  ',w,' ',h
              ss[2], ss[3] = w, h
              box=plt.Rectangle((ss[0],ss[1]),w,h,facecolor='none',\
                                          edgecolor='yellow')
              ax.add_patch(box)
              #check the position of the first point
              #move it to the left upper corner
              if ss[2] < 0.0:
                 ss[0]=ss[0]+ss[2]
              if ss[3] > 0.0:
                 ss[1]=ss[1]+ss[3]
              ss[2]=abs(ss[2])
              ss[3]=abs(ss[3])
              print 'The position of left-upper corner is ',ss[0],ss[1]
              #print 'The position of left upper corner is ',ss[0],ss[1] 
              #
              fig.canvas.draw()
              box.remove()
              
           elif event.button==3 :
              status=None
              plt.close()
        #return ss   
    cid=fig.canvas.mpl_connect('button_press_event',onbuttonpress)
       
    plt.show(block=True)
    return ss

#def sf_readclut(filename):
#    """
#    this file is used to load the color lookup table
#    it only canbe used for lut05.dat file
#    """
#    fid=open(filename,'r')
#    clt=np.loadtxt(fid,dtype=int)
#    fid.close()
#    return clt

