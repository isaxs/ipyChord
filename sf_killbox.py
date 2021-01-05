#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib import patches
import numpy as np
from matplotlib.colors import ListedColormap
from CDF_2D import tolog
import matplotlib.cm as cm
from copy import deepcopy
from sf_readclut import sf_readclut
def sf_killbox(img,log=None,grey=None):
    """
    This function is used to kill the box on the image.
    """
    global fig, status,orig,refpoint,w,h,ax,axv,axh,ss
    w,h=figure.figaspect(img['map'])
    fig=plt.figure(img['filename'],figsize=(1.2*w,1.2*h))
    ax=fig.add_axes([0,0,1,1])
    #load the color lookup table from  a pv-wave text file
    lut=sf_readclut('lut05.dat')
    clut=np.double(lut)/255
    cmapp=ListedColormap(zip(clut[0],clut[1],clut[2]),N=256)
    if grey is not None:
       cmapp=cm.Greys_r
    if log is not None:
       ax.imshow(tolog(img)['map'],interpolation='nearest',cmap=cmapp)
    else:
       ax.imshow(img['map'],interpolation='nearest',cmap=cmapp)
    plt.rcParams['toolbar']='None'
    plt.axis('off')
    status=None
    orig=None
    refpoint=None
    w,h=None,None
    ss=[0,0,0,0]
    #print ax
    def onbuttonpress(event):
        #print '***Draw circle***'
        #print event.button
        global status, orig, refpoint, w, h, fig,ax,axv,axh,ss
        #plt.clf()
        if status==None:
           if event.button == 1:
              #print 'button',event.button,' is working'
              orig=(event.xdata,event.ydata)
              ss[0]=np.around(orig[0])
              ss[1]=np.around(orig[1])
              #print ss
              #cross=ax.plot(center[0],center[1],"r+",linewidth=5)
              axv=ax.axvline(orig[0])
              axh=ax.axhline(orig[1])
              #print ax
              fig.canvas.draw()
              #cross[-1].remove()
              axv.remove()
              axh.remove()
              print 'The origin of box is at ', np.around(orig)
              status=1
        elif status==1 :
           if event.button==1 :
              #print status
              orig=(event.xdata,event.ydata)
              ss[0]=np.around(orig[0])
              ss[1]=np.around(orig[1])
              #cross=ax.plot(center[0],center[1],"r+",linewidth=5)
              axv=ax.axvline(orig[0])
              axh=ax.axhline(orig[1])
              fig.canvas.draw()
              #cross[-1].remove()
              axv.remove()
              axh.remove()
              print 'The origin of box is at ', np.around(orig)
           elif event.button==2 :
              refpoint=(event.xdata,event.ydata)
              print 'The refpoint is at ', np.around(refpoint)
              w, h = np.around(refpoint[0]-orig[0]),\
                                  np.around(refpoint[1]-orig[1])
              if w != int(w/2)*2:
                 w=w+1
              if h != int(h/2)*2:
                 h=h+1
              print 'The width and height of box are  ',abs(w),' ',abs(h)
              ss[2], ss[3] = w, h
              box=plt.Rectangle((ss[0],ss[1]),w,h,facecolor='none',\
                                          edgecolor='yellow')
              ax.add_patch(box)
              #check the position of the first point
              #move it to the left upper corner
              if ss[2] < 0 or ss[3] < 0:
                 ss[0]=ss[0]+ss[2]
                 ss[1]=ss[1]+ss[3]
                 ss[2]=abs(ss[2])
                 ss[3]=abs(ss[3])
                 print 'The position of left upper corner is ',ss[0],ss[1]
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
    res=deepcopy(img)
    print ss
    ss[0],ss[1],ss[2],ss[3]=int(ss[0]),int(ss[1]),int(ss[2]),int(ss[3])
    res['map'][ss[1]:ss[1]+ss[3]+1,ss[0]:ss[0]+ss[2]+1]=0.0
    
    return res

#def sf_readclut(filename):
#    """
#    this file is used to load the color lookup table
#    it only canbe used for lut05.dat file
#    """
#    fid=open(filename,'r')
#    clt=np.loadtxt(fid,dtype=int)
#    fid.close()
#    return clt

