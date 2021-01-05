#!/usr/bin/env python
import matplotlib.pyplot as plt
#import matplotlib.figure as figure
import numpy as np
#from matplotlib.colors import ListedColormap
#from CDF_2D import tolog
#from sf_readclut import sf_readclut
def sf_circlenew():
    """
    Attention:this function can only be used in the same scale of x-y axis
    The x-y should be integer.
    """
    global fig, status,center,refpoint,radiuss,ax,axv,axh,ss
    """
    The following lines are used in sf_circle
    w,h=figure.figaspect(img['map'])
    fig=plt.figure(img['filename'],figsize=(1.2*w,1.2*h))
    ax=fig.add_axes([0,0,1,1])
    load the color lookup table from  a pv-wave text file
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
    center=None
    refpoint=None
    radiuss=None
    #cross=None
    ss=[0,0,0]
    #print ax
    def onbuttonpress(event):
        #print '***Draw circle***'
        #print event.button
        global status, center, refpoint, radiuss,fig,ax,axv,axh,ss
        #plt.clf()
        if status==None:
           if event.button == 1:
              #print 'button',event.button,' is working'
              center=(event.xdata,event.ydata)
              ss[0]=np.around(center[0])
              ss[1]=np.around(center[1])
              #print ss
              #cross=ax.plot(center[0],center[1],"r+",linewidth=5)
              axv=ax.axvline(center[0])
              axh=ax.axhline(center[1])
              #print ax
              fig.canvas.draw()
              #cross[-1].remove()
              axv.remove()
              axh.remove()
              print 'The center of circle is at ', np.around(center)
              status=1
        elif status==1 :
           if event.button==1 :
              #print status
              center=(event.xdata,event.ydata)
              ss[0]=np.around(center[0])
              ss[1]=np.around(center[1])
              #cross=ax.plot(center[0],center[1],"r+",linewidth=5)
              axv=ax.axvline(center[0])
              axh=ax.axhline(center[1])
              fig.canvas.draw()
              #cross[-1].remove()
              axv.remove()
              axh.remove()
              print 'The center of circle is at ', np.around(center)
           elif event.button==2 :
              refpoint=(event.xdata,event.ydata)
              print 'The refpoint is at ', np.around(refpoint)
              radiuss=np.around(np.sqrt((refpoint[0]-center[0])**2+\
                                   (refpoint[1]-center[1])**2))
              print 'The radius of the circle is ', radiuss
              ss[2]=np.around(radiuss)
              circ=plt.Circle(center,radius=radiuss,facecolor='None',\
                                edgecolor=(0,0,1))
              ax.add_patch(circ)
              axv=ax.axvline(center[0])
              axh=ax.axhline(center[1])
              #ax.add_line(cross[-1])
              fig.canvas.draw()
              circ.remove()
              axv.remove()
              axh.remove()
              #cross[-1].remove()
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

