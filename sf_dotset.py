#!/usr/bin/env python
#This module is to collect a series of the coordinates of dots slected with
#mouse.
#****************************************************************************
import matplotlib.pyplot as plt
import numpy as np
#****************************************************************************
def sf_dotset():
    """
    This func is to select a series of dots with mouse. It is selected from
    the shown curve.
    """
    global fig,ax,ss,statusL,statusM,axh,axv
    #get the handle of figure and axis
    fig=plt.gcf()
    ax=plt.gca()
    statusL,statusM=None,None
    ss=[]
    print 'Drawing a line, you should select at least 2 points'
    #
    def onmouse(event):
        global fig,ax,ss,statusL,statusM,axh,axv
        if statusL==None:
           if event.button==1:
              x,y=event.xdata,event.ydata
              print 'Coordinate of current point', x, y
              ss.append([x,y])
              axv=ax.axvline(event.xdata)
              axh=ax.axhline(event.ydata)
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
              #polygondraw=plt.Polygon(ss,fill=None,edgecolor='b')
              #convert ss to to two-col array
              sxy=np.asarray(ss)
              sx,sy=sxy[:,0],sxy[:,1]
              plt.plot(sx,sy)
              #ax.add_patch(polygondraw)
              #show the patch
              fig.canvas.draw()
              print 'Press the middle button to remove the bad points'
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
              print 'Press the left button to select a new point'
           elif event.button==3:
              plt.close()

    cid=fig.canvas.mpl_connect('button_press_event',onmouse)
    plt.show(block=True)
    #remove the polygon area
    #tmp=np.ones((res['height'],res['width']))
    line=np.asarray(ss,dtype=np.float)
    return line
