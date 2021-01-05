#!/usr/bin/env python
#this function is used to show the image array in the 
#matplotlib
#***************************************************************************
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.figure as figure
from copy import deepcopy
from sf_readclut import sf_readclut
import scipy.ndimage.interpolation as interpolation
import matplotlib as mpl
import math
import time
from matplotlib.colors import LightSource
#****************************************************************************
def sf_show(img,ulev=None,dlev=None,log=None,neg=None,svg=None,cm=None,\
             win=None,abso=None,wbg=None,block=True,auto=None,lut=256,
              clip=True,noshow=False,closewin=True,cross=None,circle=0,\
               title=None,fmt='svg',dpi=100,shade=None,lsaz=270,lsalt=135,\
                slptim=1,zoom=1.2):
    """
    sf_show() used to show the image under different conditions
    Attention: this function can not show the positive and negative value
               in one image except we use the absolute value to show all the
               values in one image.
    ulev    : show the image in certain interval,ulev defines the upper level.
    dlev    : dlev defines the down level
    log     : show the image in logscale coordinate
              if log==0 show it in linear coordinate
              if log==1 show it in log() coordinate
              if log==2 show it in log(log()) coordinate
              if log==9 show it in log() coordinate
    neg     : show the negative side of the image
    svg     : export the svg format of the image
              as the volume of svg format is very small and compatible
              with the inkscape which can be used to orgnize the image
    cm      : show the image in the different colormap
              refer to:
              http://matplotlib.org/examples/color/colormaps_reference.html
              cm is a string name of colormap in the matplotlib like in matlab
    cmaps = [('Sequential', ['binary', 'Blues', 'BuGn', 'BuPu', 'gist_yarg',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
             ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool', 'copper',
                                'gist_gray', 'gist_heat', 'gray', 'hot','pink',
                                'spring', 'summer', 'winter']),
             ('Diverging', ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                            'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'seismic']),
             ('Qualitative', ['Accent', 'Dark2', 'hsv', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3', 'spectral']),
             ('Miscellaneous', ['gist_earth', 'gist_ncar', 'gist_rainbow',
                               'gist_stern','jet','brg','CMRmap','cubehelix',
                               'gnuplot', 'gnuplot2', 'ocean', 'rainbow',
                               'terrain', 'flag', 'prism'])] 
    win     : show the image in the specified window
    wbg : show the bg color of image in white color
    block   : show the image in the interactive way and block the
              command line
    lut:  If lut is not None it must be an integer giving the number of 
          entries desired in the lookup table.
          The image is shown in 256 colors in default.
    """
    #protect the virgin img 
    imgi=deepcopy(img)
    #if we show the absolute value
    if abso is not None:
       imgi['map']=np.absolute(imgi['map'])
    if neg is not None:
       mask = imgi['map'] <= 0.0
       imgi['map'] = imgi['map'] * mask
       imgi['map'] = -1.0*imgi['map']
       print('The maximal value of map is '+str(imgi['map'].max()))
    #
    #clip negative values as imshow only support positive values
    if clip == True:
       imgi['map'] = imgi['map'] * (imgi['map'] >= 0.0 )
    else:
       print('White bg is set to display zero-level')
       if cm is None:
          print('A colormap must be supplied')
       posiheight=np.amax(imgi['map'])
       negheight=np.amax(-1.0*imgi['map'])
       #get the ratio of posheight and negheight
       if log == 9:
          zlevel=math.log(abs(posiheight)+1.0)/(math.log(abs(posiheight)+1.0)+\
                             math.log(abs(negheight)+1.0))
       else:
          zlevel=abs(posiheight)/(abs(posiheight)+abs(negheight))
       zeroind=int(256*(1.0-zlevel))
       #introduce the colormap
       clrmap=plt.get_cmap(name=cm.strip(),lut=256)
       clrmapvals=clrmap(np.arange(256))
       #set the zero-level value as white
       for i in range(zeroind-5,zeroind+5):
           clrmapvals[i,:]=[1.0,1.0,1.0,1.0]
       #convert it to colormap
       cmaps=mpl.colors.LinearSegmentedColormap.from_list('newcm',clrmapvals)
       #show that in log 
       if log == 9:
          tmpposi= imgi['map'] * (imgi['map'] >= 0.0 )
          tmpneg=  imgi['map'] * (imgi['map'] <= 0.0 )
          tmpposilog = np.log( tmpposi+1.0 )
          tmpneglog  = -1.0* np.log( np.absolute(tmpneg)+1.0)
          imgi['map'] = tmpposilog + tmpneglog          
    #set the upper level of display range
    if ulev is None:
       ulev = imgi['map'].max()
    #set the down level of display range
    if dlev is None:
       dlev = imgi['map'].min()
    ###
    if log is None:
       pass
       #print 'The image is shown in the linear coordinate'
    elif log == 1:
       ulev = np.log(ulev+1.0)
       dlev = np.log(dlev+1.0)
       imgi['map'] = np.log( imgi['map']+1.0 )
    elif log == 2:
       ulev = np.log(np.log(ulev+1.0)+1.0)
       dlev = np.log(np.log(dlev+1.0)+1.0)
       imgi['map'] = np.log(np.log( imgi['map']+1.0 )+1.0)
    elif log != 9:
       print('log= 1 or 2 is used to show the image in logrithmic coordinate')
       print('The current image is shown in linear coordinate.')
    if cm is None:
       #load the color lookup table from  a text file
       #this is a default colormap, it shows the image very clearly.
       lut=sf_readclut('lut05.dat')
       clut=np.double(lut)/255 #inside the [0,1]
       #for the default colormap, we can set a white background.
       if wbg is not None:
          clut[0,0]=1.0
          clut[0,1]=1.0
          clut[0,2]=1.0
       #cmaps=ListedColormap(zip(clut[0],clut[1],clut[2]),N=256)
       cmaps=LinearSegmentedColormap.from_list('myc',clut)
    elif clip==True and cm is not None:
       cmaps=plt.get_cmap(name=cm,lut=lut)
    if not noshow:
       #create the object of show
       w,h=figure.figaspect(imgi['map'])
       if win is None:
          fig=plt.figure(imgi['filename'],figsize=(zoom*w,zoom*h))
       else:
          fig=plt.figure(win,figsize=(zoom*w,zoom*h)) 
       #get the current figure
       #ax=plt.gca()
       ax=plt.axes([0,0,1,1])
       #ax=fig.add_axes([0,0,1,1],frameon=False)
       #show  the image
       if shade is None:
           #
           pat=ax.imshow(imgi['map'],interpolation='bilinear',cmap=cmaps)
           if cross is not None:
              ax.axhline(img['center'][1]-0.5)
              ax.axvline(img['center'][0]-0.5)
           if circle != 0:
              circ=plt.Circle(img['center'],radius=circle,facecolor='None',\
                               edgecolor=(1,1,1))
              ax.add_patch(circ)
       #shaded 
       else:
          ls=LightSource(azdeg=lsaz,altdeg=lsalt)
          imgi['map']=ls.shade(imgi['map'],cmap=cmaps)
          #
          pat=ax.imshow(imgi['map'],interpolation='bilinear',cmap=cmaps)
       #set_clim is used to set the display-range of data
       #set_clim(vmin,vmax) to display the data in the interval [vmin,vmax]
       pat.set_clim(dlev,ulev)
       #set the window title
       #plt.title(img['filename'])
       if title is None:
          fig.canvas.set_window_title(imgi['filename'])
       else:
          fig.canvas.set_window_title(title)
       #maybe show a cross in the center of image
       #
       #turn interactive mode on
       plt.ion()
       #
       #ax.set_axis_off()
       plt.axis('off')
       #hide the axes
       #ax.set_axis_off()
       pat.axes.get_xaxis().set_visible(False)
       pat.axes.get_yaxis().set_visible(False)
       #hide the toolbar
       plt.rcParams['toolbar'] = 'None'
       #plt.tight_layout()
       #plt.colorbar()
       ax.axis('tight')
       #it makes the plot appear
       if type(svg)==int:
          #plt.gcf().set_size_inches(3,3)
          plt.savefig(imgi['filename'].split('.')[0]+'.'+fmt,format=fmt,\
                         transparent=True,dpi=dpi)
       elif type(svg)==str:
          plt.savefig(svg+'.'+fmt,format=fmt,transparent=True)
       else:
          pass
       #mouse action
       def onmouse(event):
           if event.button==3:
              plt.close(plt.gcf())#img['filename'])
           elif event.button==2:
              coordx,coordy=int(event.xdata),int(event.ydata)
              zval=img['map'][coordy,coordx]
              print(coordx-img['center'][0]-0.5,coordy-\
                      img['center'][1]-0.5,zval,' in pixel coord')
              print((coordx-img['center'][0]-0.5)*img['boxlen'][0],\
                   (img['center'][1]+0.5-coordy)*img['boxlen'][1],\
                                           ' in r/s coord')
       #connect the mouse action with plt
       fig.canvas.mpl_connect('button_press_event',onmouse)
       ax.autoscale(enable=True)
       ax.autoscale_view(scalex=True,scaley=True)
       #show the canvas
       if auto is None:
          if block == True:
             plt.show(block=True)
          else:
             plt.show(block=False)
       else:
          plt.draw()
          #release the memory
          if closewin==True:
             time.sleep(slptim)
             plt.close('all')
       #return the ax for the next call
       return ax
    else:
       lev=[dlev,ulev]
       return imgi,cmaps,lev

#############################################################################
def sf_show_points(ax,nparr):
    """
    This func is used to plot the numpy.ndarray into the current figure.
    Here, the arr is the coordinates of the local maximum peaks.
    This func only works with sf_show() in this module.
    """
    #get the axis handle of current figure
    ax.plot(nparr[:,1],nparr[:,0],'r.')
    plt.show()
