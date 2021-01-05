#!/usr/bin/env python
import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import sys
import os
from sf_boxfree import sf_boxfree
from savitzky_golay import savitzky_golay
import scipy.fftpack as fftpack
import scipy.signal as signal
import scipy.interpolate as interpolate
from scipy.optimize import curve_fit
from misc import unit_rs,cutcur
#############################################################################
#idf()
#idf_cosine()
#porod_get()
#Ib_get()
#minus_fl()
#forward()
#fil()
#sub_fl()
#stretch()
#sg_curv()
#dct1d()
#curread()
#curwrite()
#curplot()
#lowpass1d()
#hanning()
#mirror()
#############################################################################
def idf(c,r=20,timz=8,win='Fig 1'):
    """
    This function applies the Fourier trnasform to the curv got from preidf().
    c--curve got from preidf(),the interference function.
    r--specifying the range of r displayed on the x-axis default,50nm
    """
    
    #use the Fourier transform to get the IDF
    g1rtmp=dct1d(c,timzero=timz)
    pos=np.amax(np.where(g1rtmp[:,0] <= r))
    g1=np.zeros((pos,2),dtype=np.float)
    g1[:,1]=g1rtmp[0:pos,1]*(-1.0)
    g1[:,0]=g1rtmp[0:pos,0]
    curplot(g1,label='IDF',win=win)
    #print g1.shape,g1
    return g1
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def idf_cosine(Iintf,r=30,N=256,win='Fig 1'):
    """
    This func is used to calculate the IDF by cosine transformation.
    """
    #dx and Iy
    sx,Iy=Iintf[:,0],Iintf[:,1]
    #unit of sx
    sunit=unit_rs(sx)
    #construct r and g1
    r1=np.linspace(0.,r,N)
    g1=deepcopy(r1)
    #
    for i in range(N):
        g1i=np.trapz(np.cos(2*np.pi*sx*r1[i])*Iy,dx=sunit)*(-1.0)
        g1[i]=g1i
    #construct an two-col array
    g1r=np.zeros((N,2),dtype=np.float)
    g1r[:,0],g1r[:,1]=r1,g1*8*np.pi**2
    #plot the resulted idf curve
    curplot(g1r,label='IDF',win=win)
    return g1r
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def porod_get(curv):
    """
    This function is used to compute the pre-function (interference function )
    the interface distribution function.
    curv--return the interference function.
    """
    global fig,Iands,ax,factor,fl
    
    #don't ruin the original data
    dupcurv=deepcopy(curv)
    factor=0 
    fl=0
    #here, we set the curve with different plots in order to get the
    #fluctuation background.
    #how can we assess the plot is good? It depends on the intergral of 
    #interference function is zero.
    #in the current procedure we need to determine the power of s
    print('It only supports the NO. 0 1 2 3 4 and e typed from the keyboard')
    print('NO. 0: I(s)*s^0 vs. s plot')
    print('NO. 1: I(s)*s^1 vs. s plot')
    print('NO. 2: I(s)*s^2 vs. s plot')
    print('NO. 3: I(s)*s^3 vs. s plot')
    print('NO. 4: I(s)*s^4 vs. s plot')
    
    def press(event):
        global fig,Iands,ax,factor,fl
        #print('press', event.key)
        #sys.stdout.flush() print the keyname immediately
        sys.stdout.flush()
        if event.key == '0':
           print('press', event.key)
           #print type(event.key)
           factor=int(event.key)
           print('plot the original map')
           Iands.set_ydata(curv[:,1])
           fig.canvas.set_window_title('Original Plot I(s) vs. s')
           ax.relim()
           ax.autoscale_view()
           fig.canvas.draw()
        elif event.key == '1':
           print('press', event.key)
           factor=int(event.key)
           print('plot the I(s)*s^1 vs. s')
           Iands.set_ydata((4*np.pi**2)*curv[:,1]*curv[:,0])
           fig.canvas.set_window_title('Plot  I(s)*s^1 vs. s')
           #plt.axhline(y=0,color='r',linewidth=4)
           ax.relim()
           ax.autoscale_view()
           fig.canvas.draw()
        elif event.key == '2':
           print('press', event.key)
           factor=int(event.key)
           print('plot the I(s)*s^2 vs. s')
           Iands.set_ydata((4*np.pi**2)*curv[:,1]*curv[:,0]**2)
           fig.canvas.set_window_title('Plot  I(s)*s^2 vs. s')
           ax.relim()
           ax.autoscale_view()
           fig.canvas.draw()
        elif event.key == '3':
           print('press', event.key)
           factor=int(event.key)
           print('plot the I(s)*s^3 vs. s')
           Iands.set_ydata((4*np.pi**2)*curv[:,1]*curv[:,0]**3)
           fig.canvas.set_window_title('Plot  I(s)*s^3 vs. s')
           ax.relim()
           ax.autoscale_view()
           fig.canvas.draw()
        elif event.key == '4':
           print('press', event.key)
           factor=int(event.key)
           print('plot the I(s)*s^4 vs. s')
           Iands.set_ydata((4*np.pi**2)*curv[:,1]*curv[:,0]**4)
           fig.canvas.set_window_title('Plot  I(s)*s^4 vs. s')
           ax.relim()
           ax.autoscale_view()
           fig.canvas.draw()
        elif event.key == 'shift':
           print('press',event.key)
           print('Now switch to the fluctuation determination')
           fig.canvas.set_window_title('Fluctuation determination')
           daty=curv[:,1]
           while True:
              fltest=input('Input the fl:')
              if fltest == 'ok':
                 plt.close('all')
                 break
              datyfl=(4*np.pi**2)*(daty-float(fltest))*curv[:,0]**factor
              fl=float(fltest)
              #print(fl)
              Iands.set_ydata(datyfl)
              #evaluate the integral of the curve.
              #print 'The integral of the curve : ',np.sum(datyfl)
              plt.axhline(y=0,color='r',linewidth=1)
              ax.relim()
              ax.autoscale_view()
              plt.pause(0.001)
              fig.canvas.draw()
              print('I am fine!')
        elif event.key == 'e':
           print('press',event.key)
           print('exit from the graph')
           plt.close('all')
        else:
           print('press',event.key)
           print('This keys does not work')
    #fig=plt.gcf()
    #ax=fig.add_subplot(111)
    fig,ax=plt.subplots()
    fig.canvas.mpl_connect('key_press_event',press)
    Iands,=ax.plot(curv[:,0],curv[:,1])
    plt.show(block=True)
    return factor,fl
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Ib_get(Is,n=4):
    """
    This func is used to get the Ib where Ib=b0*s^2
    The fluctuation scattering intensity will be subtracted.
    Here we can change the b to get the perfect b0
    """
    #duplicate the Is
    Ip=deepcopy(Is)
    #
    sx=Is[:,0]
    b0=0.
    Ibfl=b0*sx**2
    #plot the curve
    Ip[:,1]=Is[:,1]*sx**4
    curplot(Ip,label='I_orig')
    #get the figure handle
    fig=plt.gcf()
    ax=fig.add_subplot(111)
    Iands,=ax.plot(sx,Ibfl,label='I_b')
    #
    while True:
          btest=input('Input the b:')
          if btest == 'ok':
             plt.close('all')
             break
          Ibfl=float(btest)*sx**(2+n)
          b0=float(btest)
          Iands.set_ydata(Ibfl)
          #
          #plt.axhline(y=0,color='r')
          ax.relim()
          ax.autoscale_view()
          fig.canvas.draw()
    #block the terminal
    plt.show(block=True)
    Ip[:,1]=(Is[:,1]-b0*sx**2)*sx**n
    return Ip
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def minus_fl(vp):
    """
    This function is used to subtract the fluctuation scattering. This fl is 
    determined by the formula I_Fl(s)=I_Fl(0)+bs^2.
    """
    #don't ruin the priginal data
    vp_pow=deepcopy(vp)
    s=vp[:,0]
    Ivp=vp[:,1]
    #show the curve
    curplot(vp,win='Power determination')
    fig=plt.gcf()
    while True:
          powe=raw_input('Input the trial power: ')
          if powe == 'ok':
                break
          power=int(powe)
          datay=Ivp*s**power
          ax=fig.gca()
          ax.lines[0].set_ydata(datay)
          ax.relim()
          ax.autoscale_view()
          fig.canvas.draw()
    #fl determination
    print('Fluctuation determination')
    fig.canvas.set_window_title('Fluctuation Determination')
    while True:
       fl0=raw_input('Input the fl: ')
       if fl0 == 'ok':
          break
       fl=float(fl0)
       datay=(Ivp-fl)*s**power*4*np.pi**2
       ax.lines[0].set_ydata(datay)
       ax.relim()
       ax.autoscale_view()
       fig.canvas.draw()
    print('The power is : ',power)
    print('The fl is : ',fl)
    return power,fl
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def forward(curv,factor=0,fl=0):
    """
    This function is used to subtract the fl from I(s) and to multiply
    the factor to I(s) by s^factor.
    """
    print('****fl is the Porod constant got by I(s)*s^n vs. s plot****')
    resc=deepcopy(curv)
    print('The fluctuation is :', fl)
    resc[:,1]=curv[:,1]-fl
    if factor == 0:
       print('We do not do anything!')
    res=deepcopy(resc)
    res[:,1]=(4*np.pi**2)*resc[:,1]*resc[:,0]**factor
    print('Multiply by the factor: I(s)*s^factor ',factor)
    return res
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fil(cu,cutoff=1,order=1,iterate=None):
    """
    This function is Butterworth low-pass used to remove the highpass noise.
    #iterate must be a positive integer.
    The cutoff range [1.0,20]. The default cutoff is 3. Stribeck often used 1.0.
    """
    resc=comb(cu)
    #apply the lowpass filter once
    reslow=lowpass1d(resc,cutoff=cutoff,order=order)
    resc[:,1]=(resc[:,1]-reslow[:,1])*hanning(len(resc[:,1]))
    integ0=abs(resc[:,1].sum())
    relinteg=integ0/integ0
    print('Iterating......0')
    print('The sum of interference function 0 is ',integ0)
    intfer='intfer0'
    curplot(resc,label=intfer)
    #smooth the curve with butterworth filter of low pass
    if iterate is not None and iterate >= 1:
        count=int(abs(iterate))
        for i in range(count):
            #remove the noise.
            reslow=lowpass1d(resc,cutoff=cutoff,order=order)
            resc[:,1]=(resc[:,1]-reslow[:,1])*hanning(len(resc[:,1]))
            integ=abs(resc[:,1].sum())
            relinteg=integ/integ0
            print('Iterating..........'+str(i+1)+' until the end times')
            print('The sum of interference function is '+str(i+1),integ)
            print('The relative sum of interference func is '+str(i),relinteg)
    elif iterate is not None and iterate < 1.0:
        #User has requested to iterate until the integral of the interferene
        #function becomes almost zero
        iterate=abs(iterate)
        j=0
        while relinteg > iterate:
            #continue the interation
            reslow=lowpass1d(resc,cutoff=cutoff,order=order)
            resc[:,1]=(resc[:,1]-reslow[:,1])*hanning(len(resc[:,1]))
            integ=abs(resc[:,1].sum())
            relinteg=integ/abs(integ0)
            print('Iterating..........'+str(j+1)+' until it is satisfied')
            j=j+1
            print('The sum of interference function is ',integ)
            print('The relative valueof sum of interference func is ',relinteg)
    else:
        print('No more background correction iteration!')

    curplot(cu,label='orig_intf')
    curplot(reslow,label='Ap')
    curplot(resc,label='rest_intf')
    return resc
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sub_fl(cu,fl):
    """
    This func is to subtract the fl.
    """
    cur=deepcopy(cu)
    cur[:,1]=cur[:,1]-fl
    return cur
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def stretch(cu,Ap=0,order=3,beamstop=None,intf=None,power=None):
    """
    This function is used to remove the bad points around the beamstop.
    It also extrapolate the data to beam stop and I(0)=0
    """
    curv=deepcopy(cu)
    if beamstop is not None:
       print('Remove the bad points around the beam stop')
       curplot(curv)
       #get the region what we want
       ss=sf_boxfree()
       #print ss
       #extrapolate the curve into the zero point (0,0)
       goodmax=ss[0]+ss[2]
       goodmin=ss[0]
       #print s range
       pos=np.where((curv[:,0] < goodmax) & (curv[:,0] > goodmin))
       #print pos
       sfit=curv[:,0][pos]
       sunit=unit_rs(sfit)
       Ifit=curv[:,1][pos]
       smmax=curv[max(pos[0]),0]
       Ns=smmax/sunit
       snew=np.linspace(0.0,smmax,Ns)
       #the I(s=0)*s**n=0
       sfit[0],Ifit[0]=0.0,0.0
       if power is not None:
           #construct an power func to fit the curve
           def powfunc(x,a,b,c):
               return a*np.exp(b*x)+c
           popt,pcov=curve_fit(powfunc,sfit,Ifit)
           a,b,c=popt[0],popt[1],popt[2]
           print(a,b,c)
           newI=powfunc(snew,a,b,c)
       else:
           z=np.polyfit(sfit,Ifit,order)
           #print type(pos)
           p=np.poly1d(z)
           #print snew
           newI=np.polyval(p,snew)
       curtmp=np.stack((snew,newI),axis=-1)
       curv_snd=curv[max(pos[0]):,:]
       curv=np.vstack((curtmp,curv_snd))
       #curv[0,0]=0.0
       #curv[0,1]=0.0
    curplot(curv)
    print('Extrapolate the Porod region flat')
    cc=sf_boxfree()
    sPorodmin=cc[0]
    posPorod=np.where(curv[:,0] > sPorodmin)
    curv[:,1][posPorod]=abs(Ap)
    #extend the Porod region
    #get the pixel size of s
    tmpc=deepcopy(curv)
    oldlen=len(curv[:,0])
    tmpn=tmpc[1:oldlen-2,0]
    tmps=tmpn-tmpc[0:oldlen-3,0]
    sunittmp=np.average(tmps)
    #put the s in the end of s
    sendnum=int(cc[2]/sunittmp)*1.2
    send=np.arange(1,sendnum,1)*sunittmp+curv[-1,0]
    Iend=np.zeros(len(send))
    Iend[:]=abs(Ap)
    curvx=np.append(curv[:,0],send)
    curvy=np.append(curv[:,1],Iend)
    curvnew=np.zeros((len(curvx),2),dtype=np.float)
    curvnew[:,0]=curvx
    curvnew[:,1]=curvy
    curplot(curvnew)
    #get the intference function by subtracting the Ap
    if intf is not None:
       #curn=deepcopy(curvnew)
       curvnew[:,1]=curvnew[:,1]-Ap
    return curvnew
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sg_curv(curv,winsiz=21,order=3):
    """
    This function is used to smooth the curve with savitzky_golay filter
    """
    resc=deepcopy(curv)
    resc[:,1]=savitzky_golay(resc[:,1],window_size=winsiz,order=order)
    curplot(curv,label='orig')
    curplot(resc,label='smooth')
    return resc
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def dct1d(curv,timzero=8):
    """
    This function is used to do the Discrete Cosine Transforms.
    """
    tmpc=deepcopy(curv)
    #get the function of interpolation for the curve
    #f = interpolate.interp1d(curv[:,0],curv[:,1],kind='linear')
    #zoom and interpolate more points in the curve
    #get the step
    length=len(curv[:,0])
    #print length
    tmpn=tmpc[1:length-2,0]
    tmps=tmpn-tmpc[0:length-3,0]
    sunit0= np.average(tmps)
    #interpolate and zoom the curve
    #sunit=sunit0/times
    #snew=np.arange(curv[:,0][0]-sunit,curv[:,0][-1]-sunit0,sunit)+sunit
    #Inew = f(snew)
    #append mutiple zero after the I(s) times by timzero
    #print tmpc[:,1].shape
    zeroarr=np.zeros(((timzero-1)*length,1),dtype=np.float)
    #replace the zeros by last number of the curve.
    Inew=np.append(tmpc[:,1],zeroarr)
    #print zeroarr.shape
    g1r=fftpack.dct(Inew,norm='ortho')
    resc=np.zeros((len(g1r),2),dtype=np.float)
    #compute the runit
    runit=0.5/(len(Inew)*sunit0) #normal fft 1, 0.5 for dct (half wing of fft) 
    r=np.arange(len(g1r))*runit
    resc[:,0]=r
    resc[:,1]=g1r
    return resc
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def curread(cfilename,q2s=None,cut=None,Lc=None):
    """
    This file is used to read the stored curve from the hard disk.
    """
    tmp=np.loadtxt(cfilename,dtype='float',unpack=True)
    curve=np.zeros((len(tmp[0]),2),dtype=np.float)
    if q2s is not None:
        curve[:,0]=tmp[0]/(2*math.pi)
    else:
        curve[:,0]=tmp[0]
    curve[:,1]=tmp[1]
    if cut is not None:
        curve=cutcur(curve,cut)
    if Lc is not None:
        #do the Lorentz correction
        curve[:,1]=4*np.pi*curve[:,0]**2*curve[:,1]
    return curve
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def curwrite(curve,cfilename,comm='#'):
    """
    This function is used to write the curve into hard disk.
    """
    np.savetxt(cfilename,curve,header=comm)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def curplot(curve,block=False,hor=0.0,win='Fig. 1',label='abc',line='-',\
             logy=None):
    """
    This function is used to a simple curve
    """
    fig=plt.figure(win)
    ax=fig.add_subplot(111)
    x=curve[:,0]
    if logy is not None:
       y=np.log10(curve[:,1]+1.0)
    else:
       y=curve[:,1]
    Iands,=ax.plot(x,y,label=label,linestyle=line)
    ax.legend(loc='upper right',shadow=True)
    #plt.axhline(y=hor,color='r',linestyle='-')
    #close the window by mouse right click
    #set the mouse right click to kill the window.
    def onmouse(event):
        if event.button==3:
           plt.close(win)#img['filename'])
        elif event.button==2:
              coordx,coordy=event.xdata,event.ydata
              print(coordx,coordy)
    #connect the mouse action with plt
    fig.canvas.mpl_connect('button_press_event',onmouse)
    if block == False:
       plt.show(block=False)
    else:
       plt.show(block=True)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def lowpass1d(curv, cutoff=0.1, order=1,Npts=256):
    """
    This function is used to extract the low pass component of sginal.
    When we want to smooth the curve, we can use c=lowpass1d(curve,10,2) to
    get a smoothed curve.
    """
    c=deepcopy(curv)
    #make the curve a comb function with equi-distance x
    c=comb(c,Npts=Npts)
    Iss=c[:,1]
    order = 2**int(order)
    lenn=len(Iss)
    r = np.arange(lenn)
    #low pass
    factor=1.0*lenn/Iss.max()
    Iss=Iss*factor
    miIss=mirror(Iss)
    mir =mirror(r)
    ######tmp######
    #plot the curve and its mirror.
    #c_tmp=np.zeros((len(mir),2),dtype=np.float)
    #c_tmp[:,0]=mirror(c[:,0],neg=1)
    #c_tmp[:,1]=miIss
    #curplot(c_tmp,label='mi')
    ######tmp#####
    ######freq####
    freq=cutoff*lenn/256.0
    ##############
    lowy=fftpack.ifft(fftpack.ifftshift(fftpack.fftshift(fftpack.fft(miIss))\
            *(1.0/(1.0+(mir/freq)**order))))
    cy=np.real(lowy)/factor
    lenlowy=len(lowy)
    #print(type(lenlowy))
    c[:,1]=cy[int((lenlowy/2))::]
    #curplot(c,label='Ap')
    ######tmp######
    #plot the lowpass
    #low_tmp=deepcopy(c_tmp)
    #low_tmp[:,1]=cy
    #curplot(low_tmp,label='low')
    ######tmp#######
    return c
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def comb(curv,Npts=256):
    """
    This func is used to make the curve a comb function with specified Npts
    points, starting from 0.0 and extending to its present end at equidistantly
    spaced x-values.
    """
    #get the x and y
    origx,origy=curv[:,0],curv[:,1]
    xmax=origx[-1]
    #construct the new x
    nx=np.linspace(0.0,xmax,Npts)
    ny=np.interp(nx,origx,origy)
    resc=np.stack((nx,ny),axis=-1)
    return resc
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hanning(Npts):
    """
    This function is used to generate a zero-phase hanning function.
    H(n)=0.5*(1.0+cos(pi*(n-1)/(L-1)))
    The above Hanning window function is used 2020.08.08
    The 
    """
    #H=0.5+0.5*np.cos((np.pi*np.arange(length))/(length-1)) #dumped
    pilin=np.linspace(0.0,np.pi,Npts)
    H=0.5*(1.0+np.cos(pilin))
    return H
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def mirror(arr,neg=0):
    """
    This function is used to mirror the arr in order to get the mirror plus
    image.
    """
    rearr=deepcopy(arr)
    #neg is False
    if neg == 0:
       res=np.append(np.flipud(rearr),arr)
    else:
       res=np.append(-1.0*np.flipud(rearr),arr)
    return res
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def deterFl(Is,I_fl0,b):
    """
    This function determines the fluctuation scattering from the existing
    scattering intensity.
    """
    #extract the scattering vector
    s=Is[:,0]
    I_fl=I_fl0+b*s**2
    
    return I_fl
