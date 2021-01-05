#!/usr/bin/env/ python 
#this function is used to fit the G(s) or g(r) to get the distribution of
#domain or the long-spacing.
#the relative papers:
#N.Stribeck Colloid Polym Sci 271:1007-1023
#N.Stribeck Journal De Physque IV Vol.3 Page507-510
#N.Stribeck Macromol Chem Phys 2004,205,1455-1462
#N.Stribeck J Appl Cryst 2006,39,237-243
#N.Stribeck J Poly. Sci. Part B: Poly. Phys. 1999, 37, 975-981
#############################################################################
import numpy as np
from lmfit import Model,Parameters,minimize,report_fit,fit_report
import scipy.fftpack as fftpack
import math
from copy import deepcopy
import scipy.integrate as integrate
from fft1d import idf1d,intfer1d
from misc import punch_cur,lmfit_plot1d,cplot,clip_neg0,lmfit_plot1dob
from misc import cutcur,curv_zoom
from savitzky_golay import savitzky_golay
stksolo_rng=[3,25]
#############################################################################
#atf_g1r()
#fit_g1rsolo()
#minfit_g1rsolo()
#minfit_g1rduosolo()
#fit_g1rstack()
#minfit_g1rstack()
#fit_g1r2stacks()
#minfit_g1r2tacks()
#minfit_g1r2staksolo()
#fit_g1rstacksolo()
#fit_g1rstacksolo_rng()
#fit_g1rstacksolo_indep()
#minfit_g1rstacksolo()
#fit_g1r123stacks()
#fitwrite()
#minfitwrite()
#decomp_2infstacks()
#decompmin_2infstacks()
#decompmin_2infstacksolo()
#decomp_infsolo()
#decomp_infsolo_indep()
#g1r_stack()
#g1r_2stacks()
#G1s_2stacks()
#G1s_stack()
#G1s_trio()
#G1s_duo()
#G1s_uno()
#funcadd()
#GAF_sup_s()
#ATs()
#BTs()
#############################################################################
def atf_g1r(g1r,plot=0):
    """
    This func auto-fits the computed IDF with specified func.
    It does not work quite well now.
    """
    #count the loop cycles
    count=0
    #the minfit_g1r2stacksolo() is set as an example
    res,cfit=minfit_g1r2stacksolo(g1r,1,2,3,4,5,6,7,8,plot=plot)
    while True:
        tmp_chisqr=res.chisqr
        W1st=res.params['g1r_W1st'].value
        D11=res.params['g1r_D11'].value
        D12=res.params['g1r_D12'].value
        sig11=res.params['g1r_sig11'].value
        sig12=res.params['g1r_sig12'].value
        sigH1=res.params['g1r_sigH1'].value
        W2nd=res.params['g1r_W2nd'].value
        D21=res.params['g1r_D21'].value
        D22=res.params['g1r_D22'].value
        sig21=res.params['g1r_sig21'].value
        sig22=res.params['g1r_sig22'].value
        sigH2=res.params['g1r_sigH2'].value
        Wuno=res.params['g1r_Wuno'].value
        D0=res.params['g1r_D0'].value
        sigH0=res.params['g1r_sigH0'].value
        sig0=res.params['g1r_sig0'].value
        res,cfit=minfit_g1r2stacksolo(g1r,W1st,D11,D12,W2nd,D21,D22,Wuno,D0,\
                  sigH1=sigH1,sig11=sig11,sig12=sig12,sigH2=sigH2,\
                  sig21=sig21,sig22=sig22,sigH0=sigH0,sig0=sig0,\
                  method='leastsq',plot=0)
        count=count+1
        #
        if abs(res.chisqr-tmp_chisqr) < 0.000001:
           print('We are Successful after'+str(count)+' runs')
           break
    #plot the best curve
    if plot:
       lmfit_plot1dob(cfit[0,:],cfit[1,:],cfit[2,:])
#############################################################################
def fit_g1rsolo(g1r,W,D1):
    """
    Fit the IDF of isolated particles. It does not work well for the 
    determination of diameter of micro-void. Try lmfit.minimize 
    """
    #clip the negative values.
    g1r=clip_neg0(g1r)
    #get the r and g1
    r=g1r[:,0]
    g1=g1r[:,1]
    #determine the s
    intf=intfer1d(g1r,timz=8)
    sv=intf[:,0]
    W,D1=float(W),float(D1)
    g1r_mod=Model(g1r_stack,prefix='g1r_')
     #print g1r_mod.param_names
    #set the parameters
    g1r_mod.set_param_hint('g1r_W',value=W,min=0.0)
    g1r_mod.set_param_hint('g1r_D1',value=D1,min=0.1,max=100.0)
    g1r_mod.set_param_hint('g1r_sigH',value=0.3,min=0,max=1.0)
    g1r_mod.set_param_hint('g1r_sig1',value=0.3,min=0,max=1.0)
    #make the parameters
    pars=g1r_mod.make_params()
    #fit the curve
    #print g1.shape,s.shape
    res=g1r_mod.fit(g1,pars,s=sv)
    #print the fitting report
    print(res.fit_report())
    #print he RSS
    print('The RSS of best_fit ',np.sum(np.square(res.residual)))
    #return the fitting results
    lmfit_plot1d(r,g1,res.init_fit,res.best_fit)
    
    #write the fitted data
    cur_fit=np.zeros((res.ndata,3),dtype=np.float)
    cur_fit[:,0],cur_fit[:,1],cur_fit[:,2]=r,res.data,res.best_fit
    return res,cur_fit
#############################################################################
def minfit_g1rsolo(g1r,W,D1,sigH=0.3,sig1=0.3,method='leastsq'):
    """
    Fit the IDF of isolated particle. We can extract the average size and 
    one variable of distribution, one variable to present the deviation from
    the Gaussian distribution.
    """
    #get the r and g1
    ri,g1=g1r[:,0],g1r[:,1]
    #determine the s
    intf=intfer1d(g1r,timz=8)
    si=intf[:,0]
    W,D1,sigH,sig1=float(W),float(D1),float(sigH),float(sig1)
    
    #define the objective function:returns the array to be minimized
    def fcn2min(params,si,g1=None):
        W=params['g1r_W'].value
        D1=params['g1r_D1'].value
        sigH=params['g1r_sigH'].value
        sig1=params['g1r_sig1'].value
        model=g1r_solo(si,W,D1,sigH,sig1)
        if g1 is None:
           return model
        else:
           return model-g1
    
    #construct the Parameter and set the value and bound
    params=Parameters()
    params.add('g1r_W',value=W,min=0.0)
    params.add('g1r_D1',value=D1,min=0.0)
    params.add('g1r_sigH',value=sigH,min=0.0)
    params.add('g1r_sig1',value=sig1,min=0.0)
    #get the initial fit
    init_fit=fcn2min(params,si)
    res=minimize(fcn2min,params,args=(si,g1),method=method)
    print('# Number of fcn evaluations: ',res.nfev)
    print('# Number of variables      : ',res.nvarys)
    print('# Number of data Points    : ',res.ndata)
    print('chi-square                 : ',res.chisqr)
    print('redchi                     : ',res.redchi)
    report_fit(res.params)
    print('The RSS of best_fit is ', np.sum(np.square(res.residual)))
    #print res.params,
    #print params
    best_fit=fcn2min(res.params,si)
    lmfit_plot1d(ri,g1,init_fit,best_fit)
#############################################################################
def minfit_g1rduosolo(g1r,Wduo,D1,D2,Wsolo,sigH=0.3,sig1=0.3,sig2=0.3,\
                                   method='leastsq'):
    """
    This func fits the IDF of combination of solo (isolated particles) and
    duo (entities of twin-particle). Here we use the lmfit.minimize
    """
    #get the r and g1
    ri,g1=g1r[:,0],g1r[:,1]
    #determine the s
    intf=intfer1d(g1r,timz=8)
    si=intf[:,0]
    Wduo,D1,D2,Wsolo=float(Wduo),float(D1),float(D2),float(Wsolo)
    
    #define the objective function:returns the array to be minimized
    def fcn2min(params,si,g1=None):
        Wduo=params['g1r_Wduo'].value
        D1=params['g1r_D1'].value
        D2=params['g1r_D2'].value
        Wsolo=params['g1r_Wsolo'].value
        sigH=params['g1r_sigH'].value
        sig1=params['g1r_sig1'].value
        sig2=params['g1r_sig2'].value
        model=g1r_duo(si,Wduo,D1,D2,sigH,sig1,sig2)+\
                       g1r_solo(si,Wsolo,D1,sigH,sig1)
        if g1 is None:
           return model
        else:
           return model-g1
    #construct the Parameter and set the value and bound
    params=Parameters()
    params.add('g1r_Wduo',value=Wduo,min=0.0)
    params.add('g1r_D1',value=D1,min=0.0)
    params.add('g1r_D2',value=D2,min=0.0,max=400.0)
    params.add('g1r_Wsolo',value=Wsolo,min=0.0)
    params.add('g1r_sigH',value=sigH,min=0.0,max=1.0)
    params.add('g1r_sig1',value=sig1,min=0.0,max=1.0)
    params.add('g1r_sig2',value=sig2,min=0.0,max=1.0)
    #get the initial fit
    init_fit=fcn2min(params,si)
    res=minimize(fcn2min,params,args=(si,g1),method=method)
    print('# Number of fcn evaluations: ',res.nfev)
    print('# Number of variables      : ',res.nvarys)
    print('# Number of data Points    : ',res.ndata)
    print('chi-square                 : ',res.chisqr)
    print('redchi                     : ',res.redchi)
    report_fit(res.params)
    print('The RSS of best_fit is ', np.sum(np.square(res.residual)))
    #print res.params,
    #print params
    best_fit=fcn2min(res.params,si)
    lmfit_plot1d(ri,g1,init_fit,best_fit)
#############################################################################
def fit_g1rstack(g1r,W,D1,D2):
    """
    Fit the IDF. method:infinite stacks
    """
    r=g1r[:,0]
    g1=g1r[:,1]
    #determine the s
    intf=intfer1d(g1r,timz=8)
    sv=intf[:,0]
    W,D1,D2=float(W),float(D1),float(D2)
    #construct the model
    g1r_mod=Model(g1r_stack,prefix='g1r_')
    #print g1r_mod.param_names
    #set the parameters
    g1r_mod.set_param_hint('g1r_W',value=W,min=0.)
    g1r_mod.set_param_hint('g1r_D1',value=D1,min=0.1,max=10.0)
    #g1r_mod.set_param_hint('g1r_guardD',value=1.,min=0.1,max=20.0)
    g1r_mod.set_param_hint('g1r_D2',value=D2,min=0.1,max=30.0)#,\
    #                                   expr='g1r_D1+g1r_guardD')
    #g1r_mod.set_param_hint('g1r_D2',value=D2,min=0.1,max=10.0)
    g1r_mod.set_param_hint('g1r_sigH',value=0.3,min=0.05,max=1.0)
    g1r_mod.set_param_hint('g1r_sig1',value=0.3,min=0.05,max=1.0)
    g1r_mod.set_param_hint('g1r_sig2',value=0.3,min=0.05,max=1.0)
    #make these values parameterized
    pars=g1r_mod.make_params()
    #fit the curve
    #print g1.shape,s.shape
    res=g1r_mod.fit(g1,pars,s=sv)
    #print the fitting report
    print(res.fit_report())
    #print he RSS
    print('The RSS of best_fit ',np.sum(np.square(res.residual)))
    #return the fitting results
    lmfit_plot1d(r,g1,res.init_fit,res.best_fit)
    
    #write the fitted data
    cur_fit=np.zeros((res.ndata,3),dtype=np.float)
    cur_fit[:,0],cur_fit[:,1],cur_fit[:,2]=r,res.data,res.best_fit
    return res,cur_fit
###############################################################################
def minfit_g1rstack(g1r,Winf,D1,D2,sigH=0.3,sig1=0.3,sig2=0.3,method='leastsq'):
    """
    This function fits the IDF with one infinite stack.
    """
    r=g1r[:,0]
    g1=g1r[:,1]
    #determine the s
    intf=intfer1d(g1r,timz=8)
    sv=intf[:,0]
    Winf,D1,D2=float(Winf),float(D1),float(D2)
    
    #define the objective function:returns the array to be minimized
    def fcn2min(params,sv,g1=None):
        Winf=params['g1r_Winf'].value
        D1  =params['g1r_D1'].value
        D2  =params['g1r_D2'].value
        sigH=params['g1r_sigH'].value
        sig1=params['g1r_sig1'].value
        sig2=params['g1r_sig2'].value
        model=g1r_stack(sv,Winf,D1,D2,sigH,sig1,sig2)
        if g1 is None:
           return model
        return model-g1
    #construct the Parameter and set the value and bound
    params=Parameters()
    params.add('g1r_Winf',value=Winf,min=0.0)
    params.add('g1r_D1',value=D1,min=0.1,max=35.0)
    #params.add('g1r_guardD',value=1.,min=0.1,max=10.0)
    params.add('g1r_D2',value=D2,min=0.1,max=35.0)
    params.add('g1r_sigH',value=0.3,min=0.01,max=3.0)
    params.add('g1r_sig1',value=0.3,min=0.01,max=3.0)
    params.add('g1r_sig2',value=0.3,min=0.01,max=3.0)
    #get the initial fit
    init_fit=fcn2min(params,sv)
    #print(params)
    #fit the curve
    res=minimize(fcn2min,params,args=(sv,g1),method=method)
    #print the fitting report
    print('# Number of fcn evaluations: ',res.nfev)
    print('# Number of variables      : ',res.nvarys)
    print('# Number of data Points    : ',res.ndata)
    print('chi-square                 : ',res.chisqr)
    print('redchi                     : ',res.redchi)
    report_fit(res.params)
    #print the confidence interval.
    #ci=conf_interval(res,sigmas=[0.68,0.95],trace=True,verbose=False)
    #printfuncs.report_ci(ci)
    #print the  RSS
    print('The RSS of best_fit is ', np.sum(np.square(res.residual)))
    #plot the results.
    best_fit=fcn2min(res.params,sv)
    lmfit_plot1d(r,g1,init_fit,best_fit)
    #print(params)
    #write the fitted data
    cur_fit=np.zeros((res.ndata,3),dtype=np.float)
    cur_fit[:,0],cur_fit[:,1],cur_fit[:,2]=r,g1,best_fit
    return res,cur_fit
###############################################################################
def fit_g1r2stacks(g1r,W1st,D11,D12,W2nd,D21,D22):
    """
    This function fits the IDF curve with 2 stacks.
    """
    r=g1r[:,0]
    g1=g1r[:,1]
    #determine the s
    intf=intfer1d(g1r,timz=8)
    sv=intf[:,0]
    g1r_mod=Model(g1r_2stacks,prefix='g1r_')
    #convert the W1st and W2st to float
    W1st,W2nd=float(W1st),float(W2nd)
    #convert the D11 D12 D21 D22 to the float number
    D11,D12,D21,D22=float(D11),float(D12),float(D21),float(D22)
    #set the parameters
    #set the parameters of 1st group
    g1r_mod.set_param_hint('g1r_W1st',value=W1st,min=0.0)
    g1r_mod.set_param_hint('g1r_D11',value=D11,min=0.1)
    g1r_mod.set_param_hint('g1r_D12',value=D12,min=0.1)
    g1r_mod.set_param_hint('g1r_sig11',value=0.3,min=0.0,max=1.0)
    g1r_mod.set_param_hint('g1r_sig12',value=0.3,min=0.0,max=1.0)
    g1r_mod.set_param_hint('g1r_sigH1',value=0.3,min=0.0,max=1.0)
    #set the parameters of 2nd group
    g1r_mod.set_param_hint('g1r_W2nd',value=W2nd,min=0.0)
    #D21 and sig21 should be same as the D11 and sig11 (hard domain)
    g1r_mod.set_param_hint('g1r_D21',value=D21,min=0.0)#,expr='g1r_D11')
    g1r_mod.set_param_hint('g1r_sig21',value=0.3,min=0.0,max=1.0)#,\
    #                                                      expr='g1r_sig11')
    #set the parameters of D22
    #g1r_mod.set_param_hint('g1r_guardD',value=1.,min=0.1,max=50.0)
    g1r_mod.set_param_hint('g1r_D22',value=D22,min=0.0)#,\
    #                                     expr='g1r_D12+g1r_guardD')
    #g1r_mod.set_param_hint('g1r_D22',value=D22,min=0.1,max=30.0)
    g1r_mod.set_param_hint('g1r_sig22',value=0.3,min=0.0,max=1.0)
    g1r_mod.set_param_hint('g1r_sigH2',value=0.3,min=0.0,max=1.0)
    #make these values parameterized
    pars=g1r_mod.make_params()
    #fit the curve
    #print g1.shape,s.shape
    #print pars
    res=g1r_mod.fit(g1,pars,s=sv)
    #print the fitting report
    print(res.fit_report())
    print('The RSS of best_fit ',np.sum(np.square(res.residual)))
    #return the fitting results
    lmfit_plot1d(r,g1,res.init_fit,res.best_fit)
    
    #save the fitted data
    cur_fit=np.zeros((res.ndata,3),dtype=np.float)
    cur_fit[:,0],cur_fit[:,1],cur_fit[:,2]=r,res.data,res.best_fit
    return res,cur_fit
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def minfit_g1r2stacks(g1r,W1st,D11,D12,W2nd,D21,D22,sigH1=0.3,sig11=0.3,\
                  sig12=0.3,sigH2=0.3,sig21=0.3,sig22=0.3,method='leastsq'):
    """
    Fit the IDF in lmfit.minimize method. We use a combination model of
    two infinite stacks.
    """
    #get the r and g1
    ri,g1=g1r[:,0],g1r[:,1]
    #determine the s
    intf=intfer1d(g1r,timz=8)
    si=intf[:,0]
    W1st,D11,D12=float(W1st),float(D11),float(D12)
    W2nd,D21,D22=float(W2nd),float(D21),float(D22)
    #
    #define the objective function:returns the array to be minimized
    def fcn2min(params,si,g1=None):
        W1st=params['g1r_W1st'].value
        D11=params['g1r_D11'].value
        D12=params['g1r_D12'].value
        sig11=params['g1r_sig11'].value
        sig12=params['g1r_sig12'].value
        sigH1=params['g1r_sigH1'].value
        W2nd=params['g1r_W2nd'].value
        D21=params['g1r_D21'].value
        D22=params['g1r_D22'].value
        sig21=params['g1r_sig21'].value
        sig22=params['g1r_sig22'].value
        sigH2=params['g1r_sigH2'].value
        #construc the function
        model=g1rmin_2stacks(si,W1st,W2nd,D11,D12,D21,D22,sigH1,sigH2,\
                    sig11,sig12,sig21,sig22)
        if g1 is None:
           return model
        else:
           return model-g1
    #
    #construct the Parameter and set the value and bound
    params=Parameters()
    params.add('g1r_W1st',value=W1st,min=0.0)
    params.add('g1r_D11',value=D11,min=0.0)
    params.add('g1r_D12',value=D12,min=0.0,max=100.0)
    params.add('g1r_sigH1',value=sigH1,min=0.0)
    params.add('g1r_sig11',value=sig11,min=0.0)
    params.add('g1r_sig12',value=sig12,min=0.0)
    params.add('g1r_W2nd',value=W2nd,min=0.0)
    params.add('g1r_D21',value=D21,min=0.0,expr='g1r_D11')
    params.add('g1r_D22',value=D22,min=0.0,max=100.0)
    params.add('g1r_sigH2',value=sigH2,min=0.0)
    params.add('g1r_sig21',value=sig21,min=0.0,expr='g1r_sig11')
    params.add('g1r_sig22',value=sig22,min=0.0)
    #get the initial fit
    init_fit=fcn2min(params,si)
    res=minimize(fcn2min,params,args=(si,g1),method=method)
    print('# Number of fcn evaluations: ',res.nfev)
    print('# Number of variables      : ',res.nvarys)
    print('# Number of data Points    : ',res.ndata)
    print('chi-square                 : ',res.chisqr)
    print('redchi                     : ',res.redchi)
    report_fit(res.params)
    print('The RSS of best_fit is ', np.sum(np.square(res.residual)))
    #print res.params,
    #print params
    best_fit=fcn2min(res.params,si)
    #print type(res.params)
    lmfit_plot1d(ri,g1,init_fit,best_fit)
     
    #save the fitted data
    cur_fit=np.zeros((res.ndata,3),dtype=np.float)
    cur_fit[:,0],cur_fit[:,1],cur_fit[:,2]=ri,g1,best_fit
    return res,cur_fit
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def minfit_g1r2stacksolo(g1r,W1st,D11,D12,W2nd,D21,D22,Wuno,D0,sigH1=0.3,\
                  sig11=0.3,sig12=0.3,sigH2=0.3,sig21=0.3,sig22=0.3,\
                  sigH0=0.3,sig0=0.3,method='leastsq',plot=True):
    """
    Fit the IDF in lmfit.minimize method. We use a combination model of
    two infinite stacks.
    """
    #get the r and g1
    ri,g1=g1r[:,0],g1r[:,1]
    #determine the s
    intf=intfer1d(g1r,timz=8)
    si=intf[:,0]
    W1st,D11,D12=float(W1st),float(D11),float(D12)
    W2nd,D21,D22=float(W2nd),float(D21),float(D22)
    #
    #define the objective function:returns the array to be minimized
    def fcn2min(params,si,g1=None):
        W1st=params['g1r_W1st'].value
        D11=params['g1r_D11'].value
        D12=params['g1r_D12'].value
        sig11=params['g1r_sig11'].value
        sig12=params['g1r_sig12'].value
        sigH1=params['g1r_sigH1'].value
        W2nd=params['g1r_W2nd'].value
        D21=params['g1r_D21'].value
        D22=params['g1r_D22'].value
        sig21=params['g1r_sig21'].value
        sig22=params['g1r_sig22'].value
        sigH2=params['g1r_sigH2'].value
        Wuno=params['g1r_Wuno'].value
        D0=params['g1r_D0'].value
        sigH0=params['g1r_sigH0'].value
        sig0=params['g1r_sig0'].value
        #construc the function
        model=g1rmin_2stacks(si,W1st,W2nd,D11,D12,D21,D22,sigH1,sigH2,\
                    sig11,sig12,sig21,sig22)+g1r_solo(si,Wuno,D0,sigH0,sig0)
        if g1 is None:
           return model
        else:
           return model-g1
    #
    #construct the Parameter and set the value and bound
    params=Parameters()
    params.add('g1r_W1st',value=W1st,min=0.0)
    params.add('g1r_D11',value=D11,min=0.0)
    params.add('g1r_D12',value=D12,min=0.0,max=100.0)
    params.add('g1r_sigH1',value=sigH1,min=0.0,max=1.0)
    params.add('g1r_sig11',value=sig11,min=0.0,max=1.0)
    params.add('g1r_sig12',value=sig12,min=0.0,max=1.0)
    params.add('g1r_W2nd',value=W2nd,min=0.0)
    params.add('g1r_D21',value=D21,min=0.0,expr='g1r_D11')
    params.add('g1r_D22',value=D22,min=0.0,max=100.0)
    params.add('g1r_sigH2',value=sigH2,min=0.0,max=1.0,expr='g1r_sigH1')
    params.add('g1r_sig21',value=sig21,min=0.0,max=1.0,expr='g1r_sig11')
    params.add('g1r_sig22',value=sig22,min=0.0,max=1.0)
    params.add('g1r_Wuno',value=Wuno,min=0.0)
    params.add('g1r_D0',value=D0,min=0.0,expr='g1r_D11')
    params.add('g1r_sigH0',value=sigH0,min=0.0,max=1.0,expr='g1r_sigH1')
    params.add('g1r_sig0',value=sig0,min=0.0,expr='g1r_sig11')
    #get the initial fit
    init_fit=fcn2min(params,si)
    res=minimize(fcn2min,params,args=(si,g1),method=method)
    print('# Number of fcn evaluations: ',res.nfev)
    print('# Number of variables      : ',res.nvarys)
    print('# Number of data Points    : ',res.ndata)
    print('chi-square                 : ',res.chisqr)
    print('redchi                     : ',res.redchi)
    report_fit(res.params)
    print('The RSS of best_fit is ', np.sum(np.square(res.residual)))
    #print res.params,
    #print params
    best_fit=fcn2min(res.params,si)
    #print type(res.params)
    if plot is True:
       lmfit_plot1d(ri,g1,init_fit,best_fit)
    
    #save the fitted data
    cur_fit=np.zeros((res.ndata,3),dtype=np.float)
    cur_fit[:,0],cur_fit[:,1],cur_fit[:,2]=ri,g1,best_fit
    return res,cur_fit
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fit_g1rstacksolo(g1r,Winf,D1,D2,Wuno):
    """
    This function fits the IDF with one infinite stack and one solo.
    """
    r=g1r[:,0]
    g1=g1r[:,1]
    #determine the s
    intf=intfer1d(g1r,timz=8)
    sv=intf[:,0]
    Winf,D1,D2,Wuno=float(Winf),float(D1),float(D2),float(Wuno)
    #construct the model
    g1r_mod=Model(g1r_stack_solo,prefix='g1r_')
    #print g1r_mod.param_names
    #set the parameters
    g1r_mod.set_param_hint('g1r_Winf',value=Winf,min=0.)
    g1r_mod.set_param_hint('g1r_D1',value=D1,min=0.1,max=10.0)
    #g1r_mod.set_param_hint('g1r_guardD',value=1.,min=0.1,max=10.0)
    #g1r_mod.set_param_hint('g1r_D2',value=D2,min=0.1,max=25.0)
    g1r_mod.set_param_hint('g1r_D2',value=D2,min=0.1,max=25)#,\
    #                         expr='g1r_D1+g1r_guardD')
    g1r_mod.set_param_hint('g1r_Wuno',value=Wuno,min=0.)
    g1r_mod.set_param_hint('g1r_sigH',value=0.3,min=0,max=1.0)
    g1r_mod.set_param_hint('g1r_sig1',value=0.3,min=0,max=1.0)
    g1r_mod.set_param_hint('g1r_sig2',value=0.3,min=0,max=1.0)
    #make these values parameterized
    pars=g1r_mod.make_params()
    #fit the curve
    #print g1.shape,s.shape
    res=g1r_mod.fit(g1,pars,s=sv)
    #print the fitting report
    print(res.fit_report())
    print('The RSS of best_fit is ', np.sum(np.square(res.residual)))
    #plot the results.
    lmfit_plot1d(r,g1,res.init_fit,res.best_fit)
    
    #write the fitted data
    cur_fit=np.zeros((res.ndata,3),dtype=np.float)
    cur_fit[:,0],cur_fit[:,1],cur_fit[:,2]=r,res.data,res.best_fit
    return res,cur_fit
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fit_g1rstacksolo_rng(g1r,Winf,D1,D2,Wuno):
    """
    This func is used to fit the IDF with one infinite stack and one solo
    in the specified range.
    """
    #set the range and zoom the curve
    tmp_g1r=deepcopy(g1r)
    tmp_g1r=cutcur(g1r,stksolo_rng)
    tmp_g1r=curv_zoom(tmp_g1r,256.0/tmp_g1r.shape[0])
    r=tmp_g1r[:,0]
    g1=tmp_g1r[:,1]
    #determine the s
    intf=intfer1d(g1r,timz=8)
    sv=intf[:,0]
    Winf,D1,D2,Wuno=float(Winf),float(D1),float(D2),float(Wuno)
    #construct the model
    g1r_mod=Model(g1r_stack_solo_rng,prefix='g1r_')
    #print g1r_mod.param_names
    #set the parameters
    g1r_mod.set_param_hint('g1r_Winf',value=Winf,min=0.)
    g1r_mod.set_param_hint('g1r_D1',value=D1,min=0.1,max=10.0)
    #g1r_mod.set_param_hint('g1r_guardD',value=1.,min=0.1,max=10.0)
    #g1r_mod.set_param_hint('g1r_D2',value=D2,min=0.1,max=25.0)
    g1r_mod.set_param_hint('g1r_D2',value=D2,min=0.1,max=25)#,\
    #                         expr='g1r_D1+g1r_guardD')
    g1r_mod.set_param_hint('g1r_Wuno',value=Wuno,min=0.)
    g1r_mod.set_param_hint('g1r_sigH',value=0.3,min=0,max=1.0)
    g1r_mod.set_param_hint('g1r_sig1',value=0.3,min=0,max=1.0)
    g1r_mod.set_param_hint('g1r_sig2',value=0.3,min=0,max=1.0)
    #make these values parameterized
    pars=g1r_mod.make_params()
    #fit the curve
    #print g1.shape,s.shape
    res=g1r_mod.fit(g1,pars,s=sv)
    #print the fitting report
    print(res.fit_report())
    print('The RSS of best_fit is ', np.sum(np.square(res.residual)))
    #plot the results.
    lmfit_plot1d(r,g1,res.init_fit,res.best_fit)

    #write the fitted data
    cur_fit=np.zeros((res.ndata,3),dtype=np.float)
    cur_fit[:,0],cur_fit[:,1],cur_fit[:,2]=r,res.data,res.best_fit
    return res,cur_fit
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fit_g1rstacksolo_indep(g1r,Winf,D1,D2,Wuno,Duno):
    """
    This function fits the IDF with one infinite stack and one solo.
    """
    r=g1r[:,0]
    g1=g1r[:,1]
    #determine the s
    intf=intfer1d(g1r,timz=8)
    sv=intf[:,0]
    Winf,D1,D2,Wuno,Duno=float(Winf),float(D1),float(D2),float(Wuno),float(Duno)
    #construct the model
    g1r_mod=Model(g1r_stack_solo_indep,prefix='g1r_')
    #print g1r_mod.param_names
    #set the parameters
    g1r_mod.set_param_hint('g1r_Winf',value=Winf,min=0.)
    g1r_mod.set_param_hint('g1r_D1',value=D1,min=0.1,max=10.0)
    g1r_mod.set_param_hint('g1r_D2',value=D2,min=0.1,max=25)
    g1r_mod.set_param_hint('g1r_Wuno',value=Wuno,min=0.)
    g1r_mod.set_param_hint('g1r_Duno',value=Duno,min=0.)
    g1r_mod.set_param_hint('g1r_sigHinf',value=0.3,min=0,max=1.0)
    g1r_mod.set_param_hint('g1r_sigHuno',value=0.3,min=0,max=1.0)
    g1r_mod.set_param_hint('g1r_sig1',value=0.3,min=0,max=1.0)
    g1r_mod.set_param_hint('g1r_sig2',value=0.3,min=0,max=1.0)
    g1r_mod.set_param_hint('g1r_siguno',value=0.3,min=0,max=1.0)
    #make these values parameterized
    pars=g1r_mod.make_params()
    #fit the curve
    #print g1.shape,s.shape
    res=g1r_mod.fit(g1,pars,s=sv)
    #print the fitting report
    print(res.fit_report())
    print('The RSS of best_fit is ', np.sum(np.square(res.residual)))
    #plot the results.
    lmfit_plot1d(r,g1,res.init_fit,res.best_fit)
    
    #write the fitted data
    cur_fit=np.zeros((res.ndata,3),dtype=np.float)
    cur_fit[:,0],cur_fit[:,1],cur_fit[:,2]=r,res.data,res.best_fit
    return res,cur_fit
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def minfit_g1rstacksolo(g1r,Winf,D1,D2,Wuno,method='leastsq'):
    """
    This function fits the IDF with one infinite stack and one solo.
    """
    r=g1r[:,0]
    g1=g1r[:,1]
    #determine the s
    intf=intfer1d(g1r,timz=8)
    sv=intf[:,0]
    Winf,D1,D2,Wuno=float(Winf),float(D1),float(D2),float(Wuno)
    
    #define the objective function:returns the array to be minimized
    def fcn2min(params,sv,g1=None):
        Winf=params['g1r_Winf'].value
        D1  =params['g1r_D1'].value
        D2  =params['g1r_D2'].value
        Wuno=params['g1r_Wuno'].value
        sigH=params['g1r_sigH'].value
        sig1=params['g1r_sig1'].value
        sig2=params['g1r_sig2'].value
        model=g1r_stack(sv,Winf,D1,D2,sigH,sig1,sig2)+\
                   g1r_uno(sv,Wuno,D1,sigH,sig1)
        if g1 is None:
           return model
        return model-g1
    #construct the Parameter and set the value and bound
    params=Parameters()
    params.add('g1r_Winf',value=Winf,min=0.0)
    params.add('g1r_D1',value=D1,min=0.1,max=25.0)
    #params.add('g1r_guardD',value=1.,min=0.1,max=10.0)
    params.add('g1r_D2',value=D2,min=0.1,max=25.0)
    params.add('g1r_Wuno',value=Wuno,min=0.)
    params.add('g1r_sigH',value=0.3,min=0.05,max=1.0)
    params.add('g1r_sig1',value=0.3,min=0.05,max=1.0)
    params.add('g1r_sig2',value=0.3,min=0.05,max=1.0)
    
    #get the initial fit
    init_fit=fcn2min(params,sv)
    #print(params)
    #fit the curve
    res=minimize(fcn2min,params,args=(sv,g1),method=method)
    #print the fitting report
    print('# Number of fcn evaluations: ',res.nfev)
    print('# Number of variables      : ',res.nvarys)
    print('# Number of data Points    : ',res.ndata)
    print('chi-square                 : ',res.chisqr)
    print('redchi                     : ',res.redchi)
    report_fit(res.params)
    #print the confidence interval.
    #ci=conf_interval(res,sigmas=[0.68,0.95],trace=True,verbose=False)
    #printfuncs.report_ci(ci)
    #print the  RSS
    print('The RSS of best_fit is ', np.sum(np.square(res.residual)))
    #plot the results.
    best_fit=fcn2min(res.params,sv)
    lmfit_plot1d(r,g1,init_fit,best_fit)
    #print(params)
    #write the fitted data
    cur_fit=np.zeros((res.ndata,3),dtype=np.float)
    cur_fit[:,0],cur_fit[:,1],cur_fit[:,2]=r,g1,best_fit
    return res,cur_fit
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fit_g1r123stacks(g1r,W1,D1,D2,W2,W3,c=0.5,coupling=True):
    """
    This function is to fit the continued clusters. It contains solos and duos
    and trios
    """
    r=g1r[:,0]
    g1=g1r[:,1]
    #determine the s
    intf=intfer1d(g1r,timz=8)
    sv=intf[:,0]
    W1,D1,D2,c=float(W1),float(D1),float(D2),float(c)
    #construct the model
    g1r_mod=Model(g1r_123stacks,prefix='g1r_')
    #set the parameters.
    g1r_mod.set_param_hint('g1r_c',value=c,min=0.)
    g1r_mod.set_param_hint('g1r_W1',value=W1,min=0.)
    if coupling is True:
       g1r_mod.set_param_hint('g1r_W2',value=W2,min=0.,\
                      expr='g1r_W1*(1-g1r_c)*g1r_c')
       g1r_mod.set_param_hint('g1r_W3',value=W3,min=0.,\
                      expr='g1r_W2*(1-g1r_c)**2*g1r_c')
    else:
       g1r_mod.set_param_hint('g1r_W2',value=W2,min=0.,max=10.)
       g1r_mod.set_param_hint('g1r_W3',value=W3,min=0.,max=10.)
    g1r_mod.set_param_hint('g1r_D1',value=D1,min=0.0,max=10.)
    g1r_mod.set_param_hint('g1r_D2',value=D2,min=0.0,max=10.)
    g1r_mod.set_param_hint('g1r_sigH',value=0.3,min=0.0,max=1.0)
    g1r_mod.set_param_hint('g1r_sig1',value=0.3,min=0.0,max=1.0)
    g1r_mod.set_param_hint('g1r_sig2',value=0.3,min=0.0,max=1.0)
    #make these values parameterized
    pars=g1r_mod.make_params()
    #fit the curve
    #print g1.shape,s.shape
    res=g1r_mod.fit(g1,pars,s=sv)
    #print the fitting report
    print(res.fit_report())
    print('The RSS of best_fit is ', np.sum(np.square(res.residual)))
    #plot the results.
    lmfit_plot1d(r,g1,res.init_fit,res.best_fit)
    
    #write the fitted data
    cur_fit=np.zeros((res.ndata,3),dtype=np.float)
    cur_fit[:,0],cur_fit[:,1],cur_fit[:,2]=r,res.data,res.best_fit
    return res,cur_fit
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fitwrite(res,cur,name):
    """
    Write the results of fitting into hard disk.
    """
    #write the fitting results into a text file.
    #print(res.best_values)
    #The parameters and their intervals
    #print(res.init_params)
    if '.' in name:
       name=name.split('.')[0]+'_fit'
    histname=name+'.hist'
    curvname=name+'.curv'
    fid=open(histname,'w')
    fid.write('#*****Fit report*****#\n')
    fid.write(fit_report(res.params)+'\n')
    fid.write('RSS:'+str(np.sum(np.square(res.residual)))+'\n')
    #write the best values
    fid.write('#*****The fitted parameters*****#\n')
    for key,val in res.values.items():
        line='$'+key+' '+str(val)+' \n'
        fid.write(line)
    fid.close()
    
    #store the original curve and fit curve.
    comm='Original data and fitted data\n'
    comm=comm+'#1st col: r,2nd col: orig,3rd col: best_fit\n'
    np.savetxt(curvname,cur,delimiter=' ',header=comm)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def minfitwrite(res,cur,name):
    """
    Write the fitted results onto the disk.
    """
    #prepare the stored files
    if '.' in name:
       name=name.split('.')[0]+'_fit'
    histname=name+'.hist'
    curvname=name+'.curv'
    fid=open(histname,'w')
    fid.write('#*****Fit report*****#\n')
    #write the report
    fid.write(fit_report(res.params)+'\n')
    fid.write('RSS:'+str(np.sum(np.square(res.residual)))+'\n')
    #write the best values
    fid.write('#*****The fitted parameters*****#\n')
    for key,val in res.params.items():
        line='$'+key+' '+str(val)+' \n'
        fid.write(line)
    fid.close()
    
    #store the original curve and fit curve.
    comm='Original data and fitted data\n'
    comm=comm+'#1st col: r,2nd col: orig,3rd col: best_fit\n'
    np.savetxt(curvname,cur,delimiter=' ',header=comm)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def decomp_2infstacks(g1r,res):
    """
    Decompose the measured fitted IDF to two IDFs.
    res is the module returned from the fitting.
    """
    #the real vector
    r=g1r[:,0]
    
    intf=intfer1d(g1r,timz=8)
    #the reciprocal vector
    s=intf[:,0]
    
    bv=res.values
    #the first stack
    g1r1_y=g1r_stack(s,bv['g1r_W1st'],bv['g1r_D11'],bv['g1r_D12'],\
                bv['g1r_sigH1'],bv['g1r_sig11'],bv['g1r_sig12'],0.)
    #the second stack
    g1r2_y=g1r_stack(s,bv['g1r_W2nd'],bv['g1r_D21'],bv['g1r_D22'],\
                bv['g1r_sigH2'],bv['g1r_sig21'],bv['g1r_sig22'],0.)
    
    #two idfs
    g1r1=np.zeros((r.shape[0],2),dtype=np.float)
    g1r1[:,0],g1r1[:,1]=r,g1r1_y
    g1r2=np.zeros((r.shape[0],2),dtype=np.float)
    g1r2[:,0],g1r2[:,1]=r,g1r2_y
    
    #plot the curves
    cplot(g1r,label='g1r_fit')
    cplot(g1r1,label='stack1')
    cplot(g1r2,label='stack2')
    
    return g1r1,g1r2
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def decompmin_2infstacks(g1r,res):
    """
    Decompose the fitted IDF into 2 IDFs, here the IDF is fitted in
    lmfit.minimize method.
    """
    #the real vector
    r=g1r[:,0]

    intf=intfer1d(g1r,timz=8)
    #the reciprocal vector
    s=intf[:,0]

    bv=res.params
    #the first stack
    g1r1_y=g1r_stack(s,bv['g1r_W1st'],bv['g1r_D11'],bv['g1r_D12'],\
                bv['g1r_sigH1'],bv['g1r_sig11'],bv['g1r_sig12'])
    #the second stack
    g1r2_y=g1r_stack(s,bv['g1r_W2nd'],bv['g1r_D21'],bv['g1r_D22'],\
                bv['g1r_sigH2'],bv['g1r_sig21'],bv['g1r_sig22'])
    
    #two idfs
    g1r1=np.zeros((r.shape[0],2),dtype=np.float)
    g1r1[:,0],g1r1[:,1]=r,g1r1_y
    g1r2=np.zeros((r.shape[0],2),dtype=np.float)
    g1r2[:,0],g1r2[:,1]=r,g1r2_y
    #plot the curves
    cplot(g1r,label='g1r_fit')
    cplot(g1r1,label='stack1')
    cplot(g1r2,label='stack2')

    return g1r1,g1r2
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def decompmin_2infstacksolo(g1r,res):
    """
    Decompose the fitted IDF into 3 IDFs 2 infIDF and 1 solo.
    here the IDF is fitted in lmfit.minimize method.
    """
    #the real vector
    r=g1r[:,0]

    intf=intfer1d(g1r,timz=8)
    #the reciprocal vector
    s=intf[:,0]

    bv=res.params
    #the first stack
    g1r1_y=g1r_stack(s,bv['g1r_W1st'],bv['g1r_D11'],bv['g1r_D12'],\
                bv['g1r_sigH1'],bv['g1r_sig11'],bv['g1r_sig12'])
    #the second stack
    g1r2_y=g1r_stack(s,bv['g1r_W2nd'],bv['g1r_D21'],bv['g1r_D22'],\
                bv['g1r_sigH2'],bv['g1r_sig21'],bv['g1r_sig22'])
    g1r3_y=g1r_solo(s,bv['g1r_Wuno'],bv['g1r_D0'],bv['g1r_sigH0'],\
                bv['g1r_sig0'])
    #three idfs
    g1r1=np.zeros((r.shape[0],2),dtype=np.float)
    g1r1[:,0],g1r1[:,1]=r,g1r1_y
    g1r2=np.zeros((r.shape[0],2),dtype=np.float)
    g1r2[:,0],g1r2[:,1]=r,g1r2_y
    g1r3=np.zeros((r.shape[0],2),dtype=np.float)
    g1r3[:,0],g1r3[:,1]=r,g1r3_y
    
    #plot the curves
    cplot(g1r,label='g1r_fit')
    cplot(g1r1,label='stack1')
    cplot(g1r2,label='stack2')
    cplot(g1r3,label='solo')
    return g1r1,g1r2,g1r3
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def decomp_infsolo(g1r,res):
    """
    Decompose the IDF into two components, one infinite stack and one solo.
    """
    #the real vector
    r=g1r[:,0]

    intf=intfer1d(g1r,timz=8)
    #the reciprocal vector
    s=intf[:,0]

    bv=res.values
    #the first stack
    g1r1_y=g1r_stack(s,bv['g1r_Winf'],bv['g1r_D1'],bv['g1r_D2'],\
                bv['g1r_sigH'],bv['g1r_sig1'],bv['g1r_sig2'])
    #the solos
    g1r2_y=g1r_uno(s,bv['g1r_Wuno'],bv['g1r_D1'],bv['g1r_sigH'],bv['g1r_sig1'])
    
    #two idfs
    g1r1=np.zeros((r.shape[0],2),dtype=np.float)
    g1r1[:,0],g1r1[:,1]=r,g1r1_y
    g1r2=np.zeros((r.shape[0],2),dtype=np.float)
    g1r2[:,0],g1r2[:,1]=r,g1r2_y

    #plot the curves
    #cplot(g1r,label='g1r_fit',line=':',N=2)
    cplot(g1r1,label='stack1')
    cplot(g1r2,label='solos')

    return g1r1,g1r2
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def decomp_infsolo_minfit(g1r,res):
    """
    This func returns the decomposed IDF curves from the IDF fitting in 
    Minimize method.  
    """
    #get the r axis
    r=g1r[:,0]
    
    #get the s axis
    intf=intfer1d(g1r,timz=8)
    s=intf[:,0]
    
    #get the values from the keys
    Winf=res.params['g1r_Winf'].value
    D1=res.params['g1r_D1'].value
    D2=res.params['g1r_D2'].value
    Wuno=res.params['g1r_Wuno'].value
    sigH=res.params['g1r_sigH'].value
    sig1=res.params['g1r_sig1'].value
    sig2=res.params['g1r_sig2'].value
    #
    #get the g1rinf
    g1r1_y=g1r_stack(s,Winf,D1,D2,sigH,sig1,sig2)
    #the solos
    g1r2_y=g1r_uno(s,Wuno,D1,sigH,sig1)
    
    #two idfs
    g1r1=np.zeros((r.shape[0],2),dtype=np.float)
    g1r1[:,0],g1r1[:,1]=r,g1r1_y
    g1r2=np.zeros((r.shape[0],2),dtype=np.float)
    g1r2[:,0],g1r2[:,1]=r,g1r2_y

    #plot the curves
    cplot(g1r,label='g1r_fit',line=':')
    cplot(g1r1,label='stack1')
    cplot(g1r2,label='solos')

    return g1r1,g1r2
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def decomp_infsolo_indep(g1r,res):
    """
    Decompose the IDF into two components, one infinite stack and one solo.
    """
    #the real vector
    r=g1r[:,0]

    intf=intfer1d(g1r,timz=8)
    #the reciprocal vector
    s=intf[:,0]

    bv=res.values
    #the first stack
    g1r1_y=g1r_stack(s,bv['g1r_Winf'],bv['g1r_D1'],bv['g1r_D2'],\
                bv['g1r_sigHinf'],bv['g1r_sig1'],bv['g1r_sig2'],0.)
    #the solos
    g1r2_y=g1r_uno(s,bv['g1r_Wuno'],bv['g1r_Duno'],bv['g1r_sigHuno'],\
                   bv['g1r_siguno'])
    
    #two idfs
    g1r1=np.zeros((r.shape[0],2),dtype=np.float)
    g1r1[:,0],g1r1[:,1]=r,g1r1_y
    g1r2=np.zeros((r.shape[0],2),dtype=np.float)
    g1r2[:,0],g1r2[:,1]=r,g1r2_y

    #plot the curves
    cplot(g1r,label='g1r_fit',line=':')
    cplot(g1r1,label='stack1')
    cplot(g1r2,label='solos')

    return g1r1,g1r2
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def g1r_123stacks(s,W1,D1,D2,sigH,sig1,sig2,W2,W3,c):
    """
    This function is used to construct the 123 stacks (solos,duos,trios).
    """
    Fs=G1s_123stacks(s,W1,D1,D2,sigH,sig1,sig2,W2,W3)
    G1s_tmp=np.zeros((len(Fs),2),dtype=np.float)
    G1s_tmp[:,0]=s
    G1s_tmp[:,1]=Fs*(-1.0)
    g1r=idf1d(G1s_tmp)
    return g1r[:,1]
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def g1r_stack_solo_indep(s,Winf,D1,D2,sigHinf,sig1,sig2,Wuno,Duno,sigHuno,\
                           siguno):
    """
    Computed IDF from interference function of infinite stacks and independent
    solos.
    """
    Fs=G1s_stack(s,Winf,D1,D2,sigHinf,sig1,sig2)+\
                G1s_uno(s,Wuno,Duno,sigHuno,siguno)
    G1s_tmp=np.zeros((len(Fs),2),dtype=np.float)
    G1s_tmp[:,0]=s
    G1s_tmp[:,1]=Fs*(-1.0)
    g1r=idf1d(G1s_tmp)
    return g1r[:,1]
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def g1r_stack_solo(s,Winf,D1,D2,sigH,sig1,sig2,Wuno):#,guardD):
    """
    Compute IDF from interference function of two types of stacking.
    One is infinite stacks and another one is solo domain.
    The doamin size of solos is same as the size of D1
    """
    Fs=G1s_stack(s,Winf,D1,D2,sigH,sig1,sig2)+G1s_uno(s,Wuno,D1,sigH,sig1)
    G1s_tmp=np.zeros((len(Fs),2),dtype=np.float)
    G1s_tmp[:,0]=s
    G1s_tmp[:,1]=Fs*(-1.0)
    g1r=idf1d(G1s_tmp)
    return g1r[:,1]
#--------------------------------------------------------------------
def g1r_stack_solo_rng(s,Winf,D1,D2,sigH,sig1,sig2,Wuno):#,guardD):
    """
    Compute IDF from interference function of two types of stacking.
    One is infinite stacks and another one is solo domain.
    The doamin size of solos is same as the size of D1
    """
    Fs=G1s_stack(s,Winf,D1,D2,sigH,sig1,sig2)+G1s_uno(s,Wuno,D1,sigH,sig1)
    G1s_tmp=np.zeros((len(Fs),2),dtype=np.float)
    G1s_tmp[:,0]=s
    G1s_tmp[:,1]=Fs*(-1.0)
    g1r=idf1d(G1s_tmp)
    g1r=cutcur(g1r,stksolo_rng)
    g1r=curv_zoom(g1r,256.0/g1r.shape[0])
    return g1r[:,1]
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def g1r_2stacks(s,W1st,W2nd,D11,D12,D21,D22,sigH1,sigH2,sig11,sig12,\
                    sig21,sig22):
    """
    Compute the IDF from the interference function.
    It composes of two infinite stacks.
    """
    Fs=G1s_2stacks(s,W1st,W2nd,D11,D12,D21,D22,sigH1,sigH2,sig11,sig12,\
                    sig21,sig22)
    #Fs(s) is the interference function which will be subjected to the FFT.
    G1s_tmp=np.zeros((len(Fs),2),dtype=np.float)
    G1s_tmp[:,0]=s
    G1s_tmp[:,1]=Fs*(-1.0)
    g1r=idf1d(G1s_tmp)
    return g1r[:,1]
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def g1rmin_2stacks(s,W1st,W2nd,D11,D12,D21,D22,sigH1,sigH2,sig11,sig12,\
                    sig21,sig22):
    """
    Compute the IDF from the interference function.
    It composes of two infinite stacks.
    """
    Fs=G1s_2stacks(s,W1st,W2nd,D11,D12,D21,D22,sigH1,sigH2,sig11,sig12,\
                    sig21,sig22)
    #Fs(s) is the interference function which will be subjected to the FFT.
    G1s_tmp=np.zeros((len(Fs),2),dtype=np.float)
    G1s_tmp[:,0]=s
    G1s_tmp[:,1]=Fs*(-1.0)
    g1r=idf1d(G1s_tmp)
    return g1r[:,1]
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def g1r_duo(s,W,D1,D2,sigH,sig1,sig2):
    """
    This func returns the IDF computed from G1s_duo.
    """
    Fs=G1s_duo(s,W,D1,D2,sigH,sig1,sig2)
    #Fs(s) is the interference function which will be subjected to the FFT.
    G1s_tmp=np.zeros((len(Fs),2),dtype=np.float)
    G1s_tmp[:,0]=s
    G1s_tmp[:,1]=Fs*(-1.0)
    g1r=idf1d(G1s_tmp)
    return g1r[:,1]
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def g1r_uno(s,W,D1,sigH,sig1):
    """
    This function computes the IDF from the G1s_uno.
    """
    Fs=G1s_uno(s,W,D1,sigH,sig1)
    #Fs(s) is the interference function which will be subjected to the FFT.
    G1s_tmp=np.zeros((len(Fs),2),dtype=np.float)
    G1s_tmp[:,0]=s
    G1s_tmp[:,1]=Fs*(-1.0)
    g1r=idf1d(G1s_tmp)
    return g1r[:,1]
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def g1r_stack(s,W,D1,D2,sigH,sig1,sig2):#,guardD):
    """
    To compute the IDF from the interference function.
    """
    Fs=G1s_stack(s,W,D1,D2,sigH,sig1,sig2)
    #Fs(s) is the interference function which will be subjected to the FFT.
    G1s_tmp=np.zeros((len(Fs),2),dtype=np.float)
    G1s_tmp[:,0]=s
    G1s_tmp[:,1]=Fs*(-1.0)
    g1r=idf1d(G1s_tmp,timz=8)
    return g1r[:,1]
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def g1r_solo(s,W,D1,sigH,sig1):
    """
    To compute the IDF from the interference function.
    """
    Fs=G1s_uno(s,W,D1,sigH,sig1)
    #Fs(s) is the interference function which will be subjected to the FFT.
    G1s_tmp=np.zeros((len(Fs),2),dtype=np.float)
    G1s_tmp[:,0]=s
    G1s_tmp[:,1]=Fs*(-1.0)
    g1r=idf1d(G1s_tmp,timz=8)
    return g1r[:,1]
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def G1s_123stacks(s,W1,D1,D2,sigH,sig1,sig2,W2,W3):
    """
    Construct the G1s of 123 stacks (solos,duos,trios)
    """
    Fs=G1s_uno(s,W1,D1,sigH,sig1)+\
          G1s_duo(s,W2,D1,D2,sigH,sig1,sig2)+\
             G1s_trio(s,W3,D1,D2,sigH,sig1,sig2)
    return Fs
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def G1s_2stacks(s,W1st,W2nd,D11,D12,D21,D22,sigH1,sigH2,sig11,sig12,\
                                       sig21,sig22):
    """
    Interference functio of two stacks.
    """
    Fs=G1s_stack(s,W1st,D11,D12,sigH1,sig11,sig12)+\
        G1s_stack(s,W2nd,D21,D22,sigH2,sig21,sig22)
    return Fs
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def G1s_stack(s,W,D1,D2,sigH,sig1,sig2):
    """
    Interference function of stacking model
    """
    maxdist=25
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #Build initial Fs
    Fs=np.zeros(s.shape,dtype=np.float)
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    L=D2+D1
    Quad1=(sig1*D1)**2
    QuadL=Quad1+(sig2*D2)**2
    
    mul1=0
    mulL=0
    for i in range(0,maxdist):
        if mul1==1 :
           mul1=mul1-2
           mulL=mulL+1
        else:
           mul1=mul1+1
        Di=mul1*D1+mulL*L
        sigg=(mul1*Quad1+mulL*QuadL)**0.5/Di
        if mul1==0 :
           Fs=funcadd(s,Fs,sigg,sigH,Di,-2.0*W)
        else:
           Fs=funcadd(s,Fs,sigg,sigH,Di,W)
    return Fs
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def G1s_lattmod(s,W,D1,D2,sigH,sig1,sigL):
    """
    Interference function of Lattice model.
    """
    maxdist=25
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #Build initial Fs
    Fs=np.zeros(s.shape,dtype=np.float)
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    L=D1+D2
    sig1Q2=(sig1*D1)**2*0.5
    sigLQ =(sigL*L)**2
    
    mulL,mul1=0,0
    for i in range(maxdist):
        if mul1==1:
           mul1=mul1-2
           mulL=mulL+1
        else:
           mul1=mul1+1
        Di=mul1*D1+mulL*L
        sigg=(sig1Q2+mulL*sigLQ)**0.5/Di
        #
        if mul1==0:
           Fs=funcadd(s,Fs,sigg,sigH,Di,-2.0*W)
        else:
           Fs=funcadd(s,Fs,sigg,sigH,Di,W)
    
    return Fs
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def G1s_quintette(s,W,D1,D2,sigH,sig1,sig2):
    """
    Interference function combined by quintette-stacks.
    """
    maxdist=13
    multis=set_multiplier(5)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Build initial Fs
    Fs=np.zeros(len(s),dtype=np.float)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    L=D2+D1
    Quad1=(sig1*D1)**2
    QuadL=Quad1+(sig2*D2)**2
    mulL=0
    mul1=0
    for i in range(maxdist):
        if mul1==1:
           mul1=mul1-2
           mulL=mulL+1
        else:
           mul1=mul1+1
        Di=mul1*D1+mulL*L
        sigg=(mul1*Quad1+mulL*QuadL)**0.5/Di
        Fs=funcadd(s,Fs,sigg,sigH,Di,multis[i]*W)
    return Fs
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def G1s_quartet(s,W,D1,D2,sigH,sig1,sig2):
    """
    Interference function combined by quartet-stacks.
    """
    maxdist=10
    multis=set_multiplier(4)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Build initial Fs
    Fs=np.zeros(len(s),dtype=np.float)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    L=D2+D1
    Quad1=(sig1*D1)**2
    QuadL=Quad1+(sig2*D2)**2
    mulL=0
    mul1=0
    for i in range(maxdist):
        if mul1==1:
           mul1=mul1-2
           mulL=mulL+1
        else:
           mul1=mul1+1
        Di=mul1*D1+mulL*L
        sigg=(mul1*Quad1+mulL*QuadL)**0.5/Di
        Fs=funcadd(s,Fs,sigg,sigH,Di,multis[i]*W)
    return Fs
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def G1s_trio(s,W,D1,D2,sigH,sig1,sig2):
    """
    Interference function combined by trio-stacks.
    """
    maxdist=7
    #multis=[1.5,1,-2,1,1,-2,0.5]
    multis=set_multiplier(3)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Build initial Fs
    Fs=np.zeros(len(s),dtype=np.float)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    L=D2+D1
    Quad1=(sig1*D1)**2
    QuadL=Quad1+(sig2*D2)**2
    mulL=0
    mul1=0
    for i in range(maxdist):
        if mul1==1:
           mul1=mul1-2
           mulL=mulL+1
        else:
           mul1=mul1+1
        Di=mul1*D1+mulL*L
        sigg=(mul1*Quad1+mulL*QuadL)**0.5/Di
        Fs=funcadd(s,Fs,sigg,sigH,Di,multis[i]*W)
    return Fs
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def G1s_duo(s,W,D1,D2,sigH,sig1,sig2):
    """
    Interference function of two-stacks combined by F(s)
    """
    maxdist=4
    #multis=[2,1,-2,1]
    multis=set_multiplier(2)
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    #Build initial Fs
    Fs=np.zeros(len(s),dtype=np.float)
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    L=D2+D1
    Quad1=(sig1*D1)**2
    QuadL=Quad1+(sig2*D2)**2
    mulL=0
    mul1=0
    for i in range(maxdist):
        if mul1==1:
           mul1=mul1-2
           mulL=mulL+1
        else:
           mul1=mul1+1
        Di=mul1*D1+mulL*L
        sigg=(mul1*Quad1+mulL*QuadL)**0.5/Di
        Fs=funcadd(s,Fs,sigg,sigH,Di,W*multis[i])
    return Fs
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def G1s_uno(s,W,D1,sigH,sig1):
    """
    Interference function of solos.
    """
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    #Build initial Fs
    Fs=np.zeros(len(s),dtype=np.float)
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    Fs=funcadd(s,Fs,sig1,sigH,D1,W)
    return Fs
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def set_multiplier(Mit):
    """
    Set the multiplier which is applied in the weight factor.
    """
    Md=1+3*(Mit-1) #maxdist
    multis=np.arange(Md,dtype=np.float)
    C = Mit
    for i in range(1,Md+1):
        if i%3 == 0:
           multis[i-1]=-2*C
        else:
           multis[i-1]=C
        if (i-1)%3 == 0:
           C=C-1
    #normalize
    multis=multis/float(Mit)
    return multis
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def funcadd(s,Fs,sigg,sigH,Di,W):
    """
    Add the superimposed function to the function.
    W is the weighting factor of the first peak. It's equal to the half area
    of the first peak.
    sigg and sigH are the heterogeneity of domains.
    Di is the size of i-th domain thickness
    """
    Di_s=Di*s
    Fs=Fs+W*GAF_sup_s(Di_s,sigg,sigH)
    return Fs
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def GAF_sup_s(s,sigg,sigH):
    """
    F(s) is the superposition unit.
    F(s=0)=2
    s is the scattering vector.
    sigG and sigH are the deviation or heterogeneity of the ensemble of stacks.
    It's relative standard deviation of i-th distance distribution,sigh=sigH,
    sigg=sigi/di
    """
    AT=ATs(s,sigg,sigH)
    BT=BTs(s,sigg,sigH)
    Fs=2.0*np.sqrt(AT)*np.exp(-1.0*AT*BT)*np.cos(2*np.pi*AT*s)
    return Fs
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def ATs(s,sigg,sigH):
    """
    A(s) is one term composed of F(s).
    """
    AT=1.0/(1.0+np.square(2.0*np.pi*sigg*sigH*s))
    return AT
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def BTs(s,sigg,sigH):
    """
    B(s) is another term composed of F(s).
    """
    BT=2.0*np.pi**2*(sigg**2+sigH**2)*np.square(s)
    return BT
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
