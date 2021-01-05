#/usr/bin/env python
import numpy as np
from sf_show import sf_show
from copy import deepcopy
from CDF_2D import flipharmony,pklwrite
def nicepos(mask,validr=[250,600],step=[25,25]):
    """
    This function is used to determine the nice position for the X-ray beam
    on the PILATUS detector with blind bars.
    This is determined by the mirror operation. For the SAXS pattern, normally
    the pattern should be 4-quadrant symmetric.
    algorithm:if the beam is not at the central part, we need to construct all
    the remant part to get a full pattern.The constructed part will be set as 0
    and the mirror operation will reset the data of these parts.
    """
    #check the pilmask,it should be the set all the valid pixel value as 1
    #and blind pixel value as zero.
    #In principle, the beam position should be in the detector for the saxs.
    #herein, we use (x,y) stands for the beam position.
    #the (0,0) point is in the top-left-most corner
    #i,j=0,0
    #write good positions into an ascii file
    fidw=open('mirror1m.dat','w')
    fidw.write('#step,[25,25]\n')
    pilmask=deepcopy(mask)
    #determine the size of pilmask
    pily,pilx=mask['map'].shape
    #get the position in the loop
    for j in np.arange(validr[0],validr[1],step[0]):
        for i in np.arange(validr[0],validr[1],step[1]):
            #determine the size of array constructed
            print 'position of beam(x,y) ',j,' ',i
            tmpmask=np.zeros((pily*2,pilx*2),dtype=np.int)
            #print tmpmask.shape
            #put pilmask into the center of tmpmask
            tmpmask[pily-i:pily-i+pily,pilx-j:pilx-j+pilx]=mask['map']
            pilmask['map']=tmpmask
            res=flipharmony(pilmask)
            #set the filename of res
            res['filename']='BeamPos1M'+'_'+str(j)+'_'+str(i)
            #show the image
            sf_show(res,win=1,auto=1)
            yn=raw_input('Input y or n to save pos: ')
            if yn == 'y':
               mmax=pilmask['map'].max()
               msum=np.sum(pilmask['map'])
               print 'Mask: max, ',mmax,' sum, ',msum
               fidw.write(str(j)+' '+str(i)+'\n')
               sf_show(res,win=1,auto=1,svg=1)
               pklwrite(res,res['filename'])
    fidw.close()
