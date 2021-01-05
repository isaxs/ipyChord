#!/usr/bin/env python
import datetime
import numpy as np
def mydict(width=1024,height=1024,dist=100,wlen=1.54):
    """
    This function is used to create a Python dict to store all the relative
    parameters. Some domains are reserved forcibly.
    """
    #create an empty dict
    idict=dict()
    ####################################################################
    #some important information about the author who process the data
    idict['0_author']='Xuke Li'
    #idict['0_institue']='Polymer Physics, CIAC, CAS, China'
    idict['0_email']='lixuker@gmail.com'
    ####################################################################
    #the domain or key map is used to specify the 2d array of saxs or waxs
    idict['map']=np.zeros((height,width),dtype=np.float)
    #'absop' is used to present the absorption factor of the current state
    idict['absop']=0.0
    #'boxlen' is the pixel size of current array
    idict['boxlen']=[1.0,1.0]
    #'center' specifies the center of the 2d array. For example, if the 
    #2d array has [8,8] numbers the actual center should be [3.5,3.5]
    #this is the special paradigm in the saxs
    idict['center']=[width/2,height/2]
    #the datetime of exposure in formart datetime
    idict['datetime']=datetime.time()
    #'distance' specifies the distance from sample to detector
    idict['distance']=dist
    #'expot' is the exposure time for this frame
    idict['exptime']=0.0
    #'filename' is the name of this file, normally it should have extension
    idict['filename']='undefined.pkl'
    #'height' is the number of the row
    idict['height']=height
    #set the y coordinate of every pixel,it depends on the height of image
    ##########################
    #xy=np.indices((n,n))
    #x_coor,y_coor=xy[1]-n/2+0.5,xy[0]-n/2+0.5
    ##########################
    idict['y_coor']=0.
    #the intensity parameter of the beam current or the beam ion in count
    idict['ibeam']=0.0
    #'iexp' is the ion counts of the sample or experiment
    idict['iexpt']=0.0
    #'model' is the model of the detector type and the affiliation
    idict['model']='detector'
    #the position of sample
    idict['pos']=0.0
    #the pixel size
    idict['pix_size']=[172e-6,172e-6] #unit: pilatus \mum
    #'title' is the sery name to specify  series of frames
    idict['title']='title'
    #wavelength of the x-Ray
    idict['wavelength']=wlen
    #column of the 2d array
    idict['width']=width
    #set the x coordinate of every pixel,it depends on the width of image
    idict['x_coor']=0.
    #the temperature of the sample default 25 degree celsius room temperature
    idict['temp']=25.0
    #the strain of sample
    idict['clamp_dist']=0.0
    #stress the of sample
    idict['force']=0.0
    #Maybe True strain
    idict['strainT']=0.0
    #maybe True stress
    idict['stressT']=0.0
    #the dimension of the sample
    idict['dim']=0.0
    #the elapsed time for processing the sample
    idict['elapsedt']=0.0
    #maybe the comments
    idict['comment']='comments'
    #maybe pressure
    idict['pressure']=0.0
    #maybe units
    idict['unit']='nm^-1'
    #maybe material
    idict['matter']='None'
    #maybe beam size
    idict['beamsize']=[0.0,0.0]
    #maybe filter
    idict['filter']=0
    #maybe project
    idict['project']='None'
    #maybe beamline
    idict['beamline']='None'
    
    return idict
