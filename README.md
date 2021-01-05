# ipyChord
The code for processing the 2D saxs pattern.
This package is open-sourced for the researchers who are interested in the advanced (at least now) algorithm 
in extracting more structural information from the 2D anisotropic saxs pattern of polymer materials with fiber
symmetry. 
The main functionalities of this package:
1. Read the raw 2D SAXS pattern and harmonize the SAXS pattern. Reading the raw pattern is referring to the
   package 'Harmonize' means the construction of a full 2D SAXS pattern which has no invalid pixels from small
   scattering vector to large scattering vector. Thus, we have to fill the intensity inside the beamstop region
   and extrapolate the intensity of apron for SAXS pattern until I(s)=0. 
2. Transform the fully constructed pattern to the interference function. 
3. Negative Fourier transform of the interference function.
The chord distribution function (CDF) can be obtained.
If the reader is interested in the concept of the CDF, the reader can refer to the paper published by Stribeck
on Journal of Applied Crystallography 2001, 34, 4, P496-503. 

The deatled procedures are described in the tutorial.pdf
