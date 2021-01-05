# ipyChord
The code for processing the 2D saxs pattern.
This package is open-sourced for the researchers who are interested in the advanced (at least now) algorithm 
in extracting more structural information from the 2D anisotropic saxs pattern of polymer materials with fiber
symmetry. 
The main functionalities of this package:
1. Read the raw 2D SAXS pattern and harmonize the SAXS pattern. Reading the raw pattern is referring to the
   package 'Harmonize' means the construction of a full 2D SAXS pattern which has no invalid pixels from small
   scattering vector to large scattering vector. Thus, we have to fill the intensity inside the beamstop region
   and extrapolate the intensity of apron for SAXS pattern until I(s)=0. The reason for the full 2D SAXS pattern
   can refer to the physical significance of Fourier transform. The algorithm for filling the beamstop intensity is
   referring to the Guinier' law in ideal case. However, the Guinier's law is not always working in the complicate 
   case. Thus, in our code, we use the SG algorithm to fit the intensity besides the beamstop and extrapolate the 
   intensity into the beamstop using the same algorithm. 
2. 
If the reader is interested in the concept of the 
