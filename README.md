# ipyChord
The code for processing the 2D saxs pattern.
This package is open-sourced for the researchers who are interested in the advanced (at least now) algorithm 
in extracting more structural information from the 2D anisotropic saxs pattern of polymer materials with fiber
symmetry. The main procedures of this package:
1. Read the raw 2D SAXS pattern and harmonize the SAXS pattern. 
2. Transform the fully constructed pattern to the interference function. 
3. Calculate the CDF via negative Fourier transform of the interference function.

The chord distribution function (CDF) can be obtained via the above-mentioned procedures. 

If the reader is interested in the concept of the CDF, the reader can refer to the paper published by Stribeck
on the Journal of Applied Crystallography 2001, 34, 4, P496-503 and the textbook, X-ray Scattering of Soft Matter, 
written by Stribeck. 

The details are described in the Wiki page.
