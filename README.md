# ipyChord
The code for processing the 2D saxs pattern.
This package is open-sourced for the researchers who are interested in the advanced (at least now) algorithm 
in extracting more structural information from the 2D anisotropic saxs pattern of polymer materials with fiber
symmetry. The main procedures of this package:
1. Read the raw 2D SAXS pattern and harmonize the SAXS pattern. 
2. Transform the fully constructed pattern to the interference function. 
3. Calculate the CDF via negative Fourier transform of the interference function.

The chord distribution function (CDF) can be obtained via the above-mentioned procedures. 

Reference. https://onlinelibrary.wiley.com/iucr/doi/10.1107/S1600576721001369
Li, X. (2021). ipyChord : a package for evaluating small-angle X-ray scattering data of fiber symmetry. Journal of Applied Crystallography, 54(2), 680–685. https://doi.org/10.1107/S1600576721001369

If the reader is interested in the concept of the CDF, the reader can refer to the paper published by Stribeck
on the Journal of Applied Crystallography 2001, 34, 4, P496-503 and the textbook “X-ray Scattering of Soft Matter“ (https://www.springer.com/gp/book/9783540698555) written by Stribeck. 

The details are described in the Wiki page. 

What I must point out is that many code in this package are originated from the code developed by Stribeck. Many functions are translated 
from the functions programmed in PV-WAVE. I would like to express my sincerest gratitude to her. If the user are interested in the original
code developed by her, you can visit her webpage: http://www.stribeck.de/almut_e.html and contact with her. 
