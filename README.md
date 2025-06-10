# ipyChord
Code Description
This open-source package is designed for researchers interested in advanced (currently state-of-the-art) algorithms to extract structural information from 2D anisotropic SAXS patterns of polymer materials with fiber symmetry. The primary workflow includes:
1. Reading and harmonizing raw 2D SAXS patterns.
2. Transforming the fully constructed pattern into the interference function.
3. Calculating the chord distribution function (CDF) via the negative Fourier transform of the interference function.

The chord distribution function (CDF) can be obtained via the above-mentioned procedures. 

Reference. https://onlinelibrary.wiley.com/iucr/doi/10.1107/S1600576721001369

Li, X. (2021). ipyChord : a package for evaluating small-angle X-ray scattering data of fiber symmetry. Journal of Applied Crystallography, 54(2), 680–685. https://doi.org/10.1107/S1600576721001369

If the reader is interested in the concept of the CDF, the reader can refer to the paper published by Stribeck
on the Journal of Applied Crystallography 2001, 34, 4, P496-503 and the textbook “X-ray Scattering of Soft Matter“ (https://www.springer.com/gp/book/9783540698555) written by Stribeck. 

The details are described in the Wiki page. 

What I must point out is that many code in this package are originated from the code developed by Stribeck. Many functions are translated 
from the functions programmed in PV-WAVE. I would like to express my sincerest gratitude to her. If the user are interested in the original
code developed by her, you can visit her webpage: http://www.stribeck.de/almut_e.html and contact with her. 
