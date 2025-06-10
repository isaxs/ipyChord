# ipyChord
# Code Description
This open-source package is designed for researchers interested in advanced (currently state-of-the-art) algorithms to extract structural information from 2D anisotropic SAXS patterns of polymer materials with fiber symmetry. The primary workflow includes:
1. Reading and harmonizing raw 2D SAXS patterns.
2. Transforming the fully constructed pattern into the interference function.
3. Calculating the chord distribution function (CDF) via the negative Fourier transform of the interference function.

# References
The methodology is detailed in the Wiki documentation and the following publications:

1. Li, X. (2021). ipyChord: A package for evaluating small-angle X-ray scattering data of fiber symmetry . Journal of Applied Crystallography , 54(2), 680–685. https://doi.org/10.1107/S1600576721001369
2. Stribeck’s foundational work on CDF theory (Journal of Applied Crystallography , 2001, 34(4), 496–503) and his textbook X-ray Scattering of Soft Matter (Springer, ISBN 9783540698555). 

If the reader is interested in the concept of the CDF, the reader can refer to the paper published by Stribeck on the Journal of Applied Crystallography 2001, 34, 4, P496-503 and the textbook “X-ray Scattering of Soft Matter“ (https://www.springer.com/gp/book/9783540698555) written by Stribeck. 

# Acknowledgments
Many functions in this package were adapted from code originally developed by Prof. Almut Stribeck, translated from PV-WAVE to Python. The author gratefully acknowledges her contributions. For access to her original code or further inquiries, please visit her personal webpage: http://www.stribeck.de/almut_e.html .
