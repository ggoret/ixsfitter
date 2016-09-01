# ixsfitter
IXS Fitter is the fastest solution to fit rapidly a lot of IXS spectra. 

IXS Fitter is a Semi-automatized IXS data treatment package made of two python modules :

* A data extractor called spec2hdf5 which extracts and computes energy transfer and normalized intensity of a given subset of scans and detectors from spec file.

* A semi-automated fitter module based on least square minimization, a Levenberg-Marquardt algorithm, is associated with a small GUI for inputs.
* This fitting module computes (in reciprocal space) the best function, which fits IXS spectrum (a convolution of a linear combination of Lorentzian and a pseudo-Voigt).

Homepage: https://forge.epn-campus.eu/projects/ixsfitter
Licence: GPL
