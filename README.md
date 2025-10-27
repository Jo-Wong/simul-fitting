## Simultaneous Fitting for IXPE PWNe

This is a library package for fitting pulsar/nebula polarization in IXPE data. See [this paper](https://ui.adsabs.harvard.edu/abs/2023ApJ...953...28W/abstract) for a description of the simultaneous fitting method and [this paper](https://ui.adsabs.harvard.edu/abs/2024ApJ...973..172W/abstract) for its application to the first three Crab observations.

The directory is organized as follows:

simul-fitting/

├─ constants.py

├─ functions.py

├─ main.py

├─ par/

│  ├─ 20eqph\_2-8keV\_13x13\_15as.par

<ins>main.py</ins>
Runs simultaneous fitting. Initializes the binning parameters. Applies binning to the IXPE data and the IXPEobssim pulsar and nebula simulations. Runs the least-squares minimization algorithm to solve for the pulsar phase-resolved and nebula spatially-varying polarization

<ins>constants.py</ins>
Contains important details about the simulation and observation files for the specific source that's being studied. Will be used by `main.py` to pull the simulation and observation data

<ins>functions.py</ins>
Library of functions used in `main.py` to run the fit. Should not be modified!

<ins>par</ins>
Contains the binning parameters. Create a new `.par` file for each individual binning scheme
