## Simultaneous Fitting for IXPE PWNe

This is a library package for fitting IXPE PWN pulsar/nebula polarization. See [this paper](https://ui.adsabs.harvard.edu/abs/2023ApJ...953...28W/abstract) for a description of the simultaneous fitting method and [this paper](https://ui.adsabs.harvard.edu/abs/2024ApJ...973..172W/abstract) for its application to the first three Crab observations. Be sure to download the [IXPEobssim software package](https://github.com/lucabaldini/ixpeobssim) as this package relies on it.

The directory is organized as follows:
```
simul-fitting/
├─ constants.py
├─ functions.py
├─ run.py
├─ run.sh
├─ run_cal.py
├─ run_cal.sh
├─ cal_functions.py
├─ print_results.py
├─ source_<name>/
│  ├─ init.py
│  ├─ results
│  ├─ plots
│  ├─ par
│  │  ├─ lc.par
│  │  ├─ grid.par
│  │  ├─ simulfit.par
├─ simul
│  ├─ generate_simulation.sh
│  ├─ settings_<name>.sh
```
<ins>run.py, run.sh</ins>
Runs simultaneous fitting. Initializes the binning parameters from `source_<name>/par/simulfit.par`. Applies binning to the IXPE data and the IXPEobssim pulsar and nebula simulations. Runs the least-squares minimization algorithm to solve for the pulsar phase-resolved and nebula spatially-varying polarization

<ins>run\_cal.py, run\_cal.sh</ins>
Calibrates the simulated IXPE observations. Generates a lightcurve and/or a count map for each detector (or set of detectors) from the data and the simulation and stores it in `source_<name>/plots`. Calls `source_<name>/par/lc.par` or `source_<name>/par/grid.par`

<ins>constants.py</ins>
Contains the simulation and observation files for the specific source that's being studied as well as other useful details (exposure time, response file, etc). Will be used by `run.py` to find the simulation and observation data

<ins>functions.py, cal\_functions.py, print\_results.py</ins>
Contains the functions used in `run.py` and `run_cal.py`. Should not be modified!

<ins>source\_\<name\></ins>
Directory containing `init.py` (initializes the source and calibration factors) and the folders containing the output results for a specific source called `<name>`.

<ins>simul</ins>
Contains the script `generate_simulation.sh` used to generate simulations of IXPE observations using IXPEobssim. To use, call `bash generate_simulation.sh <name>`. Store all simulation settings in `settings_<name>.sh`
