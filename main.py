from functions import *
from constants import Constants_Crab

# initialize the constants
obsids = ['04001299']
Constants = Constants_Crab()

# read the parameter file
param_dict = Constants.read_parameters('par/20eqph_2-8keV_13x13_15as.par', obsids)
dfiles = Constants.get_obsids(obsids, 'DFILES')
pfiles = Constants.get_obsids(obsids, 'SFILES')
nfiles = Constants.get_obsids(obsids, 'SFILES')
dlivetimes = Constants.get_obsids(obsids, 'DLIVETIMES')
slivetimes = Constants.get_obsids(obsids, 'SLIVETIMES')

# generate data cube
dcube = generate_cube(dfiles, Constants.HOME_FOLDER, param_dict, 'DATA')

# generate simulation cubes
num_sim = 1
pcube = long_simulation(Constants.PSIM_FOLDER, param_dict, num_sim, pfiles, use_proxy_weights='auxil/wmom_esorted_Crab_12.npy') 
ncube = long_simulation(Constants.NSIM_FOLDER, param_dict, num_sim, nfiles, use_proxy_weights='auxil/wmom_esorted_Crab_12.npy')

# calculate the livetime ratio
det = np.array(param_dict['DETECTORS']) - 1
ratio = dlivetimes[:,det] / (slivetimes[:,det] * num_sim)

# apply any normalizations
for label in CUBE_LABELS:
   if label == 'W2':
      pcube[label] = pcube[label] * ratio[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]**2
      ncube[label] = ncube[label] * ratio[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]**2
   else:
      pcube[label] = pcube[label] * ratio[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
      ncube[label] = ncube[label] * ratio[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]

# perform simultaneous fit
norm_q, norm_u, norm_qerr, norm_uerr, A = simul(dcube, pcube['W'], ncube['W'], param_dict)
pd, pderr, pa, paerr, sig = find_pol(norm_q, norm_qerr, norm_u, norm_uerr)
