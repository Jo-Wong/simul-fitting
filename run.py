import argparse
from functions import *
from print_results import *

# import parameters
parser = argparse.ArgumentParser(description='Program to run simultaneous fitting for PWN polarization')
parser.add_argument('--source', type=str, help='Source name, must match the folder: source_<name>', required=True)
parser.add_argument('--obsids', type=str, nargs='+', help='List of IXPE ObsIDs to include in the fitting', required=True)
parser.add_argument('--nsim', type=int, help='Number of simulation files', required=True)
parser.add_argument('--fcut', type=float, default=None, help='Number of counts, flux cut for nebula polarization map')
parser.add_argument('--scut', type=float, default=None, help='Significance cut for nebula polarization map')
parser.add_argument('--outfile', type=str, help='name of output data file', required=True)

args = parser.parse_args()

name = args.source
exec(open(f'source_{name}/init.py', 'r').read())

obsids = args.obsids
nsim = args.nsim
fcut = args.fcut
scut = args.scut
outfile = args.outfile

# initialize constants
dfiles, pfiles, nfiles, dlivetimes, slivetimes, scale, pscale, nscale = list(Constants.get_package(obsids, det_scale, psr_scale, neb_scale).values())

# read the parameter file
param_dict = Constants.read_parameters(f'source_{name}/par/simulfit.par', obsids)

# generate data cube
dcube = generate_cube(dfiles, Constants.HOME_FOLDER, param_dict, 'DATA')

# generate simulation cubes
pcube = long_simulation(Constants.PSIM_FOLDER, param_dict, nsim, pfiles, use_proxy_weights=None)#'auxil/wmom_esorted_Crab_12.npy') 
ncube = long_simulation(Constants.NSIM_FOLDER, param_dict, nsim, nfiles, use_proxy_weights=None)#'auxil/wmom_esorted_Crab_12.npy')

# reference lightcurve
lc_x, lc = generate_lc(dfiles, Constants.HOME_FOLDER, param_dict, nbins=100, kernel=np.ones(3)/3)

# calculate the livetime ratio
det = np.array(param_dict['DETECTORS']) - 1
ratio = np.array(dlivetimes)[:,det] / (np.array(slivetimes)[:,det] * nsim) * np.array(scale)[:,det]

# apply any normalizations
for label in CUBE_LABELS:
   if label == 'W2':
      pcube[label] = pcube[label] * ratio[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]**2 * pscale[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]**2
      ncube[label] = ncube[label] * ratio[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]**2 * nscale[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]**2
   else:
      pcube[label] = pcube[label] * ratio[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] * pscale[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
      ncube[label] = ncube[label] * ratio[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] * nscale[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]

# perform simultaneous fit
# note: using I instead of W for consistent results between runs
norm_q, norm_u, norm_qerr, norm_uerr, A = simul(dcube, pcube['I'], ncube['I'], param_dict)
np.save(f'source_{name}/results/{outfile}.npy', [norm_q, norm_u, norm_qerr, norm_uerr], allow_pickle=True)

# save results
#fnames = ['pre-glitch_01001099.npy', 'pre-glitch_02001099.npy', 'pre-glitch_02006001.npy', 'post-glitch_sep.npy', 'post-glitch_oct.npy']
#labels = ['01001099', '02001099', '02006001', '04001299 SEP', '04001299 OCT']
fnames = obsids; labels = obsids
#pulsar_polarization(fnames, labels, lc_x, lc, param_dict, name, f'pol_{obsids[0]}_16')
pulsar_polarization([outfile], labels, lc_x, lc, param_dict, name, outfile)
#nebula_polarization_map(f'source_{name}/results/{outfile}.npy', name, param_dict, dcube['I'], outfile, fcut, scut)
nebula_polarization_int(f'source_{name}/results/{outfile}.npy', name, param_dict, A)
