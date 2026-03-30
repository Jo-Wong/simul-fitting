import argparse
from functions import *
from cal_functions import *

if __name__ == "__main__":

   def calibrate(name, parname, num_sim, dmin, dmax, outfile):

      # read the parameter file
      param_dict = Constants.read_parameters(f'source_{name}/par/{parname}.par', obsids)

      # generate data cube
      #shift_center(shift_ra=1, shift_dec=2, obj=Constants, param_dict=param_dict)
      dcube = generate_cube(dfiles, Constants.HOME_FOLDER, param_dict, 'DATA')
      
      # generate simulation cubes
      #param_dict['RA'] = Constants.RA
      #param_dict['DEC'] = Constants.DEC
      pcube = long_simulation(Constants.PSIM_FOLDER, param_dict, num_sim, pfiles, use_proxy_weights=None)#'auxil/wmom_esorted_Crab_12.npy') 
      ncube = long_simulation(Constants.NSIM_FOLDER, param_dict, num_sim, nfiles, use_proxy_weights=None)#'auxil/wmom_esorted_Crab_12.npy')

      # calculate the livetime ratio
      det = np.array(param_dict['DETECTORS']) - 1
      ratio = np.array(dlivetimes)[:,det] / (np.array(slivetimes)[:,det] * num_sim) * np.array(scale)[:,det]

      # apply any normalizations
      for label in CUBE_LABELS:
         if label == 'W2':
            pcube[label] = pcube[label] * ratio[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]**2 * pscale**2
            ncube[label] = ncube[label] * ratio[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]**2 * nscale**2
         else:
            pcube[label] = pcube[label] * ratio[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] * pscale
            ncube[label] = ncube[label] * ratio[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] * nscale

      # calibrate simulations
      dcube['I'] = dcube['I'][:,dmin-1:dmax,:,:,:,:]
      pcube['I'] = pcube['I'][:,dmin-1:dmax,:,:,:,:]
      ncube['I'] = ncube['I'][:,dmin-1:dmax,:,:,:,:]
    
      if parname == 'grid': 
         create_comparison_map(dcube['I'], pcube['I'], ncube['I'], f'source_{name}/plots/{outfile}.png')
      elif parname == 'lc':
         compare_phase(dcube['I'], pcube['I'], ncube['I'], param_dict, f'source_{name}/plots/{outfile}.png')

   
   parser = argparse.ArgumentParser(description='Program to calibrate IXPEobssim simulations of Pulsar and Nebula')
   parser.add_argument('source', type=str, help='Source name, must match the folder: source_<name>')
   parser.add_argument('obsid', type=str, help='IXPE observation to calibrate')
   parser.add_argument('function', type=str, help='"lc" or "grid"')
   parser.add_argument('nsim', type=int, help='Number of simulation files')
   parser.add_argument('dmin', type=int, help='lower end of detector range')
   parser.add_argument('dmax', type=int, help='upper end of detector range')
   parser.add_argument('outfile', type=str, help='name of output file')

   args = parser.parse_args()

   name = args.source
   exec(open(f'source_{name}/init.py', 'r').read())

   obsids = [args.obsid]
   dfiles, pfiles, nfiles, dlivetimes, slivetimes, scale, pscale, nscale = list(Constants.get_package(obsids, det_scale, psr_scale, neb_scale).values())

   if args.function == 'lc':
      calibrate(name, 'lc', args.nsim, args.dmin, args.dmax, args.outfile)
   elif args.function == 'grid':
      calibrate(name, 'grid', args.nsim, args.dmin, args.dmax, args.outfile)
   else:
      raise ValueError('invalid value for <function>')
