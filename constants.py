import os
import numpy as np
import warnings
from colorama import Fore, Style, init; init()

DETECTORS = [1,2,3]
PAR_KEYS = ['RA', 'DEC', 'PHASE_BINS', 'ENERGY_BINS', 'SPATIAL_BIN_T', 'SPATIAL_BIN_P', 'WEIGHTS', 'RESP_NAME', 'DETECTORS', 'ROT_PH']

class Constants:

   def __init__(self, HOME_FOLDER, PSIM_FOLDER, NSIM_FOLDER, RA, DEC, DFILES, DLIVETIMES, SFILES, SLIVETIMES, RESPONSES):
    
      self.HOME_FOLDER = HOME_FOLDER
      self.PSIM_FOLDER = PSIM_FOLDER
      self.NSIM_FOLDER = NSIM_FOLDER

      self.RA = RA
      self.DEC = DEC
      self.DFILES = DFILES
      self.DLIVETIMES = DLIVETIMES
      self.SFILES = SFILES
      self.SLIVETIMES = SLIVETIMES
      self.RESPONSES = RESPONSES

      for key in self.DLIVETIMES.keys():
         self.DLIVETIMES[key] = np.array(self.DLIVETIMES[key])

      for key in self.SLIVETIMES.keys():
         self.SLIVETIMES[key] = np.array(self.SLIVETIMES[key])

   def read_parameters(self, parfile, obsids=None):
      ''' populate param_dict with contents in <parfile> '''
      param_dict = dict.fromkeys(PAR_KEYS)
      param_dict['RA'] = self.RA
      param_dict['DEC'] = self.DEC
      if obsids is not None:
         param_dict['RESP_NAME'] = self.get_obsids(obsids, 'RESPONSES')
      else:
         del param_dict['RESP_NAME']
         warnings.warn(Fore.RED + "no obsids provided - no RESP_NAME will be defined" + Style.RESET_ALL, category=None, stacklevel=1)

      with open(parfile) as file:
         for line in file.readlines():
            if not line.startswith('#'):
               result = line.strip().split()

               if result[0] == 'SPATIAL_BIN_P':
                  if param_dict[result[0]] is None:
                     param_dict[result[0]] = dict()

                  param_dict[result[0]][eval(result[1])] = eval(result[2])
               else:
                  param_dict[result[0]] = eval(''.join(result[1:]))

      assert np.logical_not(np.any([val is None for val in param_dict.values()]))
      return param_dict

   def get_obsids(self, obsids, dtype, input_dictionary=None):
      ''' collect all the data and simulation files that will be used in the analysis '''

      output = []

      for obsid in obsids:

        if dtype == 'DFILES':
          if isinstance(self.DFILES[obsid], str):
             files = [f'{obsid}/event_nn/{self.DFILES[obsid] % x}' for x in DETECTORS]
             output.append(files)
          elif isinstance(self.DFILES[obsid], list):
             files = [f'{obsid}/event_nn/{x}' for x in self.DFILES[obsid]]
             output.append(files)
          else:
             raise ValueError(f'invalid DFILES value for {obsid}')

        if dtype == 'SFILES':
           files = [f'{obsid}/{self.SFILES[obsid] % x}' for x in DETECTORS]
           output.append(files)

        if dtype == 'RESPONSES': 
           files = [self.RESPONSES[obsid].replace('%d', str(x)) for x in DETECTORS]
           output.append(files)

        if dtype == 'DLIVETIMES':
           output.append(self.DLIVETIMES[obsid])
 
        if dtype == 'SLIVETIMES':
           output.append(self.SLIVETIMES[obsid])

        if dtype == 'OTHER':
           assert(input_dictionary is not None)
           output.append(input_dictionary[obsid])
    
      output = np.array(output)
      return output

   def get_package(self, obsids, det_scale, psr_scale, neb_scale):
      ''' 
      get some of the common variables required for simultaneous fitting 
      '''
      dfiles = self.get_obsids(obsids, 'DFILES')
      pfiles = self.get_obsids(obsids, 'SFILES')
      nfiles = self.get_obsids(obsids, 'SFILES')
      dlivetimes = self.get_obsids(obsids, 'DLIVETIMES')
      slivetimes = self.get_obsids(obsids, 'SLIVETIMES')
      scale = self.get_obsids(obsids, 'OTHER', det_scale)
      pscale = self.get_obsids(obsids, 'OTHER', psr_scale)
      nscale = self.get_obsids(obsids, 'OTHER', neb_scale)

      return {'DFILES': dfiles, 'PFILES': pfiles, 'NFILES': nfiles, 'DLIVETIMES': dlivetimes, 'SLIVETIMES': slivetimes, \
                 'SCALE': scale, 'PSCALE': pscale, 'NSCALE': nscale}

class Constants_Crab(Constants):

   def __init__(self):
      RA = 83.63275
      DEC = 22.01425

      HOME_FOLDER = os.environ['SCRATCH_FOLDER'] + 'crab/'
      PSIM_FOLDER = HOME_FOLDER + 'simulations/sky/s0.10/unweighted_75ks_pulsar_irf13_%d/'
      NSIM_FOLDER = HOME_FOLDER + 'simulations/sky/s0.10/unweighted_75ks_nebula_irf13_%d/'

      SIM_FILES = 'crab_complex_du%d_folded.fits'
      SIM_LIVETIMES = [75000, 75000, 75000]

      _01001099_1_DFILE = 'ixpe01001099_det%d_evt2_v01_aspectcorr_picorr_wcscorr_barycorr_tlcorr_foldcorr013_1_shift_matched.fits'
      _01001099_1_RESPONSE = 'ixpe_d%d_obssim20211209_offaxis274_r0_%sv013'
      _01001099_1_LIVETIMES = [43068.83279469609, 43069.4537205100, 43123.22644600272]

      _01001099_2_DFILE = 'ixpe01001099_det%d_evt2_v01_aspectcorr_picorr_wcscorr_barycorr_tlcorr_foldcorr013_2_shift_matched.fits'
      _01001099_2_RESPONSE = 'ixpe_d%d_obssim20211209_offaxis274_r0_%sv013'
      _01001099_2_LIVETIMES = [49285.71425133944, 49322.4854850471, 49321.59947443008]

      _02001099_1_DFILE = 'ixpe02001099_det%d_evt2_v02_bkgrej_wcscorr_gticorr_barycorr_folded_radio_1_shift_matched.fits'
      _02001099_1_RESPONSE = 'ixpe_d%d_obssim20230101_att_%sv013'
      _02001099_1_LIVETIMES = [74159.21030688286, 74148.48080211878, 74130.44447928667]

      _02001099_2_DFILE = 'ixpe02001099_det%d_evt2_v02_bkgrej_wcscorr_gticorr_barycorr_folded_radio_2_shift_matched.fits'
      _02001099_2_RESPONSE = 'ixpe_d%d_obssim20230101_att_%sv013'
      _02001099_2_LIVETIMES = [74212.52787867188, 74248.07193231583, 74216.40554663539]

      _02006001_DFILE = 'ixpe02006001_det%d_evt2_v01_bkgrej_wcscorr_gticorr_barycorr_folded_shift_matched.fits'
      _02006001_RESPONSE = 'ixpe_d%d_obssim20230702_att_%sv013'
      _02006001_LIVETIMES = [60201.430184304714, 60209.1314752996, 60205.203489899635]

      _04001299_DFILE = 'ixpe04001299_det%d_evt2_v01_rej_gti_wcs_bary_folded_shifted.fits'
      _04001299_RESPONSE = 'ixpe_d%d_obssim20240701_%sv013'
      _04001299_LIVETIMES = [135346.692827, np.nan, 136668.585705]

      _04001299_SEP_DFILE = 'ixpe04001299_det%d_evt2_v01_rej_gti_wcs_bary_folded_sep_shifted.fits'
      _04001299_SEP_RESPONSE = 'ixpe_d%d_obssim20240701_%sv013'
      _04001299_SEP_LIVETIMES = [68828.97280792237, np.nan, 69499.58626255709]

      _04001299_OCT_DFILE = 'ixpe04001299_det%d_evt2_v01_rej_gti_wcs_bary_folded_oct_shifted.fits'
      _04001299_OCT_RESPONSE = 'ixpe_d%d_obssim20240701_%sv013'
      _04001299_OCT_LIVETIMES = [66524.33075007730, np.nan, 67173.58739883185]
 
      DFILES = {'01001099_1': _01001099_1_DFILE, 
                '01001099_2': _01001099_2_DFILE, 
                '02001099_1': _02001099_1_DFILE, 
                '02001099_2': _02001099_2_DFILE, 
                  '02006001': _02006001_DFILE, 
                  '04001299': _04001299_DFILE,
              '04001299_SEP': _04001299_SEP_DFILE,
              '04001299_OCT': _04001299_OCT_DFILE}

      DLIVETIMES = {'01001099_1': _01001099_1_LIVETIMES, 
                   '01001099_2': _01001099_2_LIVETIMES, 
                   '02001099_1': _02001099_1_LIVETIMES, 
                   '02001099_2': _02001099_2_LIVETIMES, 
                     '02006001': _02006001_LIVETIMES, 
                     '04001299': _04001299_LIVETIMES,
                 '04001299_SEP': _04001299_SEP_LIVETIMES,
                 '04001299_OCT': _04001299_OCT_LIVETIMES}

      RESPONSES = {'01001099_1': _01001099_1_RESPONSE, 
                   '01001099_2': _01001099_2_RESPONSE, 
                   '02001099_1': _02001099_1_RESPONSE, 
                   '02001099_2': _02001099_2_RESPONSE, 
                     '02006001': _02006001_RESPONSE, 
                     '04001299': _04001299_RESPONSE,
                 '04001299_SEP': _04001299_SEP_RESPONSE,
                 '04001299_OCT': _04001299_OCT_RESPONSE}

      SFILES = {'01001099_1': SIM_FILES, 
                '01001099_2': SIM_FILES, 
                '02001099_1': SIM_FILES, 
                '02001099_2': SIM_FILES, 
                  '02006001': SIM_FILES,  
                  '04001299': SIM_FILES,
              '04001299_SEP': SIM_FILES,
              '04001299_OCT': SIM_FILES}

      SLIVETIMES = {'01001099_1': SIM_LIVETIMES, 
                    '01001099_2': SIM_LIVETIMES, 
                    '02001099_1': SIM_LIVETIMES, 
                    '02001099_2': SIM_LIVETIMES, 
                      '02006001': SIM_LIVETIMES, 
                      '04001299': SIM_LIVETIMES,
                  '04001299_SEP': SIM_LIVETIMES,
                  '04001299_OCT': SIM_LIVETIMES}

      super().__init__(HOME_FOLDER, PSIM_FOLDER, NSIM_FOLDER, RA, DEC, DFILES, DLIVETIMES, SFILES, SLIVETIMES, RESPONSES)

class Constants_Kes75(Constants):

   def __init__(self):
      RA = 281.603125
      DEC = -2.974888889

      HOME_FOLDER = os.environ['SCRATCH_FOLDER'] + 'kes75/'
      PSIM_FOLDER = HOME_FOLDER + 'simulations/sky/s0.10/unweighted_500ks_pulsar_irf13_%d/'
      NSIM_FOLDER = HOME_FOLDER + 'simulations/sky/s0.10/unweighted_500ks_nebula_irf13_%d/'

      SIM_FILES = 'kes75_complex_du%d_folded.fits'
      SIM_LIVETIMES = [500000, 500000, 500000]

      #_03001901_DFILE = ['ixpe03001901_det1_l2_boom_bkg_filter_rej_copy_clean_gti_wcs_bary_phase_shifted.fits', \
      #                   'ixpe03001901_det2_l2_boom_bkg_filter_fix_rej_copy_gti_wcs_bary_phase_shifted.fits', \
      #                   'ixpe03001901_det3_l2_boom_bkg_filter_rej_copy_gti_wcs_bary_phase_shifted.fits']
      #_03001901_RESPONSE = 'ixpe_d%d_obssim20240701_%sv013'
      #_03001901_LIVETIMES = [495800, 497334, 496606]

      _03001901_DFILE = ['ixpe03001901_det1_nn_boom_rej_clean_gti_wcs_deflare_bary_phase.fits', \
                         'ixpe03001901_det2_nn_boom_rej_gti_wcs_deflare_bary_phase.fits', \
                         'ixpe03001901_det3_nn_boom_rej_gti_wcs_deflare_bary_phase.fits']
      _03001901_RESPONSE = 'ixpe_d%d_obssim_nnw_v011'
      _03001901_LIVETIMES = [489663, 460148, 443860]

      #_04002301_DFILE = ['ixpe04002301_det1_l2_boom_bkg_filter_rej_copy_gti_wcs_bary_phase_shifted.fits', \
      #                   'ixpe04002301_det2_l2_boom_bkg_c1_filter_fix_rej_copy_gti_wcs_bary_phase.fits', \
      #                   'ixpe04002301_det3_l2_boom_bkg_filter_rej_copy_gti_wcs_bary_phase_shifted.fits']
      #_04002301_RESPONSE = 'ixpe_d%d_obssim20240701_%sv013'
      #_04002301_LIVETIMES = [483914, 295347, 483937]

      _04002301_DFILE = ['ixpe04002301_det1_nn_boom_rej_gti_wcs_deflare_bary_phase.fits', \
                         'ixpe04002301_det2_nn_boom_rej_gti_wcs_deflare_bary_phase.fits', \
                         'ixpe04002301_det3_nn_boom_rej_gti_wcs_deflare_bary_phase.fits']
      _04002301_RESPONSE = 'ixpe_d%d_obssim_nnw_v011'
      _04002301_LIVETIMES = [476426, 240607, 430062]

      DFILES = {'03001901': _03001901_DFILE, 
                '04002301': _04002301_DFILE}

      DLIVETIMES = {'03001901': _03001901_LIVETIMES, 
                    '04002301': _04002301_LIVETIMES}

      RESPONSES = {'03001901': _03001901_RESPONSE, 
                   '04002301': _04002301_RESPONSE}

      SFILES = {'03001901': SIM_FILES, 
                '04002301': SIM_FILES}

      SLIVETIMES = {'03001901': SIM_LIVETIMES, 
                    '04002301': SIM_LIVETIMES}

      super().__init__(HOME_FOLDER, PSIM_FOLDER, NSIM_FOLDER, RA, DEC, DFILES, DLIVETIMES, SFILES, SLIVETIMES, RESPONSES)
