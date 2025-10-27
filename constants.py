import os
import numpy as np

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

   def read_parameters(self, parfile, obsids):
      ''' populate param_dict with contents in <parfile> '''

      param_dict = dict.fromkeys(PAR_KEYS)
      param_dict['RA'] = self.RA
      param_dict['DEC'] = self.DEC
      param_dict['RESP_NAME'] = self.get_obsids(obsids, 'RESPONSES')

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
          files = [f'{obsid}/event_l2/{self.DFILES[obsid] % x}' for x in DETECTORS]
          output.append(files)

        if dtype == 'SFILES':
           files = [f'{obsid}/{self.SFILES[obsid] % x}' for x in DETECTORS]
           output.append(files)

        if dtype == 'RESPONSES': 
           files = [self.RESPONSES[obsid] % (x, '%s') for x in DETECTORS]
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
 
class Constants_Crab(Constants):

   def __init__(self):
      RA = 83.63275
      DEC = 22.01425

      HOME_FOLDER = os.environ['SCRATCH_FOLDER'] + 'crab/'
      PSIM_FOLDER = HOME_FOLDER + 'simulations/sky/s0.05/unweighted_150ks_pulsar_irf13_%d/'
      NSIM_FOLDER = HOME_FOLDER + 'simulations/sky/s0.05/unweighted_150ks_nebula_irf13_%d/'

      SIM_FILES = 'crab_complex_du%d_folded.fits'
      SIM_LIVETIMES = [150000, 150000, 150000]

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
 
      DFILES = {'01001099_1': _01001099_1_DFILE, 
                '01001099_2': _01001099_2_DFILE, 
                '02001099_1': _02001099_1_DFILE, 
                '02001099_2': _02001099_2_DFILE, 
                  '02006001': _02006001_DFILE, 
                  '04001299': _04001299_DFILE}

      DLIVETIMES = {'01001099_1': _01001099_1_LIVETIMES, 
                   '01001099_2': _01001099_2_LIVETIMES, 
                   '02001099_1': _02001099_1_LIVETIMES, 
                   '02001099_2': _02001099_2_LIVETIMES, 
                     '02006001': _02006001_LIVETIMES, 
                     '04001299': _04001299_LIVETIMES}

      RESPONSES = {'01001099_1': _01001099_1_RESPONSE, 
                   '01001099_2': _01001099_2_RESPONSE, 
                   '02001099_1': _02001099_1_RESPONSE, 
                   '02001099_2': _02001099_2_RESPONSE, 
                     '02006001': _02006001_RESPONSE, 
                     '04001299': _04001299_RESPONSE}

      SFILES = {'01001099_1': SIM_FILES, 
                '01001099_2': SIM_FILES, 
                '02001099_1': SIM_FILES, 
                '02001099_2': SIM_FILES, 
                  '02006001': SIM_FILES,  
                  '04001299': SIM_FILES}

      SLIVETIMES = {'01001099_1': SIM_LIVETIMES, 
                    '01001099_2': SIM_LIVETIMES, 
                    '02001099_1': SIM_LIVETIMES, 
                    '02001099_2': SIM_LIVETIMES, 
                      '02006001': SIM_LIVETIMES, 
                      '04001299': SIM_LIVETIMES}

      super().__init__(HOME_FOLDER, PSIM_FOLDER, NSIM_FOLDER, RA, DEC, DFILES, DLIVETIMES, SFILES, SLIVETIMES, RESPONSES)
