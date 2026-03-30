#!/bin/bash

# Can choose to do a simulation with PSF ('PSF_blur') or without PSF ('no_blur')
SIM_CHOICE='PSF_blur'

# append _1, _2, etc. if an OBSID has multiple spacecraft rolls
#OBS_IDS=(04001299_SEP); ROLLS=(-338.42840789212823)
OBS_IDS=(04001299_OCT); ROLLS=(-338.82947834871980)
PAR_FILES=(ixpe04001299_det_evt2_v01_bary_fit.par)
IRF_NAMES=('ixpe:obssim20240701:v013')
CYCLES=(25)
INSTRS=(s)

#OBS_IDS=(01001099_1 01001099_2 02001099_1 02001099_2 02006001 03009601)
#ROLLS=(-157.957 -158.293 -157.977 -158.885 -339.026 -337.831)
#PAR_FILES=(crab_01001099.par crab_01001099.par crab_02001099.par crab_02001099.par crab_02006001.par crab_03009601.par)
#IRF_NAMES=('ixpe:obssim20211209_offaxis274_r0:v013' 'ixpe:obssim20211209_offaxis274_r0:v013' 'ixpe:obssim20230101_att:v013' 'ixpe:obssim20230101_att:v013' 'ixpe:obssim20230702_att:v013', 'ixpe:obssim20240701:v013')
#CYCLES=(23 23 23 23 23 23) 
#INSTRS=(s s s s s s)

# Other configuration parameters
obj='nebula'
START_SIM=1
NUM_SIM=10
DURATION='75000'
CONFIG_FILE="crab_${obj}.py"
OUTPUT_BASE='crab_complex'
OUTPUT_DIR="crab/simulations/sky/s0.10/unweighted_75ks_${obj}_irf13"

BLUR=0.10
PILEUP=True
