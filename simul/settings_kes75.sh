#!/bin/bash

# Can choose to do a simulation with PSF ('PSF_blur') or without PSF ('no_blur')
export SIM_CHOICE='PSF_blur'

# append _1, _2, etc. if an OBSID has multiple spacecraft rolls
export OBS_IDS=(03001901)
export ROLLS=(-144.4088)
export PAR_FILES=(kes75_03001901_nn.par)
#export OBS_IDS=(04002301)
#export ROLLS=(-332.062)
#export PAR_FILES=(kes75_04002301.par)
export IRF_NAMES=('ixpe:obssim20240701:v013')
export CYCLES=(23)
export INSTRS=(s)

# Other configuration parameters
export obj='pulsar_03001901'
export START_SIM=1
export NUM_SIM=10
export DURATION='500000'
export CONFIG_FILE="kes75_${obj}.py"
export OUTPUT_BASE='kes75_complex'
export OUTPUT_DIR="kes75/simulations/sky/s0.10/unweighted_500ks_${obj%%_*}_irf13"

# pileup effects are negligible, according to <pileup_map>, it's less than 2.5% in the nebula
export BLUR=0.10
export PILEUP=False
