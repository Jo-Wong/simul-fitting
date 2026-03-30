#!/bin/bash
# Script for generating IXPE simulations

BIN_FOLDER=$IXPEOBSSIM_PATH/ixpeobssim/bin
CONFIG_FOLDER=$IXPEOBSSIM_PATH/ixpeobssim/config

# Input: ${1} = Source Name
source settings_${1}.sh

for i in $(seq 0 $((${#OBS_IDS[@]}-1)));
do
    # select appropriate parameters for the observation
    OBS_ID=${OBS_IDS[$i]}
    ROLL=${ROLLS[$i]}
    PAR_FILE=${PAR_FILES[$i]}
    IRF_NAME=${IRF_NAMES[$i]}
    CYCLE=${CYCLES[$i]}
    INSTR=${INSTRS[$i]}
    
    # export the ephemeris file for usage by <_pulsar.py>
    export PAR_FILE=$PAR_FILE

    # generate the cxo effective area file
    python3 generate_arf.py ${CYCLE} ${INSTR}
 
    # assign flag whether to apply PSF blurring
    if [ "$SIM_CHOICE" = "no_blur" ]; then
        APPLY_SPATIAL=False
    else
        APPLY_SPATIAL=True
    fi
    
    ROTATION=$(echo "-1 * $ROLL" | bc -l)
 
    for j in `seq $START_SIM $((START_SIM + NUM_SIM - 1))`;
        do
            
            mkdir $SCRATCH_FOLDER/${OUTPUT_DIR}_${j}
            mkdir $SCRATCH_FOLDER/${OUTPUT_DIR}_${j}/${OBS_ID}
            
            ### Generate event list ###
            python3 $BIN_FOLDER/xpobssim.py --outfile $SCRATCH_FOLDER/${OUTPUT_DIR}_${j}/${OBS_ID}/${OUTPUT_BASE} --configfile $CONFIG_FOLDER/$CONFIG_FILE --irfname $IRF_NAME --duration $DURATION --vignetting ${APPLY_SPATIAL} --dithering ${APPLY_SPATIAL} --roll ${ROLL} --apply_psf ${APPLY_SPATIAL} --rotation ${ROTATION} --blur ${BLUR} --deadtime 0.0 --pileup_corr ${PILEUP} --overwrite True
            python3 $BIN_FOLDER/xpphase.py $SCRATCH_FOLDER/${OUTPUT_DIR}_${j}/${OBS_ID}/${OUTPUT_BASE}_du1.fits --parfile $CONFIG_FOLDER/par/$PAR_FILE --suffix folded
            python3 $BIN_FOLDER/xpphase.py $SCRATCH_FOLDER/${OUTPUT_DIR}_${j}/${OBS_ID}/${OUTPUT_BASE}_du2.fits --parfile $CONFIG_FOLDER/par/$PAR_FILE --suffix folded
            python3 $BIN_FOLDER/xpphase.py $SCRATCH_FOLDER/${OUTPUT_DIR}_${j}/${OBS_ID}/${OUTPUT_BASE}_du3.fits --parfile $CONFIG_FOLDER/par/$PAR_FILE --suffix folded
        done
    
done
