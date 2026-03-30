#!/bin/bash

#python3 run.py --source crab --obsids 04001299_SEP --nsim 10 --fcut 10000 --scut 5 --outfile post-glitch_sep 
#python3 run.py --source crab --obsids 04001299_OCT --nsim 10 --fcut 10000 --scut 5 --outfile post-glitch_oct
#python3 run.py --source crab --obsids 04001299_SEP 04001299_OCT --nsim 10 --fcut 10000 --scut 5 --outfile post-glitch_merged 

python3 run.py --source kes75 --obsids 03001901 --nsim 10 --outfile 03001901
#python3 run.py --source kes75 --obsids 04002301 --nsim 10 --outfile 04002301
