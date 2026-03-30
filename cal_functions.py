import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from functions import midpoints, array_to_fits

def create_comparison_map(dcube, pcube, ncube, outfile):

   # create image maps
   obs_map = np.sum(dcube, axis=(0,1,2,3))
   sim_map = np.sum(pcube, axis=(0,1,2,3)) + np.sum(ncube, axis=(0,1,2,3))

   # find the center pixel
   nx, ny = obs_map.shape
   x = int((nx-1)/2) 
   y = int((ny-1)/2) 

   # plot comparison image
   fig, ax = plt.subplots(1,3)

   ax[0].imshow(np.log(obs_map[::-1,:]))
   ax[0].set_title('IXPE Observation')

   ax[1].imshow(np.log(sim_map[::-1,:]))
   ax[1].set_title('IXPEobssim Sim')

   im = ax[2].imshow( (sim_map[::-1,:] - obs_map[::-1,:]) / np.sqrt(sim_map[::-1,:]))
   ax[2].set_title(r'$\chi^2$ Difference')
   divider = make_axes_locatable(ax[2])
   cax = divider.append_axes('right', size='5%', pad=0.05)
   plt.colorbar(im, cax=cax)

   for axis in ax:
      axis.text(x,y,'x',ha='center',va='center',color='black')
   print(np.sum(((sim_map.T[::-1,:] - obs_map.T[::-1,:]) / np.sqrt(sim_map.T[::-1,:]))**2))
   plt.savefig(outfile, dpi=300, bbox_inches='tight')

def compare_phase(dcube, pcube, ncube, param_dict, outfile):

   data_lc = np.sum(dcube, axis=(0,1,3,4,5))
   modl_lc = np.sum(pcube + ncube, axis=(0,1,3,4,5))

   phases = midpoints(param_dict['PHASE_BINS'])

   # plot with model and peak phase
   fig, ax = plt.subplots()
   plt.step(phases,data_lc,where='mid',label='data (%.2f)' % phases[np.argmax(data_lc)])
   plt.step(phases,modl_lc,where='mid',label='model (%.2f)' % phases[np.argmax(modl_lc)])
   plt.legend()
   print(np.max(data_lc), np.max(modl_lc))
   print( np.sum(data_lc), np.sum(modl_lc))
   print( np.sum((data_lc - modl_lc)**2 / data_lc ))
   #print( np.sum(data_lc - np.min(data_lc)), np.sum(modl_lc - np.min(modl_lc)) )
   #print( np.sum((data_lc - np.min(data_lc) - (modl_lc - np.min(modl_lc)) )**2 / data_lc ))
   plt.savefig(outfile)
