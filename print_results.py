import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from functions import print_parameters, midpoints, calculate_nebula_polarization, save_polmap, find_pol, array_to_fits, create_header

def load_archival(fname, param_dict, dtype=None):
   ''' 
   load simultaneous fitting results 
   '''
   assert dtype is not None
   num_det, num_phase, num_energy, num_dec, num_ra = print_parameters(param_dict, suppress=True)

   data = np.load(fname, allow_pickle=True)

   if dtype == 'psr':
      qp, qperr, up, uperr = data[0][:num_phase], data[2][:num_phase], data[1][:num_phase], data[3][:num_phase]
      pd, pderr, pa, paerr, sig = find_pol(qp, qperr, up, uperr)
      pa[pa < 0] += 180

      return pd, pderr, pa, paerr, sig

   if dtype == 'neb':
      qn, qnerr, un, unerr = data[0][num_phase:], data[2][num_phase:], data[1][num_phase:], data[3][num_phase:]
      qn = qn.reshape(num_dec, num_ra)
      qnerr = qnerr.reshape(num_dec, num_ra)
      un = un.reshape(num_dec, num_ra)
      unerr = unerr.reshape(num_dec, num_ra)

      return qn, qnerr, un, unerr

def pulsar_polarization(fnames, labels, lc_x, lc, param_dict, name, outfile, sig_cut=None):

   # read parameters
   num_det, num_phase, num_energy, num_dec, num_ra = print_parameters(param_dict, suppress=True)
   dx = np.diff(param_dict['PHASE_BINS']) / 2

   # plot results
   fig, ax = plt.subplots(2,2)
   ax1_twin = [ax[0,0].twinx(), ax[0,1].twinx()]
   handles1 = [list(), list()]
   ax2_twin = [ax[1,0].twinx(), ax[1,1].twinx()]
   handles2 = [list(), list()]

   for fn in fnames:
      pd, pderr, pa, paerr, sig = load_archival(f'source_{name}/results/{fn}.npy', param_dict, 'psr')
      if sig_cut is None:
         mask = (sig >= 0)
      else:
         mask = (sig >= sig_cut)
      print(pd, pderr)
      print(pa, paerr)

      for i in range(2):
         line = ax1_twin[i].errorbar(midpoints(param_dict['PHASE_BINS'])[mask], pd[mask], xerr=dx[mask], yerr=pderr[mask], fmt='', linestyle='none', label='04001299')
         handles1[i].append(line)

      for i in range(2):
         line = ax2_twin[i].errorbar(midpoints(param_dict['PHASE_BINS'])[mask], pa[mask], xerr=dx[mask], yerr=paerr[mask], fmt='', linestyle='none', label='04001299')
         handles2[i].append(line)

   for i in range(2):
      #ax1_twin[i].set_ylim(0,0.2)
      ax1_twin[i].set_ylim(0,1)
      ax1_twin[i].legend(handles=handles1[i], labels=labels, fontsize='x-small')
      ax1_twin[i].yaxis.set_label_position("left")
      ax1_twin[i].tick_params(labelleft=True, labelright=False, left=True, right=False)

      ax2_twin[i].hlines(0,0,1,linestyle='dashed',color='black')
      #ax2_twin[i].set_ylim(40,180)
      ax2_twin[i].set_ylim(0,180)
      ax2_twin[i].yaxis.set_major_locator(ticker.MultipleLocator(20))
      ax2_twin[i].legend(handles=handles2[i], labels=labels, fontsize='x-small')
      ax2_twin[i].yaxis.set_label_position("left")
      ax2_twin[i].tick_params(labelleft=True, labelright=False, left=True, right=False)

      ax[0,i].step(lc_x, lc, color='lightgray', where='pre')
      ax[1,i].step(lc_x, lc, color='lightgray', where='pre')

      ax[0,i].set_title('PD (%)')
      ax[0,i].set_yticks([])

      ax[1,i].set_title('PA (deg)')
      ax[1,i].set_xlabel('Phase')
      ax[1,i].set_yticks([])

   #ax[0,0].set_xlim(0.015,0.235)
   #ax[0,1].set_xlim(0.43,0.63)
   #ax[1,0].set_xlim(0.015,0.235)
   #ax[1,1].set_xlim(0.43,0.63)

   plt.subplots_adjust(hspace=0.3)
   plt.savefig(f'source_{name}/plots/pulsar_{outfile}.png', dpi=300, bbox_inches='tight')


def nebula_polarization_map(fname, name, param_dict, dcube_I, outfile, flux_cut=None, sig_cut=None):

   # gather the nebula polarization
   qn, qnerr, un, unerr = load_archival(fname, param_dict, 'neb')

   # flux-cut
   imap = np.sum(dcube_I, axis=(0,1,2,3))

   if flux_cut is not None:
      qn[imap < flux_cut] = np.nan
      qnerr[imap < flux_cut] = np.nan
      un[imap < flux_cut] = np.nan
      unerr[imap < flux_cut] = np.nan

   # significance-cut
   pdn, pdnerr, pan, panerr, sign = find_pol(qn, qnerr, un, unerr)

   if sig_cut is not None:
      pdn[sign < sig_cut] = np.nan
      pdnerr[sign < sig_cut] = np.nan
      pan[sign < sig_cut] = np.nan
      panerr[sign < sig_cut] = np.nan

   save_polmap(pdn, pan, sign, f'source_{name}/plots/{outfile}_nebula_pd.fits', f'source_{name}/plots/{outfile}_nebula_pa.reg', f'source_{name}/plots/{outfile}_nebula_sig.fits', param_dict, scale=1)
   array_to_fits(f'source_{name}/plots/{outfile}_nebula_I.fits', np.sum(dcube_I, axis=(0,1,2,3))[::-1,:], create_header(param_dict))

def nebula_polarization_int(fname, name, param_dict, A):
   
   # gather the nebula polarization
   qn, qnerr, un, unerr = load_archival(fname, param_dict, 'neb')

   # calculate spatially-averaged nebula polarization
   avg_pd, avg_pderr, avg_pa, avg_paerr, avg_sig = calculate_nebula_polarization(qn, qnerr, un, unerr, A, param_dict)
   print(avg_pd, avg_pderr, avg_pa, avg_paerr, avg_sig)


# list of results ('pre-glitch.npy')
#fnames = ['pre-glitch_01001099.npy', 'pre-glitch_02001099.npy', 'pre-glitch_02006001.npy', 'post-glitch_sep.npy', 'post-glitch_oct.npy']
#labels = ['01001099', '02001099', '02006001', '04001299 SEP', '04001299 OCT']

#pulsar_polarization(fnames, labels, 'simulfit.par', 'pulsar_pol.png')

#fig, ax = plt.subplots(2,1)
#ax_twin = ax[0].twinx()
#ax[0].step(lc_x, lc, where='mid', color='lightgray')
#ax_twin.errorbar(midpoints(param_dict['PHASE_BINS'])[mask], qp[mask], xerr=lc_dx[mask], yerr=qperr[mask], fmt='', linestyle='none', label='04001299 Q')
#ax_twin.errorbar(midpoints(param_dict['PHASE_BINS'])[mask], pre_qp[mask], xerr=lc_dx[mask], yerr=pre_qperr[mask], fmt='', linestyle='none', label='Pre-Glitch Q')
#ax_twin.hlines(0,0,1,linestyle='dashed',color='black')
#ax_twin.set_ylim(-0.5,0.5)
#ax_twin.legend()
#ax_twin = ax[1].twinx()
#ax[1].step(lc_x, lc, where='mid', color='lightgray')
#ax_twin.errorbar(midpoints(param_dict['PHASE_BINS'])[mask], up[mask], xerr=lc_dx[mask], yerr=uperr[mask], fmt='', linestyle='none', label='04001299 U')
#ax_twin.errorbar(midpoints(param_dict['PHASE_BINS'])[mask], pre_up[mask], xerr=lc_dx[mask], yerr=pre_uperr[mask], fmt='', linestyle='none', label='Pre-Glitch U')
#ax_twin.hlines(0,0,1,linestyle='dashed',color='black')
#ax_twin.set_ylim(-0.5,0.5)
#ax_twin.legend()
#ax[0].set_xlim(0,1)
#ax[1].set_xlim(0,1)
#plt.savefig('pulsar_new.png', dpi=300, bbox_inches='tight')
#plt.clf()
