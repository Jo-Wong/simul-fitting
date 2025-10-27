import numpy as np
import os
import copy
import scipy
from scipy.optimize import lsq_linear
import astropy.io.fits as fits

from ixpeobssim.evt.event import xEventFile
from ixpeobssim.irf.arf import xEffectiveArea
from ixpeobssim.irf.mrf import xModulationResponse
from ixpeobssim.irf.modf import xModulationFactor
from ixpeobssim.utils.astro import angular_separation, square_sky_grid
from ixpeobssim.utils.units_ import degrees_to_arcmin

RESP_FOLDER  = os.environ['PYTHONPATH'].split(':')[-1] + '/ixpeobssim/caldb/ixpe/'

DATA_LABELS = ['PHASE', 'ENERGY', 'RA', 'DEC', 'Q', 'U', 'W', 'MU']
CUBE_LABELS = ['I', 'WQ/MU', 'WU/MU', 'W', 'W2', 'WMU']

def midpoints(array):
    """  
    Return the midpoints of a given array of numbers
    """
    
    array = np.array(array)
    return (array[0:-1] + array[1:]) / 2

def create_header(param_dict):
    
    # Header format
    # Note: will not exactly correct pixel grid spacing, dif declinations will have dif spacing due to tang
    # projection, but for small ROI, should be apprx correct
    # Note: CDELT2 is not exactly pix_side either, but should be apprx equal
    
    RA = param_dict['RA']
    DEC = param_dict['DEC']
    
    if param_dict['SPATIAL_BIN_T'] == 'grid':

        ra_bins, dec_bins = generate_posbins(param_dict)
        num_ra = len(ra_bins) - 1
        num_dec = len(dec_bins) - 1

        header = fits.Header()
        header['NAXIS']   = 2  
        header['NAXIS1']  = num_ra
        header['NAXIS2']  = num_dec
        header['CTYPE1']  = 'RA---TAN'     
        header['CUNIT1']  = 'deg     '     
        header['CRPIX1']  = int(np.ceil(num_ra / 2))     
        header['CRVAL1']  = RA     
        header['CDELT1']  = -param_dict['SPATIAL_BIN_P']['pix_width'] / 3600     
        header['CTYPE2']  = 'DEC--TAN'     
        header['CUNIT2']  = 'deg     '     
        header['CRPIX2']  = int(np.ceil(num_dec / 2))     
        header['CRVAL2']  = DEC     
        header['CDELT2']  = param_dict['SPATIAL_BIN_P']['pix_width'] / 3600  
     
    elif param_dict['SPATIAL_BIN_T'] == 'circular':
        radius = param_dict['SPATIAL_BIN_P']['radius']
        pix_side = 2*radius / 3600 
     
        header = fits.Header()
        header['NAXIS']   = 2  
        header['NAXIS1']  = 1  
        header['NAXIS2']  = 1  
        header['CTYPE1']  = 'RA---TAN'     
        header['CUNIT1']  = 'deg     '     
        header['CRPIX1']  = 1     
        header['CRVAL1']  = RA     
        header['CDELT1']  = -pix_side     
        header['CTYPE2']  = 'DEC--TAN'     
        header['CUNIT2']  = 'deg     '     
        header['CRPIX2']  = 1     
        header['CRVAL2']  = DEC     
        header['CDELT2']  = pix_side 

    else:
        print('Spatial binning type not recognized.')
     
    return header

def array_to_fits(outfile, data_array, header = None):
    """  
    Creates a FITS file from a data stored as an ndarray
    """   
    if header == None:
        fits.writeto(outfile, data_array, overwrite=True)
    else:
        fits.writeto(outfile, data_array, header=header, overwrite=True)

def find_pol(norm_q, norm_qerr, norm_u, norm_uerr):
    pd = np.sqrt(norm_q**2 + norm_u**2)
    pa = np.rad2deg(0.5 * np.arctan2(norm_u, norm_q))
    pd_err = np.sqrt((norm_q * norm_qerr)**2 + (norm_u * norm_uerr)**2) / pd 
    pa_err = np.rad2deg(( 0.5 * np.sqrt((norm_u * norm_qerr)**2 + (norm_q * norm_uerr)**2 ) / pd**2))
    pa_err = pa_err % 180
    sig = pd / pd_err
    
    return pd, pd_err, pa, pa_err, sig

def save_polmap(pd_n, pa_n, sig_n, pi_outfile, pa_outfile, sig_outfile, param_dict, scale=1, save_all=False):
    '''  
    Create polarization map for nebula, with PD encoded in pixel values and PA represented as region file of vectors.
    Currently only supports grid spatial binning.
    '''
    
    RA  = param_dict['RA']
    DEC = param_dict['DEC']
    img_side = param_dict['SPATIAL_BIN_P']['img_width']
    pix_side = param_dict['SPATIAL_BIN_P']['pix_width']
    
    ra_bins, dec_bins = generate_posbins(param_dict)
    num_ra = len(ra_bins) - 1
    num_dec = len(dec_bins) - 1
    
    ra_bins = np.asarray(ra_bins)
    dec_bins = np.asarray(dec_bins)

    dec_ticks = (dec_bins[1:] + dec_bins[0:-1]) / 2
    ra_ticks = (ra_bins[1:] + ra_bins[0:-1]) / 2
    dec_grid, ra_grid = np.meshgrid(dec_ticks, ra_ticks[::-1], indexing='ij')

    dy = np.cos(np.deg2rad(pa_n) + np.pi/2) # for B-field, positive ^
    dx = np.sin(np.deg2rad(pa_n) + np.pi/2) # for B-field, positive >

    dy *= pix_side / 3600 / 2
    dx *= pix_side / 3600 / 2
    dy *= pd_n * scale
    dx *= pd_n * scale 
    ul_dec = dec_grid + dy 
    dr_dec = dec_grid - dy 
    ul_ra = ra_grid + dx / np.cos(np.radians(DEC))
    dr_ra = ra_grid - dx / np.cos(np.radians(DEC))

    if not save_all:
        # Write PA vectors as region file
        # Orange = 2 sigma, Yellow = 3, Green = 5
        flines = open(pa_outfile[:-4] + '2sig.reg', 'w') 
        flines.write('# Region file format: DS9 version 4.1\n')
        flines.write('global color=orange dashlist=8 3 width=3 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        flines.write('fk5\n')
        for i in range(num_dec):
            for j in range(num_ra):
                if np.logical_and(sig_n[i][j] > 2, sig_n[i][j] <=3):
                    if np.logical_not(np.logical_or(np.isnan(dx[i][j]), np.isnan(dy[i][j]))):
                        _ = flines.write('line(%.14f, %.14f, %.14f, %.14f) # line=0 0\n' % (ul_ra[i][j], ul_dec[i][j], dr_ra[i][j], dr_dec[i][j]) )
        flines.close()

        flines = open(pa_outfile[:-4] + '3sig.reg', 'w') 
        flines.write('# Region file format: DS9 version 4.1\n')
        flines.write('global color=yellow dashlist=8 3 width=3 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        flines.write('fk5\n')
        for i in range(num_dec):
            for j in range(num_ra):
                if np.logical_and(sig_n[i][j] > 3, sig_n[i][j] <=5):
                    if np.logical_not(np.logical_or(np.isnan(dx[i][j]), np.isnan(dy[i][j]))):
                        _ = flines.write('line(%.14f, %.14f, %.14f, %.14f) # line=0 0\n' % (ul_ra[i][j], ul_dec[i][j], dr_ra[i][j], dr_dec[i][j]) )
        flines.close()

        flines = open(pa_outfile[:-4] + '5+sig.reg', 'w') 
        flines.write('# Region file format: DS9 version 4.1\n')
        flines.write('global color=green dashlist=8 3 width=3 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        flines.write('fk5\n')
        for i in range(num_dec):
            for j in range(num_ra):
                if sig_n[i][j] > 5: 
                    if np.logical_not(np.logical_or(np.isnan(dx[i][j]), np.isnan(dy[i][j]))):
                        _ = flines.write('line(%.14f, %.14f, %.14f, %.14f) # line=0 0\n' % (ul_ra[i][j], ul_dec[i][j], dr_ra[i][j], dr_dec[i][j]) )
        flines.close()
    else:
        # Write all PA vectors as region file
        flines = open(pa_outfile[:-4] + '.reg', 'w') 
        flines.write('# Region file format: DS9 version 4.1\n')
        flines.write('global color=blue dashlist=8 3 width=3 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        flines.write('fk5\n')
        for i in range(num_dec):
            for j in range(num_ra):
                if np.logical_not(np.logical_or(np.isnan(dx[i][j]), np.isnan(dy[i][j]))):
                    _ = flines.write('line(%.14f, %.14f, %.14f, %.14f) # line=0 0\n' % (ul_ra[i][j], ul_dec[i][j], dr_ra[i][j], dr_dec[i][j]) )
        flines.close()
     
    header = create_header(param_dict)
    hdul = fits.HDUList([fits.PrimaryHDU(header = header, data = pd_n)])
    hdul.writeto(pi_outfile, overwrite = True)
    hdul = fits.HDUList([fits.PrimaryHDU(header = header, data = sig_n)])
    hdul.writeto(sig_outfile, overwrite = True)

def shift_center(d_ra, d_dec, param_dict):
   """
   Perform artifical shifts to the RA/DEC in <param_dict> for calibration.
   + d_ra: will shift image to the right
   + d_dec: will shift image upward
   """
   param_dict['DEC'] = Constants.DEC - shift_dec / 3600
   param_dict['RA'] = Constants.RA + shift_ra / 3600 / np.cos(np.radians(param_dict['DEC']))

   return param_dict

def response_file(resp_name, resp_type):
    if resp_type == 'ARF':
        return RESP_FOLDER + 'gpd/cpf/arf/' + resp_name + '.arf'
    
    elif resp_type == 'MRF':
        return RESP_FOLDER + 'gpd/cpf/mrf/' + resp_name + '.mrf'

    elif resp_type == 'MODF':
        return RESP_FOLDER + 'gpd/cpf/modfact/' + resp_name[:-4] + 'mfact_' + resp_name[-4:] + '.fits'
     
    elif resp_type == 'VIG':
        return RESP_FOLDER + 'xrt/bcf/vign/' + resp_name[:-4] + 'vign_' + resp_name[-4:] + '.fits'

def truncate(irf_name):
    ''' Facility to parse irfname that has been modified (i.e. by ixpecalcarf) into a digestable form for <vign>, <rmf> '''
    comp = irf_name.split('_')
    
    new_irf_name = comp[0:3]
    if 'alpha075' in comp:
        new_irf_name.append('alpha075')
    
    new_irf_name.append(comp[-1])

    return '_'.join(new_irf_name)

def square_sky_grid(nside, center, half_size):
    """Create a square, regular grid in ra and dec.

    Arguments
    ---------
    nside : int
        The number of points for either side of the grid

    center : (float, floar)
        The ra, dec center of the grid in decimal degrees.

    half_size : float
        The half-size of the grid in decimal degrees.
    """
    ra0, dec0 = center
    grid = numpy.linspace(-half_size, half_size, nside)
    ra = grid / numpy.cos(numpy.radians(dec0)) + ra0 
    dec = grid + dec0
    return numpy.meshgrid(ra, dec)

def rect_sky_grid(nside, center, half_size):
    """Create a rectangular grid in ra and dec. (Modified from <square_sky_grid>)
    
    Arguments
    ---------
    nside : array of ints, (ra, dec)
        The number of points for either side of the grid

    center : (float, float)
        The ra, dec center of the grid in decimal degrees.

    half_size : array of floats, (ra, dec)
        The half-size of the grid in decimal degrees.
    """
    ra0, dec0 = center
    grid_ra = np.linspace(-half_size[0], half_size[0], nside[0])
    grid_dec = np.linspace(-half_size[1], half_size[1], nside[1])
    ra = grid_ra / np.cos(np.radians(dec0)) + ra0
    dec = grid_dec + dec0
    return np.meshgrid(ra, dec)

def generate_posbins(params):
    """
    Creates RA and DEC position bins given center position, and pixel and image sizes.
    Pixel width is fixed, image size is adjusted to ensure odd number of ra bins/dec bins.
    Image size is always scaled down.
    """
    ra = params['RA']
    dec = params['DEC']
    img_size = params['SPATIAL_BIN_P']['img_width']
    pix_size = params['SPATIAL_BIN_P']['pix_width']

    if isinstance(img_size, (int, float)):
        num_edges = int(np.floor(img_size / pix_size) + 1)

        if np.mod(num_edges, 2) == 1:
            num_edges -= 1

        ra_grid, dec_grid = square_sky_grid(num_edges, (ra, dec), (pix_size * (num_edges-1)) / 3600 / 2)

        return ra_grid[0,:], dec_grid[:,0]

    elif isinstance(img_size, tuple) and len(img_size) == 2:

        # Assumes that img_size = (ra, dec)
        num_edges = [int(np.floor(x / pix_size)) + 1 for x in img_size]
        num_edges = [x - np.mod(x, 2) for x in num_edges]
        num_edges = np.array(num_edges)

        ra_grid, dec_grid = rect_sky_grid(num_edges, (ra, dec), (pix_size * (num_edges-1)) / 3600 / 2)

        return ra_grid[0,:], dec_grid[:,0]

    else:
        raise TypeError('Image size must be either a number or tuple.')

def get_spatial_dims(params):
    '''
    Convenience function to get the number of dec, ra bins for a given parameter setting
    '''

    if params['SPATIAL_BIN_T'] == 'grid':
        ra_bins, dec_bins = generate_posbins(params)
        return len(ra_bins)-1, len(dec_bins)-1

    elif params['SPATIAL_BIN_T'] == 'circular' or params['SPATIAL_BIN_T'] == 'annulus' or params['SPATIAL_BIN_T'] == 'file':
        return 1, 1

    else:
        print('invalid spatial binning')
        return

def print_parameters(params, suppress=False):
    '''  
    Convenience function for printing a parameter dictionary.
    '''

    phase_bins  = np.array(params['PHASE_BINS'])
    energy_bins = np.array(params['ENERGY_BINS'])

    if 'DFILE' in params.keys():
       num_obs    = len(params['DFILE'])

    num_det    = len(params['DETECTORS'])
    num_phase  = len(phase_bins) - 1
    num_energy = len(energy_bins) - 1
    num_ra, num_dec = get_spatial_dims(params)
    
    if suppress==False:

        print('RA: ', params['RA'])
        print('DEC: ', params['DEC'])
        print('Weights on/off: ', ('On' if params['WEIGHTS'] else 'Off'))
        if len(set(np.diff(phase_bins))) == 1:
            print('Phase binning: equal-width, %d bins' % num_phase)
        else:
            print('Phase binning: variable bins', phase_bins)

        print('Energy binning: %d bins' % num_energy)
        print('Energy Bins: ', energy_bins)

        print('Spatial binning: ', params['SPATIAL_BIN_T'])

        if params['SPATIAL_BIN_T'] == 'grid':
            img_side = params['SPATIAL_BIN_P'].get('img_width')
            pix_side = params['SPATIAL_BIN_P'].get('pix_width')
            print('Pixel width: ', pix_side)

            if isinstance(img_side, tuple):
                print('Image width: ~', (img_side[1], img_side[0]))
            else:
                print('Image width: ~', img_side)

            print('True image width:', num_dec * pix_side, 'x', num_ra * pix_side)

        elif params['SPATIAL_BIN_T'] == 'circular' or params['SPATIAL_BIN_T'] == 'annulus':
            print('ROI Radius: ', params['SPATIAL_BIN_P'].get('radius'))

        elif params['SPATIAL_BIN_T'] == 'file':
            print('Region file: ', params['SPATIAL_BIN_P'].get('filepath'))

        print('Cube Shape: ', (num_phase, num_energy, num_dec, num_ra))

        print('Response function: ', [params['RESP_NAME'][i][0] % 'alpha075_' if params['WEIGHTS'] else params['RESP_NAME'][i][0] % '' for i in range(len(params['RESP_NAME']))])

        print('Model phase rotation: ', params['ROT_PH'])

        if 'DFILE' in params.keys():
           print('Data File: ', params['DFILE'])
   
    if 'DFILE' in params.keys(): 
       return num_obs, num_det, num_phase, num_energy, num_dec, num_ra
    else:
       return num_det, num_phase, num_energy, num_dec, num_ra

def process_data(fname):
    event_file = xEventFile(fname)

    phase = event_file.phase_data()
    energy  = event_file.energy_data()
    ra, dec = event_file.sky_position_data(mc=False)

    return phase, energy, ra, dec

def load_data(fits_folder, fits_file, data_type, params, resp_name, use_proxy_weights):
    data = dict.fromkeys(DATA_LABELS)

    fname = fits_folder + '/' + fits_file
    print(fname)

    dfile = fits.open(fname)
    devt  = dfile['EVENTS'].data

    weights = params['WEIGHTS']

    if weights:
        resp_name = resp_name % 'alpha075_'
    else:
        resp_name = resp_name % ''

    modf = xModulationFactor(response_file(truncate(resp_name), 'MODF'))

    if data_type == 'DATA':
        phase, energy, ra, dec = process_data(fname)
    elif data_type == 'MODEL':
        if 'PHASE' in devt.columns.names:
            phase = np.mod(devt['PHASE'] + params['ROT_PH'], 1)
        else:
            phase = None
        energy = devt['ENERGY']
        ra     = devt['RA']
        dec    = devt['DEC']
    else:
        print('Invalid data type')
        return

    data['PHASE'] = phase
    data['ENERGY'] = energy
    data['RA'] = ra
    data['DEC'] = dec
    data['Q'] = devt['Q']
    data['U'] = devt['U']
    data['MU'] = modf(energy)

    if weights:
        if use_proxy_weights is not None:
            print('using proxy weights')

            wmom_proxy_list, ebins = np.load(use_proxy_weights, allow_pickle=True)
            indices = np.digitize(data['ENERGY'], ebins.tolist())
            wmom_proxy = np.zeros(len(indices))

            for idx, (e1, e2) in enumerate(zip(ebins[0:-1], ebins[1:])):
                key = '%.1f_%.1f' % (e1, e2)
                wmom_proxy[indices==idx+1] = np.random.choice(wmom_proxy_list[key], np.sum(indices==idx+1), replace=True)


            data['W'] = wmom_proxy
            del ebins, wmom_proxy_list

        else:
            print('using wmom col')
            data['W'] = devt['W_MOM']

    else:
        data['W'] = np.ones(len(energy))

    # If region file provided for spatial filtering, filter now:
    if params['SPATIAL_BIN_T'] == 'file':
        mask = xEventFile(fname).ds9_region_file_mask(params['SPATIAL_BIN_P']['filepath'], mc=False)

        for label in DATA_LABELS:
            if data[label] is not None:
               data[label] = data[label][mask]

    return data

def bin_data(data_dict, params):

    _, num_phase, num_energy, num_dec, num_ra = print_parameters(params, suppress=True)[-5:]
    phase_bins  = params['PHASE_BINS']
    energy_bins = params['ENERGY_BINS']

    if params['SPATIAL_BIN_T'] == 'grid':
        ra_bins, dec_bins = generate_posbins(params)

    elif params['SPATIAL_BIN_T'] == 'circular':
        rad = params['SPATIAL_BIN_P'].get('radius')

    elif params['SPATIAL_BIN_T'] == 'annulus':
        inner_rad, outer_rad = params['SPATIAL_BIN_P'].get('radius')

    elif params['SPATIAL_BIN_T'] == 'file':
        print('data has already been filtered by specified region file')

    else:
        print('Invalid spatial binning type.')
        return

    cube = dict.fromkeys(CUBE_LABELS)
    for key in cube.keys():
        cube[key] = np.ndarray((num_phase, num_energy, num_dec, num_ra))
    print('Cube shape: ', (num_phase, num_energy, num_dec, num_ra))

    phase  = data_dict['PHASE']
    energy = data_dict['ENERGY']
    ra     = data_dict['RA']
    dec    = data_dict['DEC']
    q      = data_dict['Q']
    u      = data_dict['U']
    w      = data_dict['W']
    mu     = data_dict['MU']

    def phase_mask(phase, pmin, pmax):
        ''' convenience function to create a mask to select phases between pmin and pmax'''

        # make sure the phases range from [0,1] 
        assert(np.all(np.logical_and(phase >= 0, phase <= 1)))

        # We only consider cases where the first phase bin < 0 or last phase bin > 1.
        if pmin >= 0 and pmax <= 1:
            mask = np.logical_and(np.greater(phase, pmin), np.less_equal(phase, pmax))

        if pmin < 0 and pmax <= 1:
            mask = np.logical_or(np.less_equal(phase, pmax), np.greater(phase, np.mod(pmin,1)))

        if pmin >= 0 and pmax > 1:
            mask = np.logical_or(np.greater(phase, pmin), np.less_equal(phase, np.mod(pmax,1)))

        return mask


    for i in range(num_phase):

        mask_ph = phase_mask(phase, phase_bins[i], phase_bins[i+1])

        for j in range(num_energy):
            emin, emax = energy_bins[j], energy_bins[j+1]
            mask_e  = np.logical_and(np.greater(energy, emin), np.less_equal(energy, emax))

            mask = np.logical_and(mask_ph, mask_e)

            pm  = phase[mask]
            em  = energy[mask]
            rm  = ra[mask]
            dm  = dec[mask]
            qm  = q[mask]
            um  = u[mask]
            wm  = w[mask]
            mum = mu[mask]

            if params['SPATIAL_BIN_T'] == 'grid':
                for k in range(num_dec):
                    dec_min, dec_max = dec_bins[k], dec_bins[k+1]
                    mask_dec = np.logical_and(np.greater(dm, dec_min), np.less_equal(dm, dec_max))

                    for l in range(num_ra):
                        ra_min, ra_max = ra_bins[l], ra_bins[l+1]
                        mask_ra  = np.logical_and(np.greater(rm, ra_min), np.less_equal(rm, ra_max))

                        mask = np.logical_and(mask_dec, mask_ra)

                        # We've reversed the order of RA indexing to match ds9 image
                        cube['I'][i][j][k][len(ra_bins)-2-l] = np.sum(mask)
                        cube['WQ/MU'][i][j][k][len(ra_bins)-2-l] = np.sum(wm[mask] * qm[mask] / mum[mask])
                        cube['WU/MU'][i][j][k][len(ra_bins)-2-l] = np.sum(wm[mask] * um[mask] / mum[mask])
                        cube['W'][i][j][k][len(ra_bins)-2-l] = np.sum(wm[mask])
                        cube['W2'][i][j][k][len(ra_bins)-2-l] = np.sum(wm[mask]**2)
                        cube['WMU'][i][j][k][len(ra_bins)-2-l] = np.sum(wm[mask] * mum[mask])

            if params['SPATIAL_BIN_T'] == 'circular' or params['SPATIAL_BIN_T'] == 'annulus':
                sep = degrees_to_arcmin(angular_separation(rm, dm, params['RA'], params['DEC']))

                if params['SPATIAL_BIN_T'] == 'circular':
                    mask = sep < rad / 60

                if params['SPATIAL_BIN_T'] == 'annulus':
                    mask = (sep < outer_rad / 60) & (sep > inner_rad / 60)

                cube['I'][i][j][0][0] = np.sum(mask)
                cube['WQ/MU'][i][j][0][0] = np.sum(wm[mask] * qm[mask] / mum[mask])
                cube['WU/MU'][i][j][0][0] = np.sum(wm[mask] * um[mask] / mum[mask])
                cube['W'][i][j][0][0] = np.sum(wm[mask])
                cube['W2'][i][j][0][0] = np.sum(wm[mask]**2)
                cube['WMU'][i][j][0][0] = np.sum(wm[mask] * mum[mask])

            if params['SPATIAL_BIN_T'] == 'file':
                cube['I'][i][j][0][0] = np.sum(mask)
                cube['WQ/MU'][i][j][0][0] = np.sum(wm * qm / mum)
                cube['WU/MU'][i][j][0][0] = np.sum(wm * um / mum)
                cube['W'][i][j][0][0] = np.sum(wm)
                cube['W2'][i][j][0][0] = np.sum(wm**2)
                cube['WMU'][i][j][0][0] = np.sum(wm * mum)

    return cube

def generate_cube(files, folder, params, dtype, use_proxy_weights=None):
    '''  
    Generates binned data cube.
    
    Number of observations is decided by the length of <files>. <params> is only used to decide the
    detectors, phase, energy, and spatial binning.
    
    <use_proxy_weights> should be the file containing the proxies. If not using, set to None. Used 
    to assign proxy weights for simulation events, which don't have associated weights.
    '''
  
    # initialize the data cube, <obs_cube>
    obs_cube = dict.fromkeys(CUBE_LABELS)

    # create a copy of the parameter dictionary since we might make changes
    params_copy = copy.deepcopy(params)

    params_copy['DFILE'] = files 
    num_obs, num_det, num_phase, num_energy, num_dec, num_ra = print_parameters(params_copy, suppress=True)
   
    # initialize each value in <obs_cube>
    for label in CUBE_LABELS:
        obs_cube[label] = np.zeros(shape=(num_obs, num_det, num_phase, num_energy, num_dec, num_ra))
   
    # populate data cube for each observation/detector 
    for obs in range(num_obs):
        for i, det in enumerate(params_copy['DETECTORS']):
            print('Generating cube for det %d' % det) 
            data = load_data(folder, files[obs][det-1], dtype, params_copy, params_copy['RESP_NAME'][obs][det-1], use_proxy_weights)

            # should theoretically be outside the detector loop, but assuming that all detectors would either have or not have PHASE
            if data['PHASE'] is None: 
                print('No phase column exists. Changing PHASE_BINS = [0,1] and creating dummy phase data = 0.5')
                params_copy['PHASE_BINS'] = [0,1]
                data['PHASE'] = 0.5 * np.ones_like(data['ENERGY'])

            cube = bin_data(data, params_copy)
     
            for label in CUBE_LABELS:
                obs_cube[label][obs][i] = cube[label]
    
    print('done!')
    
    return obs_cube

def long_simulation(fname, param_dict, num_sim, INPUT_FILE, use_proxy_weights):
    '''  
    Like in <generate_cube> itself, the number of observations is decided by <INPUT_FILE>. The parameter
    dictionary only sets the detectors, phase, energy, and spatial binning.
    '''
    
    cube = None
    
    for i in range(1,num_sim + 1):
        temp = generate_cube(INPUT_FILE, fname%i, param_dict, 'MODEL', use_proxy_weights=use_proxy_weights)
    
        if cube is None:
           cube = copy.deepcopy(temp)
        else: 
           for label in CUBE_LABELS:
               cube[label] += temp[label]
     
    return cube 

def add_covariance(dcube):
    ''' Add VAR_Q, VAR_U, COV_QU terms to dcube'''
    
    # collect data and calculate covariance
    q_tot = dcube['WQ/MU']
    u_tot = dcube['WU/MU']
    w_tot = dcube['W']
    w2_tot = dcube['W2']
    mu_tot = dcube['WMU']

    avg_mu_tot = mu_tot / w_tot
    avg_q_tot = q_tot / w_tot
    avg_u_tot = u_tot / w_tot

    avg_q_tot = np.clip(avg_q_tot, -1, 1)
    avg_u_tot = np.clip(avg_u_tot, -1, 1)

    dcube['VAR_Q'] = 2 * w2_tot / avg_mu_tot**2 * (1 - (avg_q_tot * avg_mu_tot)**2 / 2) 
    dcube['VAR_U'] = 2 * w2_tot / avg_mu_tot**2 * (1 - (avg_u_tot * avg_mu_tot)**2 / 2) 
    dcube['COV_QU'] = -w2_tot * avg_q_tot * avg_u_tot

def simul(dcube, pcube, ncube, param_dict):
    
    add_covariance(dcube)
    return simul_bare(dcube, pcube, ncube, param_dict)

def simul_bare(dcube, pcube, ncube, param_dict):
    '''
    Simultaneous fitting. Data cube should already have covariance terms. User also 
    needs to provide the model cubes and parameters.
    Note: pspec(det,ph,e) and nspec(det,e,ra,dec) should already be normalized.
    '''

    w_tot = dcube['W']
    q_tot = dcube['WQ/MU']
    u_tot = dcube['WU/MU']
    var_q_tot = dcube['VAR_Q']
    var_u_tot = dcube['VAR_U']
    cov_qu_tot = dcube['COV_QU']

    # collect parameters
    num_obs, num_det, num_phase, num_energy, num_dec, num_ra = w_tot.shape

    phase_bins = param_dict['PHASE_BINS']

    # number of data points
    num_points = num_obs * num_det * num_phase * num_energy * num_dec * num_ra

    # bin spacing    
    bin_width = np.diff(phase_bins)

    # empty parameter array
    norm_q = np.empty((num_phase + num_dec * num_ra))
    norm_u = np.empty((num_phase + num_dec * num_ra))
    norm_qerr = np.empty((num_phase + num_dec * num_ra))
    norm_uerr = np.empty((num_phase + num_dec * num_ra))

    Wp = pcube / (pcube + ncube) * w_tot
    Wn = ncube / (pcube + ncube) * w_tot

    # If dcube['W'] == 0 somewhere, then so will Q and U
    mask = np.logical_not(w_tot == 0)
    print(np.sum(mask))

    # Also, for now, since we're using the models exactly, there's a risk we might get NaN's. Filtering those.
    mask = np.logical_and(mask, ~(pcube+ncube == 0))

    A = np.zeros(shape=(2 * num_points, 2 * (num_phase + num_dec * num_ra)))
    b = np.zeros(shape=(2 * num_points))

    count = 0
    for obs in range(num_obs):
        for det in range(num_det):
            for i in range(num_phase):
                for l in range(num_energy):
                    for j in range(num_dec):
                        for k in range(num_ra):
                            A[2*count][2*i] = Wp[obs][det][i][l][j][k]
                            A[2*count+1][2*i + 1] = Wp[obs][det][i][l][j][k]
                            A[2*count][2*(num_phase + j * num_ra + k)] = Wn[obs][det][i][l][j][k]
                            A[2*count+1][2*(num_phase + j * num_ra + k) + 1] = Wn[obs][det][i][l][j][k]

                            count += 1

    b[::2]  = q_tot.ravel()
    b[1::2] = u_tot.ravel()

    # Create the weight matrix
    weight = []
    row = []
    col = []

    ws = np.zeros((2,2))
    mask_rinse = []

    for obs in range(num_obs):
        for det in range(num_det):
            for i in range(num_phase):
                for l in range(num_energy):
                    for j in range(num_dec):
                        for k in range(num_ra):
                            ws[0,0] = var_q_tot[obs,det,i,l,j,k]
                            ws[1,1] = var_u_tot[obs,det,i,l,j,k]
                            ws[1,0] = cov_qu_tot[obs,det,i,l,j,k]
                            ws[0,1] = cov_qu_tot[obs,det,i,l,j,k]

                            try:
                                ws_inv = np.linalg.inv(ws)
                                ws_ch = np.linalg.cholesky(ws_inv)

                                if np.any(np.isnan(ws_ch)) or mask[obs,det,i,l,j,k] == 0:
                                    mask_rinse.append(0)

                                else:
                                    weight.append(ws_ch.ravel())
                                    mask_rinse.append(1)

                            except:
                                print(ws)
                                print('Cannot invert variance matrix or perform Cholesky decomp.')
                                mask_rinse.append(0)

    weight = np.asarray(weight).ravel()

    count = np.sum(mask_rinse)
    row = np.repeat(np.arange(2 * count), 2)
    col = np.repeat(np.asarray([(2*i,2*i+1) for i in range(count)]), 2, axis=0).ravel()

    A = A[np.repeat(mask_rinse, 2).astype(bool), :]
    b = b[np.repeat(mask_rinse, 2).astype(bool)]

    w3 = scipy.sparse.csr_matrix((weight, (row, col)), shape=(2*count,2*count))
    A3 = scipy.sparse.csr_matrix(A)
    b3 = scipy.sparse.csr_matrix(b)

    Aw = np.asarray((w3.transpose() @ A3).todense()) #np.matmul(w2.T, A2)
    bw = np.squeeze(np.asarray((w3.transpose() @ b3.transpose()).todense())) #np.matmul(w2.T, b2)
    params = lsq_linear(Aw, bw, bounds = (-1,1), verbose=2)
    norm_q = params.x[::2]
    norm_u = params.x[1::2]

    err = np.sqrt(np.diag(scipy.sparse.linalg.inv(A3.transpose() @ w3 @ w3.transpose() @ A3).todense()))
    norm_qerr = np.asarray(err[::2])
    norm_uerr = np.asarray(err[1::2])

    return norm_q, norm_u, norm_qerr, norm_uerr, A

def calculate_nebula_polarization(qn, qnerr, un, unerr, A):
   ''' calculates spatially-averaged nebula polarization '''

   polarization = np.zeros(2*(num_phase+num_dec*num_ra))
   polarization[2*num_phase::2] = qn.ravel()
   polarization[2*num_phase+1::2] = un.ravel()
   pol_flux = A @ polarization

   flux = np.zeros(2*(num_phase+num_dec*num_ra))
   flux[2*num_phase::2] = 1 
   flux[2*num_phase+1::2] = 1 
   count_flux = A @ flux

   polarization_err = np.zeros(2*(num_phase+num_dec*num_ra))
   polarization_err[2*num_phase::2] = qnerr.ravel()**2
   polarization_err[2*num_phase+1::2] = unerr.ravel()**2
   pol_flux_err = np.power(A,2) @ polarization_err

   pol_avg_qerr = np.sqrt(np.sum(pol_flux_err[::2]))  / np.sum(count_flux[::2])
   pol_avg_uerr = np.sqrt(np.sum(pol_flux_err[1::2])) / np.sum(count_flux[::2])

   pol_avg_pd, pol_avg_pderr, pol_avg_pa, pol_avg_paerr, pol_avg_sig = find_pol(pol_avg_q, pol_avg_qerr, pol_avg_u, pol_avg_uerr)
   
   return pol_avg_pd, pol_avg_pderr, pol_avg_pa, pol_avg_paerr, pol_avg_sig
