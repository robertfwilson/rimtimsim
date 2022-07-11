
from .injection import *
from .romsim import *




def make_injected_image(RomImg, lightcurve_params):




    return 1.



def save_simulated_fits_image(data, err, t, datadir):


    data, err = test_image.make_realistic_image(bandpass='F146', exptime=46.8,
                                                oversample=True)
    
    t = t0+i*dt+TimeDelta(np.random.randn(1)[0]*0.005, format='sec')
    
    hdr = test_image.make_header(obs_time = t.isot)
    hdu = fits.PrimaryHDU(data, header=hdr)
    
    fname='simdata/run4/roman_btds_field1_f146_sca07_'+hdr['DATE-OBS']+'_sim.fits'
    hdu.writeto(fname , overwrite=True)
    
    
    fname = data_dir+'roman_btds_field1_f146_sca07_'+hdr['DATE-OBS']+'_sim.fits'


    




