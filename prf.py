from .utils import *


class RomanPRF(object):

    def __init__(bandpass, sca, spectype='M0V',):

        self.bandpass=bandpass
        self.sca=sca
        self.spectype=spectype


    def _get_psf_model(self, ):

        if fname is None:
            fname = ROMSIM_PACKAGE_DIR + '/psf_models/'+bandpass+'/wfi_psf_'+self.spectype+'_'+bandpass+'_'+self.sca+'.fits'

        hdul = fits.open(fname)
        self.psf_fits = hdul
        self.psf_model = self.psf_fits[0].data
        self.prf_model = self.psf_fits[1].data

        self.psf_model/=np.sum(self.psf_model)
        self.prf_model/=np.sum(self.prf_model)
        
        hdul.close()

        
    def _build_PRF_scene_FFT(self, coords, mags, ):

        return 1.

    def build_PRF_scene(self, coords, mags):

        return 1. 

    def _interpolate_PSF(self, ):

        return 1.

    def interpolate_scene(self, dx, dy):

        return 1. 
        

