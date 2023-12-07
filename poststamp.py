from .romsim import RomanImage, ROMSIM_PACKAGE_DIR
import numpy as np
from tqdm import tqdm
import pandas as pd

from .injection import update_stellar_catalog
from .utils import matrix_solve, linear_leastsq


class TimeSeriesCutout(RomanImage):

    def __init__(self, star, frame_size, bkg_stars=None, **kw):

        self.frame_size=frame_size
        self.star = star
        self.bkg_stars=bkg_stars
        self.subframe=frame_size

        super().__init__(subframe=frame_size, bandpass=star.bandpass, **kw)

    

    def generate_random_bkg_stars(self, detector_catalog=None, detector_size=(4096,4096)):

        if detector_catalog==None:
            detector_catalog = pd.read_csv(ROMSIM_PACKAGE_DIR+'/data/starcats/bulge_catalog_mags.dat', index_col=0)
    
        n_stars_exp = len(detector_catalog) * (self.frame_size[0]*self.frame_size[1]) / (detector_size[0] * detector_size[1])
    
        n_stars = np.random.poisson(n_stars_exp)
    
        bkg_stars = detector_catalog.iloc[np.random.choice(np.arange(len(detector_catalog)), size=n_stars)].copy()
    
        #random_position = *frame_size[1]
        bkg_stars['xcol']=np.random.rand(n_stars) * self.frame_size[0]
        bkg_stars['ycol']=np.random.rand(n_stars) * self.frame_size[1]
    
        bkg_stars=bkg_stars.reset_index()
        self.bkg_stars=bkg_stars
        
        return bkg_stars
        


    def get_simulated_cutout(self, star, star_xy, frame_size=(32,32),
                             bandpass='F146',multi_accum=[1,2,3,4,4,4], n_zodi=5.):

        self.star_xy=star_xy

        target_mag_timeseries = pd.DataFrame([star._dflux_to_dmag()], index=[-1])

        #print(star._dflux_to_dmag(), sum(star._dflux_to_dmag()))

        target_star = pd.DataFrame({'sicbroid':[-1], 'xcol':[star_xy[0]], 'ycol':[star_xy[1]],self.bandpass:[star.mag]}, index=[-1])

        
        bkg_catalog=self.bkg_stars.reset_index()
        all_stars = pd.concat([target_star, bkg_catalog[['sicbroid','xcol','ycol',self.bandpass]]])

        self.SourceCatalog = all_stars
        imgs = []
        errs=[]
        
        i=min(target_mag_timeseries.columns)
        
        for t in tqdm(star.time):

            delta_mags = target_mag_timeseries[i]
            
            new_star_cat = update_stellar_catalog(all_stars, delta_mags, mag_col='F146')


            data, data_err = self.make_realistic_image(oversample=True, bandpass='F146', \
                                                    read_style='cas22_ramp', return_err=True, \
                                                    multiaccum_table=multi_accum, \
                                                    star_list=new_star_cat[:,1:],
                                                  trim_psf_kernel=True,  )

            i+=1
            imgs.append(data)
            errs.append(data_err)

        self.cutouts = imgs
        self.cutout_errs=errs
            
        return imgs


    def get_PSF_lightcurve(self, stars=[-1], assume_constant_bkg=True,
                           assume_constant_bkg_stars=True, dithered=False):

        target_stars = self.SourceCatalog.loc[stars]        
        
        if assume_constant_bkg:
            # Calculate the Zodiacal and thermal background in the simulated images
            sky_bkg = self.n_min_zodiacal_background * self.wfiprops.minzodi_background
            sky_bkg += self.wfiprops.thermal_background
            
        if assume_constant_bkg_stars:
            bkg_stars = self.SourceCatalog.drop(stars)

        target_scene = self._get_target_scene(target_stars)
        bkg_scene = self._get_background_scene(bkg_stars )

        self.target_scene = target_scene
        self.bkg_scene = bkg_scene
        
        data = np.array(self.cutouts)
        data_err = np.sqrt(np.array(self.cutout_errs))

        #Replace Saturated Pixels
        sat_mask = ~np.isfinite(data)
        data[sat_mask]=1e5
        data_err[sat_mask] = np.inf


        #if np.shape(data_err)!=np.shape(data):
        #data_err = np.ones_like(data)
            
        
        #if dithered:
        # Currently only works for non-dithered data. Whch is cool for right now.     
        n_frames = np.array(data).shape[0]

        # Create Design Matrix
        #A =  np.vstack([target_scene.ravel()]).T

        # Solve for the Flux in each frame
        flux_weights = np.array([ matrix_solve(x=target_scene, y=data[i]-bkg_scene, y_err=data_err[i]) for i in range(len(data)) ])
        
        flux, flux_err = flux_weights.reshape(-1, 2).T

        #print(flux, flux_err)

        lightcurve = {'time':self.star.time, 'psf_flux':flux, 'psf_flux_err':flux_err, 'injected_flux': self.star.d_flux+1.}
        
        return lightcurve


    def _get_background_scene(self, bkg_stars):
        
        bkg_scene = self._make_expected_source_image(star_list=bkg_stars.to_numpy(),
                                                     oversample=True)
        self.bkg_scene = bkg_scene
        
        return bkg_scene
    
    def _get_target_scene(self, target_stars):
        
        target_scene = self._make_expected_source_image(star_list=target_stars.to_numpy(),oversample=True, include_sky=False)
        
        self.target_scene = target_scene
        
        return target_scene

        
    
    def set_base_target_catalog(self, catalog):

        self.targ_stars = catalog



    def get_timeseries(self, ):

        return 1. 


    def calc_img_cutout(self, ):

        return 1. 


   
