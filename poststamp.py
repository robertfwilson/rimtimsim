from .romsim import RomanImage
import numpy as np
from tqdm import tqdm
import pandas as pd

from .injection import update_stellar_catalog



class TimeSeriesCutout(RomanImage):

    def __init__(self, star, frame_size, bkg_stars=None, **kw):

        self.frame_size=frame_size
        self.star = star
        self.bkg_stars=bkg_stars
        self.subframe=frame_size

        super(TimeSeriesCutout, self).__init__(subframe=frame_size, **kw)

    

    def generate_random_bkg_stars(self, detector_catalog, detector_size=(4096,4096)):
    
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
                      multi_accum=[1,2,3,4,4,4], n_zodi=5.):

        target_mag_timeseries = pd.DataFrame([star._dflux_to_dmag()], index=[-1])

        target_star = pd.DataFrame({'sicbroid':[-1], 'xcol':[star_xy[0]], 'ycol':[star_xy[1]],'F146':[star.mag]}, index=[-1])

        
        bkg_catalog=self.bkg_stars.reset_index()
        all_stars = pd.concat([target_star, bkg_catalog[['sicbroid','xcol','ycol','F146']]])

        imgs = []
        
        i=min(target_mag_timeseries.columns)
        
        for t in tqdm(star.time):

            delta_mags = target_mag_timeseries[i]
            
            new_star_cat = update_stellar_catalog(all_stars, delta_mags, mag_col='F146')


            data_f146 = self.make_realistic_image(oversample=True, bandpass='F146', \
                                                    read_style='ramp', return_err=False, \
                                                    multiaccum_table=multi_accum, \
                                                    star_list=new_star_cat[:,1:],
                                                  trim_psf_kernel=True,  )

            i+=1
            imgs.append(data_f146)
            
        return imgs 
        
    

        
    
    def set_base_target_catalog(self, catalog):

        self.targ_stars = catalog



    def get_timeseries(self, ):

        return 1. 


    def calc_img_cutout(self, ):

        return 1. 


   
