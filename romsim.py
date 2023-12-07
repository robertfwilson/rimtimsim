#from tqdm.notebook import tqdm
import numpy as np
from scipy.signal import fftconvolve, oaconvolve, convolve2d
#from scipy.interpolate import LinearNDInterpolator

#from sklearn.linear_model import LinearRegression

#import datetime
from astropy.io import fits
from astropy.time import Time, TimeDelta


from .wfi import WFI_Properties
from .sutr import *
from .utils import *

import os
ROMSIM_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))








class RomanImage(object):
    
    def __init__(self,stars=None,bandpass='F146',detector='SCA07',
                 star_type='M0V',
                 exptime=54.,nstars=300, detector_size=(4096,4096),
                 psf_fits=None, n_min_zodiacal_background=5.,
                 buffer=4, use_oversampled=False, subframe=False,
                 coords=('17:45:41.01','-29:56:10.13') ):
        
        #if psf_fits is None:
        #    self.psf_fits = fits.open(psf_datadir+'wim_psf_'+star_type.lower()+'_fullset/WFI_PSF_F146_'+detector+'_x2048_y2048_j008mas_'+star_type.upper()+'_o.fits')
        #else:
        #    self.psf_fits=fits.open(psf_fits)


        self.wfiprops = WFI_Properties()
        
        self.coords = coords
        self.bandpass = bandpass
        self.detector=detector
        self.subframe = subframe
        self.n_min_zodiacal_background = n_min_zodiacal_background

        self._get_psf_model(fname=psf_fits)
            
        #self.psf_model = self.psf_fits[0].data
        #self.prf_model = self.psf_fits[1].data
        
        self.psf_oversample_factor = self.psf_model.shape[0]//self.prf_model.shape[0]
        self.detector_size=detector_size
        
        self.blank_image = np.zeros(np.array(self.detector_size))
        
        if stars is None:
            self.stars = self._make_star_list(nstars)
        else:
            self.stars=stars

        if not(subframe==False):
            star_cut = self.stars[:,1]<subframe[0]
            star_cut&= self.stars[:,2]<subframe[1]
            self.stars = self.stars[star_cut]
        else:
            star_cut = self.stars[:,1]<detector_size[0]
            star_cut&= self.stars[:,2]<detector_size[1]
            self.stars = self.stars[star_cut]
            


    def _get_psf_model(self, fname=None, spectype='M0V', bandpass='F146'):

        if fname is None:
            fname = ROMSIM_PACKAGE_DIR + '/psf_models/'+bandpass+'/wfi_psf_'+spectype+'_'+bandpass+'_'+self.detector+'.fits'

        hdul = fits.open(fname)
        self.psf_fits = hdul
        self.psf_model = self.psf_fits[0].data
        self.prf_model = self.psf_fits[1].data

        self.psf_model/=np.sum(self.psf_model)
        self.prf_model/=np.sum(self.prf_model)
        
        hdul.close()
            
            
    def _make_star_list(self, nstars, mag_lim = [7,28]):
        
        xpos = np.random.rand(nstars) * self.detector_size[0]
        ypos = np.random.rand(nstars) * self.detector_size[1]
        mags = np.random.rand(nstars) * (mag_lim[1]-mag_lim[0]) + mag_lim[0]
        
        return np.transpose([np.argsort(xpos),xpos,ypos,mags])
    
    
    
    def _convert_mag_to_electrons(self, mag, exptime=1.):
        zeropoint=self.wfiprops.zeropoint_mag
        return exptime * 10.**(-0.4 * (mag-zeropoint) )
    
    
    def _get_read_noise(self, image):
        readnoise = self.wfiprops.read_noise
        return np.random.standard_normal(image.shape ) * readnoise

    def _get_poisson_noise(self, expectation, shape=None):
        return np.random.poisson(expectation, size=shape).astype(float)
    
    def _get_dark_noise(self, image, exptime=1.):
        darknoise = self.wfiprops.dark_current
        return np.random.poisson(darknoise*exptime+np.zeros_like(image) )


    def _get_sky_background(self, image, exptime):
        sky_expectation = self.n_min_zodiacal_background * self.wfiprops.minzodi_background*exptime
        return self._get_poisson_noise(sky_expectation, shape=image.shape)
    
    def _get_thermal_background(self, image, exptime):
        return self._get_poisson_noise(self.wfiprops.thermal_background*exptime, shape=image.shape)

    def _add_all_noise(self, expected_source_flux, exptime, read=True):

        image = self._get_poisson_noise(expected_source_flux * exptime,)
        image += self._get_dark_noise(image, exptime=exptime)
        image += self._get_sky_background(image, exptime=exptime)
        image += self._get_thermal_background(image, exptime=exptime)

        if read:
            image += self._get_read_noise(image, )

        return image
        


    def _make_expected_source_image(self, oversample=False, star_list=None, trim_psf_kernel=True, include_sky=True):

        if not(self.subframe == False):
            detector_size = self.subframe
        else:
            detector_size = self.detector_size

        
        if oversample:
            image = np.zeros((detector_size[0]*self.psf_oversample_factor, \
                              detector_size[1]*self.psf_oversample_factor) )
            
            kernel = self.psf_model

            if trim_psf_kernel:
                dx,dy = image.shape

                mid_x, mid_y = kernel.shape[0]//2, kernel.shape[1]//2
                kernel = kernel[mid_x-dx:mid_x+dx,mid_y-dy:mid_y+dy]
                
        
        else:
            image = np.zeros(detector_size)
            kernel = self.prf_model
        
        if star_list is None:
            star_list = self.stars
            
        #print('... Adding Stars ...')
        nstar=0
        for i,s in enumerate(star_list):
            _,x0,y0,m = s

            #if x0<self.detector_size[0] and y0<self.detector_size[0]:
            
            f = self._convert_mag_to_electrons(m,)
            
            if oversample:
                x0*=self.psf_oversample_factor
                y0*=self.psf_oversample_factor
            
            image[int(x0),int(y0)] += f
                
            nstar+=1
        
        #print('... Added {} Stars ...'.format(nstar))
        #print('... Performing PSF Convolution ... ')

        expected_source_image = fftconvolve(image, kernel, mode='same')

        if include_sky:
            sky_expectation = self.n_min_zodiacal_background * self.wfiprops.minzodi_background
            sky_expectation += self.wfiprops.thermal_background
        else:
            sky_expectation=0.
        
        if oversample:
            
            return bin_image(expected_source_image, self.psf_oversample_factor) + sky_expectation
        else:
            return expected_source_image + sky_expectation



        '''
    def _make_frames_ramp_sampling(self, expected_source_flux, n_reads):

        
        expected_flux_image: nd array of same shape as image
        The expected number of electrons for 1 sec of integration
        
        
        
        # add small number to avoid computer rounding giving negative numbers
        expected_source_flux += 1e-10
        readout_time = self.wfiprops.readout_time

        saturation_limit = self.wfiprops.saturation_limit# / readout_time      

        #first_frame = self._get_poisson_noise(expected_source_per_read,shape=None)
        #first_frame += self.wfiprops.bias
        #first_frame += self.add_dark_noise(first_frame, exptime=self.wfiprops.readout_time)
        #first_frame += self.add_read_noise(first_frame, exptime=self.wfiprops.readout_time)

        first_frame = self._add_all_noise(expected_source_flux, exptime=readout_time, read=False)
        first_frame += self.wfiprops.bias
        first_frame = self._quantize(first_frame)
        
        
        saturation_mask = first_frame >= saturation_limit
        first_frame[saturation_mask] = saturation_limit
        
        frames = [first_frame]

        
        for n in range(1,n_reads):

            delta_flux = self._add_all_noise(expected_source_flux, exptime=readout_time, read=False)
            delta_flux = self._quantize(delta_flux)
            
            saturation_mask &= frames[-1]+delta_flux >= saturation_limit
            delta_flux[saturation_mask] = 0

            nth_frame = frames[-1] + delta_flux
            nth_frame[saturation_mask]=saturation_limit
            
            frames.append(nth_frame)

        for f in frames:
            saturation_mask = f>=saturation_limit
            f+=self._get_read_noise(f)
            f[saturation_mask] = np.nan
            
            
        frames = np.array(frames)
        #difference_frames = frames[1:]-frames[:-1]

        #frame_times = (np.arange(0, n_reads)+1)*readout_time
        
        return frames
        

        
    def _create_resultant_frames(self, frames, ngroup=2, n_resultant_frames=None):



        resultants = combine_read_frames_into_resultant_frames(read_frames, multiaccum_table,
                                                               saturation_limit=self.saturation_limit)

        
        if n_resultant_frames is None:
            n_resultant_frames = frames.shape[0]//ngroup


        remainder = len(frames)%ngroup
        frame_times = np.arange(1, len(frames)+1) * self.wfiprops.readout_time

        resultant_frames = []
        group_times = []

        #difference_frames=frames[1:]-frames[:-1]
        
        for n in range(remainder, len(frames), ngroup):

            group_frames=frames[n:n+ngroup,:,:]

            #sat_masks = [gf>=self.wfiprops.saturation_limit for gf in group_frames]

            # works for ngroup==2, but should generalize this later
            #partially_saturated = np.logical_and(sat_masks[0]<self.wfiprops.saturation_limit,
            #                                     sat_masks[1]>=self.wfiprops.saturation_limit)

            #group_frames[1][partially_saturated] = group_frames[0][partially_saturated]+difference_frames[n][partially_saturated]
            
            
            resultant_frames.append(np.mean(group_frames,axis=0))
            group_times.append(np.mean(frame_times[n:n+ngroup]))


        if remainder>0:
            for n in range(remainder):
                resultant_frames.insert(n, frames[n,:,:])
                group_times.insert(n, frame_times[n])
       
                
                    

        return np.array(resultants), np.array(group_times)
       '''

    '''
    def _measure_rate_slopes(self, resultant_frames, group_times):

        
        nframes, n_x, n_y = resultant_frames.shape

        rate_slope_image = np.zeros((n_x,n_y))


        #group_exptime = self.wfiprops.readout_time * ngroup
        #group_times = (np.arange(nframes) + 0.5)*group_exptime

        i=np.arange(nframes)+1
        
        for xi in range(n_x):
            for yi in range(n_y):

                accumulated_charge = resultant_frames[:,xi,yi]

                # detect saturation
                not_saturated = accumulated_charge<self.wfiprops.saturation_limit


                if sum(not_saturated)>2:
                    t = group_times[not_saturated]
                    f = accumulated_charge[not_saturated]

                    #rate = (f[-1]-f[0])/(t[-1]-t[0])
                    
                    S = f[-1] / np.sqrt(self.wfiprops.read_noise**2. + f[-1])

                    if S<5:
                        P=0.
                    elif S<10:
                        P=0.4
                    elif S<20:
                        P=1
                    elif S<50:
                        P=3
                    elif S<100:
                        P=6
                    else:
                        P=10
                    
                    i_mid=len(t)/2
                    w = (i[not_saturated]-i_mid)**P
                    
                    #rate_0 = (f[-1]-f[0])/(t[-1]-t[0])
                    #p0 = rate_0, 0
                    

                    X=np.vstack([t, np.ones(len(t))]).T

                    Xw = X * w[:, None]
                    fw = f * w
                    m, c = np.linalg.lstsq(Xw, fw, rcond=None)[0]
                    
                    rate=m
                    
                    
                elif sum(not_saturated)==2:
                    rate = (accumulated_charge[1]-accumulated_charge[0])/(group_times[1]-group_times[0])
                    
                #elif sum(not_saturated)==1:

                #    rate = (accumulated_charge[0]-self.wfiprops.bias)/self.wfiprops.readout_time

                elif sum(not_saturated)<=1:
                    rate = np.nan

                rate_slope_image[xi,yi]=rate

                #if not(rate>10):
                #    print(accumulated_charge[1:]-accumulated_charge[:-1], 3.04*rate)

        

        rate_slope_image = measure_rate_slopes(resultant_frames, multiaccum_table, saturation_limit, read_noise)

        
        return rate_slope_image
    '''
    
    
    def _make_stellar_image(self, bandpass, exptime, star_list=None,
                            oversample=True, multiaccum_table=None,read_style='ramp',trim_psf_kernel=False):


        self.wfiprops.get_bandpass_properties(bandpass)
        readtime = self.wfiprops.readout_time

        expected_source_flux = self._make_expected_source_image(oversample=oversample, star_list=star_list,trim_psf_kernel=trim_psf_kernel)
            
        saturation_limit = self.wfiprops.saturation_limit
        read_noise = self.wfiprops.read_noise

        
        if read_style=='ramp':
            
            n_reads = sum(multiaccum_table)
            #n_resultant_frames = 6
            #n_group = int(n_reads//n_resultant_frames)
            

            #print('... Simulating Individual Frames ...')
            #simulated_frames = self._make_frames_ramp_sampling(expected_source_flux, n_reads=n_reads)

            simulated_frames = get_ramp_samples(expected_source_flux, n_reads, readtime,
                                                read_noise=read_noise,
                                                saturation_limit=saturation_limit)

            
            #resultant_frames, group_times = self._create_resultant_frames(simulated_frames, ngroup=ngroup, n_resultant_frames=None)


            resultant_frames = combine_read_frames_into_resultant_frames(simulated_frames,
                                                                         multiaccum_table,
                                                                         saturation_limit=saturation_limit)

            #print('... Measuring Count Rate Slopes ...')
            rate_slope_image = measure_rate_slopes(resultant_frames, multiaccum_table,
                                                   saturation_limit=saturation_limit,
                                                   read_noise=read_noise,
                                                   frametime=readtime, )
            
            saturation_mask = np.isnan(rate_slope_image)

            return rate_slope_image, saturation_mask
        
            #source_image = rate_slope_image
            #source_image[saturation_mask] = np.nan #self.wfiprops.saturation_limit#/self.wfiprops.readout_time            

        elif read_style=='cas22_ramp':

            n_reads = sum(multiaccum_table)
            #n_resultant_frames = 6
            #n_group = int(n_reads//n_resultant_frames)
            

            #print('... Simulating Individual Frames ...')

            simulated_frames = get_ramp_samples(expected_source_flux, n_reads, readtime,
                                                read_noise=read_noise,
                                                saturation_limit=saturation_limit)

            resultant_frames = combine_read_frames_into_resultant_frames(simulated_frames,
                                                                         np.array(multiaccum_table),
                                                                         saturation_limit=saturation_limit)

            slope, var = cas22_fit_ramps(resultant_frames, np.array(multiaccum_table),
                                              saturation_limit=saturation_limit,
                                              read_noise=read_noise, frame_time=readtime)

            return slope, var
                
        elif read_style=='ccd':
                        
            source_image = expected_source_flux

            pixel_saturation_time = self.wfiprops.saturation_limit/(expected_source_flux/readtime) 
            
            saturation_mask = source_image>self.wfiprops.saturation_limit*readtime
            source_image[saturation_mask] = self.wfiprops.saturation_limit
            
            
        else:
            print('read_style must be either \'ccd\' or \'fowler\' or \'ramp\' ')
       
        
        return source_image, saturation_mask
    
    
    def _quantize(self, image):
        return np.round(image, 0)

    
    
#    def _get_wcs(self, obs_time='2032-09-18T00:00:00.123'):
#        
#        ra_targ = galsim.Angle.from_hms(self.coords[0])
#        dec_targ = galsim.Angle.from_dms(self.coords[1])
#        targ_pos = galsim.CelestialCoord(ra=ra_targ, dec=dec_targ)
#        pa = galsim.Angle.from_dms('34:00:00')
        
#        date = datetime.datetime(2032, 8, 16)

#        wcs_dict = roman.getWCS(targ_pos, PA=pa, date=None, SCAs=None, PA_is_FPA=True)
        
#        return wcs_dict
    
    
    def make_header(self, exptime,bandpass,obs_time='2026-09-18T00:00:00.123', 
                    timeformat='isot',):
        
        #wcs_dict = self._get_wcs(obs_time=obs_time)
        #hdr = wcs_dict[int(self.detector[-2:])].header.header

        hdr = fits.Header()
        
        texp = TimeDelta(exptime, format='sec')
        t = Time(obs_time, format=timeformat)
        
        hdr['FILTER'] =   (bandpass, 'filter used')
        hdr['NCOL'] = (self.subframe[0], 'number of columns in image')
        hdr['NROW'] = (self.subframe[1], 'number of rows in image')
        hdr['DETECTOR'] = (self.detector, 'detector assembly')
        
        hdr['TSTART'] = (t.to_value('jd'), 'observation start in Julian Date')
        hdr['TEND'] = ((t+texp).to_value('jd'), 'observation end in Julian Date')
        hdr['DATE-OBS'] = (t.to_value('isot'), 'observation start in UTC Calendar Date')
        hdr['DATE-END'] = ((t+texp).to_value('isot'), 'observation end in UTC Calendar Date')
        hdr['EXPOSURE'] = (texp.to_value('sec'), 'time on source in s')
        
        return hdr
    
    
    
    def make_realistic_image(self, bandpass, exptime=None, oversample=False, star_list=None, read_style='ramp', return_err=False, multiaccum_table=None,trim_psf_kernel=False):


        if bandpass != self.wfiprops.bandpass:
            
            self.wfiprops = self.wfiprops.get_bandpass_properties(bandpass)
            self._get_psf_model(bandpass=bandpass)

        if star_list is None:
            star_list = self.stars

        else:
            if star_list.shape[1]!=4:
                print('star_list MUST HAVE FORM [id, x, y, mag]')
            else:
                if not(self.subframe==False):
                    star_cut = star_list[:,1]<self.subframe[0]
                    star_cut &= star_list[:,2]<self.subframe[1]
                    self.stars = star_list[star_cut]
                else:
                    self.stars = star_list

        
        image, saturation_mask = self._make_stellar_image(bandpass,exptime,oversample=oversample, star_list=None, read_style=read_style,multiaccum_table=multiaccum_table,trim_psf_kernel=trim_psf_kernel)

        if read_style=='cas22_ramp':

            if return_err:
                return image, saturation_mask
            else:
                return image
            

        if read_style=='ramp':
            
            if return_err:

                t_read=self.wfiprops.readout_time
                ngroups = exptime//(2*t_read)
                read_noise=self.wfiprops.read_noise
                
                sat_limit=self.wfiprops.saturation_limit
                
                
                err = ramp_readout_noise_model(image, ngroups=ngroups,
                                               n_per_group=2, t_frame=t_read,
                                               sat_limit=sat_limit,
                                               read_noise=read_noise)

                err[saturation_mask] = np.inf
                return image, err
            
            return image
            
            
        image += self._add_all_noise(image, exptime)
        image_w_detector_effects = self._quantize(image)

        
        return image_w_detector_effects
    
    
    
    def create_fits_image(self, data_dir, bandpass, exptime, obs_time, multiaccum_table=None, oversample=True, star_list=None, read_style='ramp', return_err=False, time_format='isot', overwrite_fits=False, frameno=None, ):
        
        if read_style == 'ramp':
            exptime = sum(multiaccum_table) * self.wfiprops.readout_time
            
        data = self.make_realistic_image(bandpass=bandpass, exptime=exptime, oversample=oversample,
                                         star_list=star_list, read_style=read_style,
                                         return_err=return_err, multiaccum_table=multiaccum_table)
        
        if return_err:
            img,img_err = data
        else:
            img = data
        
        hdr = self.make_header(exptime, bandpass, obs_time, time_format )
        
        
        hdr['READMODE'] = read_style
        
        if read_style=='ramp':
            flux_unit = 'electrons per second'
        else:
            flux_unit = 'electrons'
            
        hdr['UNIT'] = flux_unit
        
        
        hdu = [fits.PrimaryHDU(img, header=hdr)]
        
        if return_err:
            hdu.append( fits.ImageHDU(img_err) )
            
        hdul = fits.HDUList( hdu )
        
        timestring = Time(obs_time, format=time_format).to_value('isot', subfmt='date_hm')
        
        timestring = timestring.replace(':','').replace('-','')
        
        
        fits_filename = data_dir + 'romsim_'+bandpass+'_'+self.detector+'_field01_'+read_style+'_subframe_{:04}x{:04}_'.format(
            self.subframe[0],self.subframe[1])+timestring+'_sim.fits'
        
        hdul.writeto(fits_filename , overwrite=overwrite_fits)
        hdul.close()
        
        return img




def ramp_readout_noise_model(f, ngroups, n_per_group, t_frame=3.04,
                             read_noise=11.,n_pixel=1, noise_floor=5.,
                             sat_limit=1e5):
    
    n = ngroups + np.zeros_like(f)
    m = n_per_group + np.zeros_like(f)
    tg = t_frame * n_per_group
    
    
    read_noise_term = (12*(n-1))/(m*n*(n+1)) * read_noise**2.
    photon_noise_term = (6.*(n*n+1)*(n-1))/(5.*n*(n+1)) * tg * f * n_pixel
    correlated_noise_term = (2*(m*m-1)*(n-1))/(m*n*(n+1)) * t_frame * f * n_pixel
    
    
    noise_tot = read_noise_term + photon_noise_term + correlated_noise_term + noise_floor**2. #+ saturation_floor
    
    return noise_tot/(tg*n) 


