import batman 
import pandas as pd
from tqdm import tqdm


from scipy.interpolate import griddata

from gadfly import StellarOscillatorKernel, Hyperparameters, GaussianProcess, PowerSpectrum

from .utils import *

msun = 1.9891e30
rsun = 695500000.0
G = 6.67384e-11
AU = 149597870700.0
Mbol_sun=4.74




class TimeSeries(object):

    '''
       Base class used to update light curves at a given time. Will have features that are inherited by other
          classes: Seismology, Transit, Rotation
       Includes functions needed to create a light curve, update time series for injection into images
    '''

    def __init__(self, time=None, baseline=None, cadence=None, exposure=None, bandpass=None, t0=0., mag=None):

        if time is None:
            
            self.exposure = exposure * u.s
            self.baseline = baseline * u.day
            self.cadence = cadence * u.minute

            self.time = np.arange(t0, self.baseline.to(u.day).value, self.cadence.to(u.day).value)

        else:
            self.time = time
            
        self.d_flux = np.zeros_like(self.time)
        self.bandpass=bandpass
        self.mag=mag


    def _dflux_to_dmag(self, ):
        return -2.5*np.log10(1.+self.d_flux)




class SeismologyTimeSeries(TimeSeries):


    def set_stellar_parameters(self, Radius, Mass, Teff, Lum=None, Mbol=None, bandpass='W146'):
        
        self.mstar = Mass * u.M_sun
        self.rstar = Radius * u.R_sun
        self.teff = Teff * u.K

        if Lum is None:
            if Mbol is None:
                print('\n.\n.\n.\n.ERROR: MUST SPECIFY EITHER M_bol or LUMINOSITY\n.\n.\n.\n.')
            self.lum = u.L_sun * 10. ** (-0.4 * (Mbol-Mbol_sun) )
            
        else:
            self.lum = Lum * u.L_sun

        self.bandpass = bandpass
        self.seismology_kernel = self._get_seismology_kernel()


    def _get_seismology_kernel(self, i=0):

        hp = Hyperparameters.for_star(
        mass=self.mstar,
        radius=self.rstar,
        temperature=self.teff,
        luminosity=self.lum,
        name = f"star_{i}",
        bandpass="WFIRST/WFI."+self.bandpass )

        # generate a celerite2-compatible kernel
        kernel = StellarOscillatorKernel(hp, texp=self.exposure)

        self.seismology_kernel=kernel
        
        return kernel

    def get_seismology_lightcurve(self, return_units=False):

        t = (self.time * u.day).to(1/u.uHz).value  # gadfly  requires these units    

        try:
            gp = GaussianProcess(self.seismology_kernel,t=t )
        except AttributeError:
            self.seismology_kernel=self._get_seismology_kernel()
            gp = GaussianProcess(self.seismology_kernel,t=t )

        if return_units:
            return gp.sample(return_quantity=False)


        simulated_lc = gp.sample(return_quantity=True).to(u.dimensionless_unscaled).value
        
        self.d_flux = simulated_lc
            
        return simulated_lc



class TransitTimeSeries(TimeSeries):


    def _get_batman(self, ):

        self.model = call_the_bat_signal(self.time)
        self.model_params = batman.TransitParams()
        

    def set_stellar_parameters(self, Radius, Mass, Teff, Fe_H=0., Lum=None, Mbol=None):

        self.rstar=Radius * u.R_sun
        self.mstar=Mass * u.M_sun
        self.teff=Teff * u.K
        self.logg = u.Dex((c.G * self.mstar / self.rstar**2).cgs)
        self.met = u.Dex(Fe_H)

        if Lum is None:
            if Mbol is None:
                print('\n.\n.\n.\n.ERROR: MUST SPECIFY EITHER M_bol or LUMINOSITY\n.\n.\n.\n.')
            self.lum = u.L_sun * 10. ** (-0.4 * (Mbol-Mbol_sun) )
            
        else:
            self.lum = Lum * u.L_sun


    def set_planet_parameters(self, period, planetRadius, t0=None, ecc=0., omega=0., impact=0. ):

        self.rplanet = planetRadius * u.earthRad
        self.orbital_period = period * u.day
        self.ecc=ecc
        self.omega=omega * u.deg
        self.impact=impact
        self.t0=t0


    def _calc_batman_parameters(self):

        self.model_params.rp = (self.rplanet.to(u.R_sun).value / self.rstar.to(u.R_sun).value)
        self.model_params.per = self.orbital_period.value
        self.model_params.t0=self.t0

        a3 = self.orbital_period**2. * c.G * self.mstar / (4. * np.pi**2.)
        
        self.model_params.a = (a3**(1./3.)).to(u.R_sun).value / self.rstar.to(u.R_sun).value

        if self.ecc is None:
            self.ecc=0.
            
        self.model_params.ecc=self.ecc
        self.model_params.w = self.omega.value
        self.model_params.inc = np.rad2deg( np.arccos(self.impact /self.model_params.a) )

        # Update this to interpolate on a grid later
        self.model_params.limb_dark = "nonlinear"        #limb darkening model
        self.model_params.u = self._get_limb_darkening() 


    def _get_limb_darkening(self, limbcoeff_interpolator=None):

        '''
        TODO: Update this so that it gets Limb-Darkening Parameters from the file in data
        '''

        return [1.1233, -1.7547, 1.5448, -0.5273]
    
    def get_transit_lightcurve(self, ):

        self._get_batman()
        self._calc_batman_parameters()

        simulated_lc = self.model.light_curve(self.model_params)-1.
        
        self.d_flux = simulated_lc

        return simulated_lc




class MultiStarTimeSeries(TimeSeries):
    pass


class StellarTimeSeries(TransitTimeSeries,  SeismologyTimeSeries):    
    
    pass


    

    


def get_rp_rstar(rp, rstar):
    return rp / (rstar * 109.076)

def get_a_rstar(per, mstar, rstar):
    per_SI = per * 86400.0
    mass_SI = mstar * msun
    a3 = per_SI ** 2 * G * mass_SI / (4 * np.pi ** 2)
    return a3 ** (1.0 / 3.0) / (rstar * rsun)


def get_lightcurve_parameters(per, rp, stars, limbcoeffs_interp=None):
    
    t0 = np.random.rand(len(per)) * per
    
    sicids = stars['sicbro_id']
    rp_rstar = get_rp_rstar(rp , stars['Radius'] )
    a_rstar = get_a_rstar(per, stars['Mass'], stars['Radius'])

    # e distribution from Van Eylen+2015
    e = np.random.beta(1.03, 13.6, size=len(per))
    w = np.random.rand(len(per))*360. # rnadom number between 0-360

    b = np.random.rand(len(per)) #impact parameter between 0,1

    i = np.rad2deg( np.arccos(b / a_rstar) )
    #i = np.zeros(len(per))+90. # convert b into inclination

    if limbcoeffs_interp is None:
        a1,a2,a3,a4 = 1.1233+np.zeros(len(per)),  -1.7547+np.zeros(len(per)), 1.5448+np.zeros(len(per)), -0.5273+np.zeros(len(per))

    else:
        a1,a2,a3,a4=interpolate_limb_darkening_coeffs(stars['Teff'], stars['logg'], stars['[M/H]'], limbcoeffs_interp)
        
    
    lcparams = np.transpose([sicids, t0, per, rp_rstar, a_rstar, e, w, i, a1, a2, a3, a4])
    
    return lcparams



def interpolate_limb_darkening_coeffs(teff, logg, met, limbcoeffs_interpolator):
    
    input_limbcoeffs = np.array([teff,logg,met]).T
    a1,a2,a3,a4 = limbcoeffs_interpolator(input_limbcoeffs).T
    
    return a1,a2,a3,a4
    


def call_the_bat_signal(t):
    
    params = batman.TransitParams()
    params.t0 = 0.                        #time of inferior conjunction
    params.per = 1.                       #orbital period
    params.rp = 0.1                       #planet radius (in units of stellar radii)
    params.a = 15.                        #semi-major axis (in units of stellar radii)
    params.inc = 90.                      #orbital inclination (in degrees)
    params.ecc = 0.                       #eccentricity
    params.w = 90.                        #longitude of periastron (in degrees)
    params.limb_dark = "nonlinear"        #limb darkening model
    params.u = [1.1233, -1.7547, 1.5448, -0.5273]
    
    if hasattr(t, "__len__"):
        m = batman.TransitModel(params, t, )
    else:
        m = batman.TransitModel(params, np.array([t]))
    
    return m


def get_transit_flux_at_time(m, par):
    
    sicbroid, t0, per, rp_rstar, a_rstar, e, w, i, a1, a2, a3, a4 = par
    
    params = batman.TransitParams()
    params.t0 = t0                        #time of inferior conjunction
    params.per = per                       #orbital period
    params.rp = rp_rstar                       #planet radius (in units of stellar radii)
    params.a = a_rstar                        #semi-major axis (in units of stellar radii)
    params.inc = i                      #orbital inclination (in degrees)
    params.ecc = e                      #eccentricity
    params.w = w                        #longitude of periastron (in degrees)
    params.limb_dark = "nonlinear"        #limb darkening model
    params.u = [a1, a2, a3, a4]
        
    return m.light_curve(params)


def update_catalog_magnitudes(t, stars, lightcurve_parameters):
    
    star_copy = stars.copy()
    mod = call_the_bat_signal(t)
    
    for lp in lightcurve_parameters:
        
        star_id = lp[0]
        
        f = get_transit_flux_at_time(mod,lp) 
        dm = -2.5 * np.log10(f)
        
        star_copy[(star_id==stars[:,0]).nonzero()[0],3] += dm
    
    return star_copy



def calculate_lightcurves(lc_params, tobs=72., cadence=0.0104167):
    
    times = np.arange(0, tobs, cadence)
    mod = call_the_bat_signal(times)
    
    delta_mags = []
    for lp in lc_params:
        
        f = get_transit_flux_at_time(mod,lp) 
        dm = -2.5 * np.log10(f)
        delta_mags.append(dm)
        
    return pd.DataFrame(np.array(delta_mags), index=lc_params[:,0].astype(int))




def update_stellar_catalog(stellar_catalog, delta_mag, mag_col='mag'):
    
    new_starsdf = stellar_catalog.copy()
    
    new_starsdf[mag_col] = stellar_catalog[mag_col].add(delta_mag, fill_value=0.)
    newstars_numpy =  np.append(np.vstack(new_starsdf.index.to_numpy()),
                                new_starsdf.to_numpy(), axis=1 )

    return newstars_numpy









    
        
    




