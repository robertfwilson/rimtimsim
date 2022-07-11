import numpy as np
import batman 
import pandas as pd
from tqdm import tqdm

msun = 1.9891e30
rsun = 695500000.0
G = 6.67384e-11
AU = 149597870700.0


def get_rp_rstar(rp, rstar):
    return rp / (rstar * 109.076)

def get_a_rstar(per, mstar, rstar):
    per_SI = per * 86400.0
    mass_SI = mstar * msun
    a3 = per_SI ** 2 * G * mass_SI / (4 * np.pi ** 2)
    return a3 ** (1.0 / 3.0) / (rstar * rsun)


def get_lightcurve_parameters(per, rp, Stars_w_transits):
    
    t0 = np.random.rand(len(per)) * per
    
    sicids = Stars_w_transits['sicbro_id']
    rp_rstar = get_rp_rstar(rp , Stars_w_transits['Radius'] )
    a_rstar = get_a_rstar(per, Stars_w_transits['Mass'], Stars_w_transits['Radius'])
    
    e = np.zeros(len(per))
    w = np.zeros(len(per))
    i = np.zeros(len(per))+90.
    a1,a2,a3,a4 = 1.1233+np.zeros(len(per)),  -1.7547+np.zeros(len(per)), 1.5448+np.zeros(len(per)), -0.5273+np.zeros(len(per))
    
    lcparams = np.transpose([sicids, t0, per, rp_rstar, a_rstar, e, w, i, a1, a2, a3, a4])
    
    return lcparams


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
    for lp in tqdm(lc_params):
        
        f = get_transit_flux_at_time(mod,lp) 
        dm = -2.5 * np.log10(f)
        delta_mags.append(dm)
        
    return pd.DataFrame(np.array(delta_mags), index=lc_params[:,0].astype(int))




def update_stellar_catalog(stellar_catalog, delta_mag):
    
    new_starsdf = stellar_catalog.copy()
    
    new_starsdf['mag'] = stellar_catalog['mag'].add(delta_mag, fill_value=0.)
    newstars_numpy =  np.append(np.vstack(new_starsdf.index.to_numpy()),
                                new_starsdf.to_numpy(), axis=1 )

    return newstars_numpy



def write_fits_file():
    
    
    return 1.





    
        
    




