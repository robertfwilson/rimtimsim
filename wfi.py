from astropy import units as u
import numpy as np

wfi_filters = 'F062 F087 F106 F129 F158 F184 F213 F146'.split()
wfi_filters_alt = 'R062 Z087 Y106 J129 H158 F184 K213 W146'.split()


wfi_minzodi_background = {'F062':0.27,'F087':0.27, 'F106':0.29, 'F129':0.28, 'F158':0.26, 'F184':0.15, 'F213':0.13, 'F146':0.85}

wfi_thermal_background={'F062':0.,'F087':0., 'F106':0., 'F129':0., 'F158':0.04, 'F184':0.17, 'F213':4.52, 'F146':0.98}

# NOTE: F213 ZEROPOINT IS MADE UP, DO NOT TRUST IT
wfi_filter_zeropoint_mags ={'F062':26.559,'F087':26.413, 'F106':26.435, 'F129':26.403, 'F158':26.433, 'F184':25.954, 'F213':25.454, 'F146':27.648}


# Define a class to hold relevant properties for convenience
class WFI_Properties(object):

    def __init__(self, bandpass=None):
        
        self.dark_current = 0.005
        self.saturation_limit=1e5
        self.readout_time = 3.04
        self.cds_read_noise = 16.
        self.read_noise = 11.
        self.pixel_scale = 0.11 #arcsec/pixel
        self.bias=1e3
        self.detector_size=(4096,4096)
        self.interpixel_capacitance = np.array([[0,0,0], [0,1,0], [0,0,0]])
        self.bandpass=bandpass


        if bandpass is None:
            self.thermal_background = None
            self.zeropoint_mag = None
            self.minzodi_background = None
        else:
            self.get_bandpass_properties(bandpass)


    def get_bandpass_properties(self, bandpass):

        if not(bandpass in wfi_filters):
            if not(bandpass in wfi_filters_alt):
                print('BANDPASS MUST BE ONE OF {}'.format(wfi_filters))
                print('BANDPASS SPECIFIC PROPERTIES REMAIN UNDEFINED\n')
                return self
                
            else:
                bandpass = wfi_filters[wfi_filters_alt.index(bandpass)]

        if bandpass=='F213':
            print('WARNING: F213 ZEROPOINT IS NOT TRUSTED. USE AT YOUR OWN PERIL. \n\n')

        self.zeropoint_mag = wfi_filter_zeropoint_mags[bandpass]
        self.thermal_background = wfi_thermal_background[bandpass]
        self.minzodi_background = wfi_minzodi_background[bandpass]
        self.bandpass=bandpass

        return self


    

    
    

    

    

