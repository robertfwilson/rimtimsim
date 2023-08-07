import numpy as np





def get_ramp_samples(expected_flux, nreads, readtime=3.04, read_noise=10., 
                     saturation_limit=1e5):
    
    delta_flux = [np.random.poisson(expected_flux*readtime) for i in range(nreads)]
    reads = np.cumsum(delta_flux, axis=0).astype(float)
    reads += np.random.normal(nreads)*read_noise
    
    reads[reads>saturation_limit]=saturation_limit
    
    return np.round(reads,0)


def combine_read_frames_into_resultant_frames(read_frames,multiaccum_table,saturation_limit=1e5):
    
    #group_times=np.arange(1, len(read_frames)+1) * readtime
    group_frame_fluxes = []
    
    for groupnum, nframes in enumerate(multiaccum_table):
        
        last_frame = sum(multiaccum_table[:groupnum+1])
        first_frame = sum(multiaccum_table[:groupnum])
        
        frames = read_frames[first_frame:last_frame]
        
        frames[frames>=saturation_limit] = 1e10
        
        group_avg_flux = np.sum(frames,axis=0) 
        group_avg_flux/=nframes
                
        group_frame_fluxes.append(group_avg_flux)
    
    return np.array(group_frame_fluxes)


def get_group_times(multiaccum_table, readtime=3.04):
    
    frame_times = np.arange(1, sum(multiaccum_table)+1) * readtime
    
    group_times=[]
    
    for groupnum, nframes in enumerate(multiaccum_table):
        last_frame = sum(multiaccum_table[:groupnum+1])
        first_frame = sum(multiaccum_table[:groupnum])
            
        group_times.append(np.mean(frame_times[first_frame:last_frame]))

    return np.array(group_times)


def measure_rate_slopes(resultant_frames,multiaccum_table,saturation_limit=1e5,read_noise=11.,
                        frametime=3.04):

    nframes, n_x, n_y = resultant_frames.shape

    rate_slope_image = np.zeros((n_x,n_y))
    
    group_times = get_group_times(multiaccum_table,readtime=frametime)

    i=np.arange(nframes)+1
        
    for xi in range(n_x):
        for yi in range(n_y):

            accumulated_charge = resultant_frames[:,xi,yi]

            # detect saturation
            not_saturated = accumulated_charge<saturation_limit
            n_good_reads = sum(not_saturated)
            

            if n_good_reads>2:
                t = group_times[not_saturated]
                f = accumulated_charge[not_saturated]
                N_i = np.array(multiaccum_table)[not_saturated]
                S = f[-1] / np.sqrt(read_noise**2. + f[-1])
                
                if S<5:
                    P=0.
                elif S<10:
                    P=0.4
                elif S<20:
                    P=1.
                elif S<50:
                    P=2.
                elif S<100:
                    P=6.
                else:
                    P=10.
                    
                i_mid=len(t)/2
                t_mid = (t[-1]+t[0])/2.
                
                #w = np.abs(i[not_saturated]-i_mid)**(P)
                
                # weights for unevenly-sampled resultants from Casertano+2022
                w = np.sqrt( ((1.+P)*N_i)/(1.+P*N_i) * np.abs(t-t_mid)**P )
                    
                    #rate_0 = (f[-1]-f[0])/(t[-1]-t[0])
                    #p0 = rate_0, 0                    

                X=np.vstack([t, np.ones(len(t))]).T
                Xw = X * w[:, None] 
                fw = f *  w 
                
                #W = np.diag(w)
                #Xw = np.dot(W,X)
                #fw = np.dot(f,W)
                
                #print(Xw, fw)
                
                m, c = np.linalg.lstsq(Xw, fw, rcond=None)[0]
                    
                rate=m
                    
                    
            elif n_good_reads==2:
                rate = (accumulated_charge[1]-accumulated_charge[0])/(group_times[1]-group_times[0])
            
            elif n_good_reads==1:
                rate=accumulated_charge[0]/group_times[0]
                
            else:
                rate = np.nan

            rate_slope_image[xi,yi]=rate

                #if not(rate>10):
                #    print(accumulated_charge[1:]-accumulated_charge[:-1], 3.04*rate)


    return rate_slope_image


