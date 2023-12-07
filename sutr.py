import numpy as np


def get_ramp_samples(expected_flux, nreads, readtime=3.04, read_noise=11., 
                     saturation_limit=1e5):
    
    delta_flux = [np.random.poisson(expected_flux*readtime) for i in range(nreads)]
    reads = np.cumsum(delta_flux, axis=0).astype(float)
    reads += np.random.normal(scale=read_noise, size=reads.shape)
    
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


def measure_rate_slopes(resultant_frames,multiaccum_table,saturation_limit=1e5,read_noise=11.,frametime=3.04,bias=1e3):

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
            

            if n_good_reads>=2:
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
                #m, c = np.linalg.solve( Xw.T.dot(Xw), Xw.T.dot(np.vstack(fw) ) )
                    
                rate=m
                    
                    
            #elif n_good_reads==2:
            #    rate = (accumulated_charge[1]-accumulated_charge[0])/(group_times[1]-group_times[0])
                            
            elif n_good_reads==1:
                rate=(accumulated_charge[0])/group_times[0]
                
            else:
                rate = np.nan

            rate_slope_image[xi,yi]=rate

                #if not(rate>10):
                #    print(accumulated_charge[1:]-accumulated_charge[:-1], 3.04*rate)


    return rate_slope_image



def cas22_fit_ramps(resultant_frames, multiaccum, saturation_limit,
                       read_noise, frame_time, ):
    
    nframes, n_x, n_y = resultant_frames.shape
    slope_means = np.zeros((n_x,n_y))
    slope_variances = slope_means.copy()
    
    t_group = np.array( get_group_times(multiaccum,readtime=frame_time) )
    tau_group = np.array( cas22_variance_time_tau(multiaccum, frame_time) )

    for ix in range(n_x):
        for iy in range(n_y):

            R_i = resultant_frames[:,ix, iy]
            sat_mask = R_i<saturation_limit

            if sum(sat_mask)==1:
                slope = R_i[0]/t_group[0]
                var = slope + read_noise**2.

            elif sum(sat_mask)==0:
                slope = np.nan
                var = np.inf

            else:
                N_i = multiaccum[sat_mask]

                slope, var = cas22_slope_mean_and_variance(R_i[sat_mask], N_i,t_group[sat_mask], tau_group[sat_mask], read_noise=read_noise)
                #slope=cas22_slope_mean_and_variance(R_i[sat_mask], N_i,t_group[sat_mask], tau_group[sat_mask], read_noise=read_noise)
                
            slope_means[ix,iy] = slope
            slope_variances[ix,iy] = var
                
            
    return slope_means , slope_variances



def cas22_variance_time_tau(multiaccum, frame_time=3.04):

    #n_reads = sum(multiaccum)


    tau_i = [np.sum([ (2*(N_i-k)-1)*(frame_time*(1+k+sum(multiaccum[:i])))/N_i**2. for k in range(N_i) ]) for i,N_i in enumerate(multiaccum)]

    
    # Equation 14 from Casertano+2022
    #tau = np.zeros_like(multiaccum, dtype=np.float32)

    #ik = 1
    
    #for i,N_i in enumerate(multiaccum):

    #    for k in range(N_i):

    #        t_ik = ik * frame_time
    #        tau[i] += (2.*(N_i-k)-1.)*t_ik/N_i**2.

    #        ik+=1
            

        #t_i = np.sum(multiaccum[:i+1])*frame_time
        
        #tau[i] = np.sum( [ for k in range(N_i)]  )

        #print([(2.*(N_i-k)-1.)*(t_i+k*frame_time)/N_i**2. for k in range(N_i)] )

    #print(tau_i, tau)
    return tau_i

    

def cas22_weights(R_i, N_i, t_group, read_noise):

    S_max = R_i[-1] #np.nanmax(R_i, axis=0)
    s = S_max / np.sqrt(read_noise**2. + S_max)

    #P = np.zeros_like(s)

    t_mid = (t_group[-1]+t_group[0])/2.


    if s<5:
        P=0
    elif s<10:
        P=0.4
    elif s<20:
        P=1.
    elif s<50:
        P=3.
    elif s<100:
        P=6.
    else:
        P=10.

    return S_max, np.sqrt( ((1.+P)*N_i)/(1.+P*N_i) * np.abs(t_group-t_mid)**P )
                    


    
    #P[np.logical_and(s>=5, s<10)] = 0.4
    #P[np.logical_and(s>=10, s<20)] = 1
    #P[np.logical_and(s>=20, s<50)] = 3
    #P[np.logical_and(s>=50, s<100)] = 6
    #P[s>=100] = 10
    #return w_i 


def cas22_slope_mean_and_variance(R_i, N_i, t_group, tau_group, read_noise=11.):

    '''

    Calculates Noise using algorithm from Casertano+2022:
    https://www.stsci.edu/files/live/sites/www/files/home/roman/_documents/Roman-STScI-000394_DeterminingTheBestFittingSlope.pdf
    
    '''

    S_max, w_i = cas22_weights(R_i, N_i, t_group, read_noise)

    F0=np.sum(w_i)
    F1=np.sum(w_i*t_group)
    F2=np.sum(w_i*t_group**2.)
    
    D = F2*F0 - F1**2.
    K = (F0*t_group - F1) * w_i/D

    slope = np.sum(K*R_i)

    n_reads = len(N_i)

    # Equations 
    V_r = np.sum(K**2.*read_noise**2./N_i)

    #print(R_i[-1], R_i[0], S_max)

    
    V_s = np.sum(K**2.*tau_group) + np.sum([2*K[i]*K[j]*t_group[i] for i in range(n_reads) for j in range(n_reads) if i<j ])

    #V_s_new=np.sum(K**2. * tau_group) + 2. * np.sum( [ K[i] * K[j]* t_group[i] for i in range(n_reads-1) for j in range(i, n_reads)  ] )
    
    #for i in range()

    #print(V_s, V_s_new, S_max)
       
    variance = V_r + V_s * slope #S_max/t_group[-1]
    
    
    return slope , variance

