import numpy as np

import astropy.units as u
import astropy.constants as c
from astropy.time import Time


import os

RIMTIMSIM_DIRECTORY = os.path.dirname(os.path.realpath(__file__))


def matrix_solve(x, y, y_err=None, ):

    A = np.vstack([x.ravel()]).T
    if y_err is None:
        A_0 = A
        y_0 = np.vstack(y.ravel())
    else:
        A_0 = np.vstack((1./y_err).ravel()) * A
        y_0 = np.vstack( y.ravel() ) * np.vstack((1./y_err).ravel())
        
    w_i = np.linalg.solve(A_0.T.dot(A_0), A_0.T.dot( y_0 ) , )

    # Get the uncertainties
    dof = y.ravel().shape[0] - w_i.ravel().shape[0]
    covar=np.linalg.inv(A_0.T.dot(A_0))
    resids = (y_0 - A_0.dot(w_i))
    sigma2 = np.sum(resids**2.)/dof

    std_err = np.sqrt(np.diag(sigma2*covar))

    return w_i.ravel(), std_err.ravel() 

    
    
    

def get_uncertainty(x, y, w_i, y_err=None, ):

    A = np.vstack([x.ravel()]).T
    dof = y.ravel().shape[0] - w_i.ravel().shape[0]

    if y_err is None:
        A_0 = A
        y_0 = np.vstack(y.ravel())
    else:
        A_0 = np.vstack((1./y_err).ravel()) * A
        y_0 = np.vstack( y.ravel() ) * np.vstack((1./y_err).ravel())

    covar=np.linalg.inv(A_0.T.dot(A_0))
    errs = (y_0 - A_0.dot(w_i))
    sigma2 = np.sum(errs**2.)/dof

    return np.sqrt(np.diag(sigma2 * covar))
        


def linear_leastsq(x, y, y_err=None):

    X = np.vstack(x.ravel()).T
    if y_err is None:
        return np.linalg.lstsq(X, y.ravel(), rcond=None)
    else:
        w = 1./y_err.ravel()**2.
        Xw = X * w[:,None]
        yw = y.ravel() * w
        return np.linalg.lstsq(Xw, yw, rcond=None)
        


def bin_image(img, binsize):
    imgbin_list = [img[i::binsize,j::binsize] for i in range(binsize) for j in range(binsize)]
    return np.sum(imgbin_list, axis=0)


        
