import numpy as np
from scipy.special import hyp2f1, gammaln


def get_r2(iv, dv, stack_intercept=True):
    """ Regress dv onto iv and return r-squared.
    
    Parameters
    ----------
    iv : numpy array
        Array of shape N (samples) x K (features)
    dv : numpy array
        Array of shape N (samples) x 1
    stack_intercept : bool
        Whether to stack an intercept (vector with ones of length N).
    
    Returns
    -------
    r2 : float
        R-squared model fit.
    """
    
    if iv.ndim == 1:
        # Add axis if shape is (N,)
        iv = iv[:, np.newaxis]
    
    if stack_intercept:
        iv = np.hstack((np.ones((iv.shape[0], 1)), iv))
    
    beta = np.linalg.lstsq(iv, dv)[0]
    dv_hat = iv.dot(beta).squeeze()
    r2 = 1 - (((dv - dv_hat) ** 2).sum() / ((dv - dv.mean()) ** 2).sum())
    
    return r2


def vectorized_corr(arr, arr_2D):
    """ Computes the correlation between an array and each column
    in a 2D array (each column represents a variable) in a vectorized
    way. 
    
    Parameters
    ----------
    arr : numpy array
        Array of shape (N,)
    arr_2D : numpy array
        Array of shape (N, P), with P indicating different variables that
        will be correlated with arr
        
    Returns
    -------
    corrs : numpy array
        Array of shape (P,) with all correlations between arr and columns in arr_2D
    """
    
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    
    arr_c, arr_2D_c = arr - arr.mean(), arr_2D - arr_2D.mean(axis=0)
    r_num = np.sum(arr_c * arr_2D_c, axis=0)    
    r_den = np.sqrt(np.sum(arr_c ** 2, axis=0) * np.sum(arr_2D_c ** 2, axis=0))
    corrs = r_num / r_den
    return corrs


def vectorized_partial_corr(arr, c, arr_2D, stack_intercept=True):
    """ Computes the correlation between an array and each column
    in a 2D array (each column represents a variable) in a vectorized
    way. 
    
    Parameters
    ----------
    arr : numpy array
        Array of shape (N,)
    c : numpy array
        Array of shape (N,) that should be partialled out of arr_2D and arr
    arr_2D : numpy array
        Array of shape (N, P), with P indicating different variables that
        will be correlated with arr
        
    Returns
    -------
    corrs : numpy array
        Array of shape (P,) with all correlations between arr and columns in arr_2D
    """

    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    
    if c.ndim == 1:
        # Add axis if shape is (N,)
        c = c[:, np.newaxis]
    
    if stack_intercept:
        c = np.hstack((np.ones((c.shape[0], 1)), c))

    arr_resid = arr - c.dot(np.linalg.lstsq(c, arr, rcond=None)[0])
    arr_2d_resid = arr_2D - c.dot(np.linalg.lstsq(c, arr_2D, rcond=None)[0])
    
    return vectorized_corr(arr_resid, arr_2d_resid)


def vectorized_semipartial_corr(arr, c, arr_2D, which='2D', stack_intercept=True):
    """ Computes the semipartial correlation between an array and each column
    in a 2D array (each column represents a variable) in a vectorized
    way. 
    
    Parameters
    ----------
    arr : numpy array
        Array of shape (N,)
    c : numpy array
        Array of shape (N,) that should be partialled out of arr_2D and arr
    arr_2D : numpy array
        Array of shape (N, P), with P indicating different variables that
        will be correlated with arr
        
    Returns
    -------
    corrs : numpy array
        Array of shape (P,) with all correlations between arr and columns in arr_2D
    """

    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    
    if c.ndim == 1:
        # Add axis if shape is (N,)
        c = c[:, np.newaxis]
    
    if stack_intercept:
        c = np.hstack((np.ones((c.shape[0], 1)), c))

    if which == '2D':
        arr_2D_resid = arr_2D - c.dot(np.linalg.lstsq(c, arr_2D, rcond=None)[0])
        return vectorized_corr(arr, arr_2D_resid)
    else:
        arr_resid = arr - c.dot(np.linalg.lstsq(c, arr)[0])
        return vectorized_corr(arr_resid, arr_2D)


def rpdf(rho, n, rs):
    """ rho = population correlation coefficient. """
    lnum = np.log(n-2) + gammaln(n-1) + np.log((1-rho**2)**(.5*(n-1))) + np.log((1-rs**2)**(.5*(n-4)))
    lden = np.log(np.sqrt(2*np.pi)) + gammaln(n-.5) + np.log((1-rho*rs)**(n-3/2))
    fac = lnum - lden
    hyp = hyp2f1(.5, .5, (2*n-1)/2, (rho*rs+1)/2)
    return np.exp(fac) * hyp
