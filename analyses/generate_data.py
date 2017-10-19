import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt


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
    r2 = pearsonr(dv_hat, dv)[0] ** 2
    if np.isnan(r2):
        r2 = 0
    
    return r2



def generate_data(n_samp, k_feat, c_type, corr_cy, signal_r2, confound_r2=None, verbose=False):
    """ Generate data with known (partial) R2 "structure".
    
    Parameters
    ----------
    n_samp : int
        Number of samples (N) in the data (X, y, and c)
    k_feat : int
        Number of features (K) in the data (X)
    c_type : str
        Either "continuous" or "categorical". If categorical,
        the data a balanced vector with ones and zeros
    corr_cy : float
        Number between -1 and 1, specifying the correlation
        between the confound (c) and the target (y)
    signal_r2 : float
        Number between 0 and 1, specifying the explained variance
        of y using X, independent of the confound contained in X;
        (technically, the semipartial correlation rho(xy.c)
    confound_r2 : float or None
        Number between 0 and 1 (or None), specifying the shared variance 
        explained of y of x and c (i.e. the explained variance 
        of the confound-related information in x). If None,
        no confound R2 will be left unspecified (which can be used
        to specify a baseline).
    verbose : bool
        Whether to print (extra) relevant information
    
    Returns
    -------
    X : numpy array
        Array of shape N (samples) x K (features) with floating point numbers
    y : numpy array
        Array of shape N (samples) x 1 with binary numbers {0, 1}
    c : numpy array
        Array of shape N (samples) x 1 with either binary {0, 1}
        or continuous (from normal dist, 0 mean, 1 variance) values,
        depending on what you set for the `c_type` argument.
    """
    
    if n_samp % 2 != 0:
        raise ValueError("Please select an even number of samples "
                         "(Makes things easier.)")

    if confound_r2 is not None:
        if np.abs(corr_cy) < np.sqrt(confound_r2):
            raise ValueError("The desired corr_cy value is less than the square "
                             "root of the desired confound R-squared ... This is "
                             "impossible to generate.")
        
    # Generate y (balanced, 50% class 0, 50% class 1)
    y = np.repeat([0, 1], repeats=n_samp / 2)
    
    # Generate c (confound), with given correlation corr_cy
    if c_type == 'categorical':
        # Simply shift ("roll") y to create correlation using the "formula":
        # to-shift = N / 4 * (1 - corr_cy)
        to_roll = int((n_samp / 4) * (1 - corr_cy))
        c = np.roll(y, to_roll)
    elif c_type == 'continuous':
        # If c is continuous, just sample y + random noise
        noise_factor = 100
        c = y + np.random.randn(n_samp) * noise_factor
        corr = pearsonr(c, y)[0]
        
        while np.abs(corr - corr_cy) > 0.01:
            # Decrease noise if the difference is too big
            noise_factor -= 0.01
            c = y + np.random.randn(n_samp) * noise_factor
            corr = pearsonr(c, y)[0]        
    else:
        raise ValueError("For c_type, please select from {'continuous', "
                         "'categorical'}")
    
    # Define X as a matrix of N-samples by K-features
    X = np.zeros((n_samp, k_feat))
    
    # Pre-allocate arrays for average signal_r2 values and confound_r2 values
    signal_r2_values = np.zeros(k_feat)
    confound_r2_values = np.zeros(k_feat)
    
    icept = np.ones((n_samp, 1))
    
    iterator = tqdm_notebook(np.arange(k_feat)) if verbose else np.arange(k_feat)
    for i in iterator:
        
        # Define generative parameters (gen_beta_y = beta-parameter for y in model of X)
        # Upon advice of Steven S., 'reset' generative parameters after each generation
        gen_beta_y = 1
        gen_beta_c = 1
        noise_factor = 1
        
        while True:
            
            should_continue = False
            
            # Generate X as a linear combination of y, c, and random noise
            this_c = 0 if confound_r2 is None else c
            this_X = (gen_beta_y * y + gen_beta_c * this_c + np.random.randn(n_samp) * noise_factor)
            this_X = this_X[:, np.newaxis]
            
            # Fit y = b1X
            y_x_r2 = get_r2(iv=this_X, dv=y, stack_intercept=True)  # B + C
            
            # Increase/decrease noise if difference observed r(yx)**2 is too big/small,
            # because if y_x_r2 < (signal_r2 + confound_r2), you won't find proper data anyway
            tmp_confound_r2 = 0 if confound_r2 is None else confound_r2
            difference_obs_vs_desired = y_x_r2 - (signal_r2 + tmp_confound_r2)
            if np.abs(difference_obs_vs_desired) > 0.01:
                # If correlation too small/big, adjust noise factor and CONTINUE
                if difference_obs_vs_desired < 0:
                    noise_factor -= 0.01
                else:
                    noise_factor += 0.01
                continue
            
            if confound_r2 is None:
                # We don't care about confound_r2
                unique_var_x = y_x_r2
            else:
                # Fit y = b1X + b2C
                y_xc_r2 = get_r2(iv=np.hstack((this_X, c[:, np.newaxis])), dv=y,
                             stack_intercept=True)  # B + C + D
                resid_y = 1 - y_xc_r2  # A = 1 - (B + C + D)

                # Fit y = b1C
                y_c_r2 = get_r2(iv=c, dv=y, stack_intercept=True)  # C + D
                unique_var_x = y_xc_r2 - y_c_r2  # B
            
            # Increase/decrease generative param for y if difference 
            # r(yx.c) is too small/big ...
            difference_obs_vs_desired = unique_var_x - signal_r2
            if np.abs(difference_obs_vs_desired) > 0.01:
                if difference_obs_vs_desired < 0:
                    gen_beta_y += 0.01
                else:
                    gen_beta_y -= 0.01
                
                if confound_r2 is None:
                    continue
                else:
                    should_continue = True
            else:
                if confound_r2 is None:
                    break
                    
            unique_var_c = y_xc_r2 - y_x_r2  # D
            shared_var_xc = 1 - resid_y - unique_var_x - unique_var_c  # C

            # Also check if shared variance of c and x (component C) is appropriate;
            # if not, adjust generative parameter and CONTINUE
            difference_obs_vs_desired = shared_var_xc - confound_r2
            if np.abs(difference_obs_vs_desired) > 0.01:
                if difference_obs_vs_desired < 0:
                    gen_beta_c += 0.01
                else:
                    gen_beta_c -= 0.01
                should_continue = True
            
            if should_continue:
                continue
            else:
                break

        # If we didn't encounter a "break" statement, we must have found
        # data with the correct specifications ...
        X[:, i] = this_X.squeeze()
        signal_r2_values[i] = unique_var_x
        if confound_r2 is not None:
            confound_r2_values[i] = shared_var_xc
        
    if verbose:
        print("Signal r2: %.3f" % signal_r2_values.mean())
        
        if confound_r2 is not None:
            print("Confound r2: %.3f" % confound_r2_values.mean())
        
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(np.corrcoef(X.T), aspect='auto', cmap='RdBu')
        plt.title("Correlations between features")
        plt.colorbar()
        plt.grid('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Signal R2 values")
        plt.hist(signal_r2_values, bins='auto')
        
        plt.subplot(1, 3, 3)
        plt.title("Confound R2 values")
        plt.hist(confound_r2_values, bins='auto')
        plt.tight_layout()
        plt.show()
    
    if confound_r2 is None:
        return X, y
    else:
        return X, y, c
