import numpy as np
from scipy.stats import ttest_ind, pearsonr
from sklearn.model_selection import StratifiedKFold
# General packages
import numpy as np
import seaborn as sns
import pandas as pd

# Bunch of scikit-learn stuff
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.feature_selection import f_classif
from sklearn.externals import joblib as jl
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder

# Specific statistics-functions
from scipy.stats import pearsonr

# Misc.
from tqdm import tqdm

from copy import deepcopy

# Plotting
import matplotlib.pyplot as plt
sns.set_style("ticks")

# Custom code! (install skbold by `pip install skbold`; counterbalance.py is in cwd)
from confounds import ConfoundRegressor
from counterbalance import CounterbalancedStratifiedSplit

from utils import get_r2
import time

class DataGenerator:

    def __init__(self, N, K, corr_cy, signal_r2, confound_r2=None,
                 c_type='continuous', y_type='binary', tolerance=0.01,
                 verbose=False):
        """
        Parameters
        ----------
        N : int
            Number of samples (N) in the data (X, y, and c)
        K : int
            Number of features (K) in the data (X)
        c_type : str
            Type of confound; either "continuous" or "binary". If binary,
            the data a balanced vector with ones and zeros
        y_type : str
            Type of target; either "continuous" or "binary".
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
        tolerance : float
            How much an observed statistic (corr_cy, signal_r2, confound_r2) may
            deviate from the desired value.
        verbose : bool
            Whether to print (extra) relevant information
        """
        self.N = N
        self.K = K
        self.corr_cy = corr_cy
        self.signal_r2 = signal_r2
        self.confound_r2 = confound_r2
        self.c_type = c_type
        self.y_type = y_type
        self.tolerance = tolerance
        self.verbose = verbose

    def generate(self):
        """ Generates X, y, and (optionally) c. """
        self._check_settings()
        self._init_y_and_c()

        # Define X as a matrix of N-samples by K-features
        X = np.zeros((self.N, self.K))

        # Pre-allocate arrays for average signal_r2 values and confound_r2 values
        signal_r2_values = np.zeros(self.K)
        confound_r2_values = np.zeros(self.K)

        icept = np.ones((self.N, 1))
        iterator = tqdm_notebook(np.arange(self.K)) if self.verbose else np.arange(self.K)

        for i in iterator:

            should_continue = False
            # Define generative parameters (gen_beta_y = beta-parameter for y in model of X)
            gen_beta_y, gen_beta_c = 1, 1
            noise_factor = 1
            this_c = 0 if self.confound_r2 is None else self.c
            tmp_confound_r2 = 0 if self.confound_r2 is None else self.confound_r2

            c_iter = 0
            start_time = time.time()
            while True:

                this_time = time.time()
                if c_iter > 100000:
                    gen_beta_y, gen_beta_c, noise_factor = 1, 1, 1
                    c_iter = 0

                if (start_time * 1000 - this_time * 1000) > 10:
                    print("Something's wrong")
                    print("C: %.3f, y: %.3f, noise: %.3f" % (gen_beta_y, gen_beta_c, noise_factor))
                # Generate X as a linear combination of y, c, and random noise
                this_X = (gen_beta_y * self.y + gen_beta_c * this_c + np.random.randn(self.N) * noise_factor)
                r2_X = pearsonr(this_X, self.y)[0] ** 2

                difference_obs_vs_desired = r2_X - (self.signal_r2 + tmp_confound_r2)
                if np.abs(difference_obs_vs_desired) > self.tolerance:  # should be even more strict
                    # If correlation too small/big, adjust noise factor and CONTINUE
                    if difference_obs_vs_desired < 0:
                        noise_factor -= 0.01
                    else:
                        noise_factor += 0.01
                    c_iter += 1
                    continue

                if self.confound_r2 is None and not should_continue:
                    signal_r2_values[i] = r2_X
                    X[:, i] = this_X
                    break

                c_tmp = np.hstack((icept, this_c[:, np.newaxis]))
                X_not_c = this_X - c_tmp.dot(np.linalg.lstsq(c_tmp, this_X, rcond=None)[0])
                this_signal_r2 = pearsonr(X_not_c, self.y)[0] ** 2
                this_confound_r2 = r2_X - this_signal_r2

                difference_obs_vs_desired = this_confound_r2 - self.confound_r2
                if np.abs(difference_obs_vs_desired) > self.tolerance:
                    if difference_obs_vs_desired < 0:
                        gen_beta_c += 0.01
                    else:
                        gen_beta_c -= 0.01
                    should_continue = True
                else:
                    should_continue = False

                difference_obs_vs_desired = this_signal_r2 - self.signal_r2
                if np.abs(difference_obs_vs_desired) > self.tolerance:
                    if difference_obs_vs_desired < 0:
                        gen_beta_y += 0.01
                    else:
                        gen_beta_y -= 0.01
                    should_continue = True
                else:
                    should_continue = False

                if should_continue:
                    c_iter += 1
                    continue
                else:  # We found it!
                    X[:, i] = this_X
                    signal_r2_values[i] = this_signal_r2
                    confound_r2_values[i] = this_confound_r2
                    break
        self.X = X
        self.signal_r2_values = signal_r2_values
        self.confound_r2_values = confound_r2_values
        if self.verbose:
            self._generate_report()

        return self

    def return_vals(self):
        """ Returns X, y, and (optionally) c. """
        if self.confound_r2 is not None:
            return self.X, self.y, self.c
        else:
            return self.X, self.y

    def _generate_report(self):
        """ If verbose, prints some stuff to check. """
        print("Signal r2: %.3f" % self.signal_r2_values.mean())

        if self.confound_r2 is not None:
            print("Confound r2: %.3f" % self.confound_r2_values.mean())

        if self.confound_r2 is not None:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(np.corrcoef(self.X.T), aspect='auto', cmap='RdBu')
            plt.title("Correlations between features")
            plt.colorbar()
            plt.grid('off')

            plt.subplot(1, 3, 2)
            plt.title("Signal R2 values")
            plt.hist(self.signal_r2_values, bins='auto')

            plt.subplot(1, 3, 3)
            plt.title("Confound R2 values")
            plt.hist(self.confound_r2_values, bins='auto')
            plt.tight_layout()
            plt.show()

    def _check_settings(self):
        """ Some checks of sensible parameters. """
        if self.N % 2 != 0:
            raise ValueError("Please select an even number of samples "
                             "(Makes things easier.)")

        if self.confound_r2 is not None:
            if np.abs(self.corr_cy) < np.sqrt(self.confound_r2):
                raise ValueError("The desired corr_cy value is less than the square "
                                 "root of the desired confound R-squared ... This is "
                                 "impossible to generate.")

        VAR_TYPES = ['binary', 'continuous']
        if self.y_type not in VAR_TYPES:
            raise ValueError("y_type must be one of %r" % VAR_TYPES)

        if self.c_type not in VAR_TYPES:
            raise ValueError("c_type must be one of %r" % VAR_TYPES)

    def _init_y_and_c(self):
        """ Initializes y and c. """
        if self.y_type == 'binary':
            y = np.repeat([0, 1], repeats=self.N / 2)
        else:  # assume continuous
            y = np.random.normal(0, 1, self.N)

        if self.c_type == 'binary':
            if self.y_type == 'binary':
                # Simply shift ("roll") y to create correlation using the "formula":
                # to-shift = N / 4 * (1 - corr_cy)
                to_roll = int((self.N / 4) * (1 - self.corr_cy))
                c = np.roll(y, to_roll)
            else:  # y is continuous
                c = y.copy()
                this_corr_cy = pearsonr(c, y)[0]
                i = 0
                while np.abs(this_corr_cy - self.corr_cy) > self.tolerance:
                    np.shuffle(c)
                    this_corr_cy = pearsonr(c, y)
                    i += 1

                    if i > 10000:
                        raise ValueError("Probably unable to find good corr_cy value")
        else:
            # If c is continuous, just sample y + random noise
            noise_factor = 10
            c = y + np.random.randn(self.N) * noise_factor
            this_corr_cy = pearsonr(c, y)[0]

            i = 0
            while np.abs(this_corr_cy - self.corr_cy) > self.tolerance:
                # Decrease noise if the difference is too big
                noise_factor -= 0.01
                c = y + np.random.randn(self.N) * noise_factor
                this_corr_cy = pearsonr(c, y)[0]
                i += 1

                if i > 10000:
                    # Reset noise factor
                    noise_factor = 10
                    i = 0

        self.y = y
        self.c = c

from utils import vectorized_semipartial_corr, vectorized_corr

def run_without_confound_control(X, y, c, pipeline, cv, arg_dict, sim_nr=None):
    """ Run a classification analysis using without controlling for confounds.
    
    Parameters
    ----------
    X : numpy array
        Array of shape N (samples) x K (features) with floating point numbers
    y : numpy array
        Array of shape N (samples) x 1 with binary numbers {0, 1}
    c : numpy array
        Array of shape N (samples) x 1 with either binary {0, 1}
        or continuous (from normal dist, 0 mean, 1 variance) values
    pipeline : Pipeline-object
        A scikit-learn Pipeline-object
    n_splits : int
        Number of splits to generate in the K-fold routine
    arg_dict : dict
        Dictionary with arguments used in data generation
        (i.e. args fed to generate_data function)
        
    Returns
    -------
    results : pandas DataFrame
        DataFrame with data parameters (from arg-dict) and fold-wise scores.
    """
    
    results = pd.concat([pd.DataFrame(arg_dict, index=[i]) for i in range(n_splits)])
    results['method'] = ['None'] * n_splits
    results['sim_nr'] = [sim_nr] * n_splits
    results['score'] = cross_val_score(estimator=pipeline, X=X, y=y, cv=cv, scoring=scoring)
    
    return results


def run_with_ipw(X, y, c, pipeline, cv, arg_dict, sim_nr=None):
    """ Run a classification analysis using without controlling for confounds.
    
    Parameters
    ----------
    X : numpy array
        Array of shape N (samples) x K (features) with floating point numbers
    y : numpy array
        Array of shape N (samples) x 1 with binary numbers {0, 1}
    c : numpy array
        Array of shape N (samples) x 1 with either binary {0, 1}
        or continuous (from normal dist, 0 mean, 1 variance) values
    pipeline : Pipeline-object
        A scikit-learn Pipeline-object
    n_splits : int
        Number of splits to generate in the K-fold routine
    arg_dict : dict
        Dictionary with arguments used in data generation
        (i.e. args fed to generate_data function)
        
    Returns
    -------
    results : pandas DataFrame
        DataFrame with data parameters (from arg-dict) and fold-wise scores.
    """
    
    results = pd.concat([pd.DataFrame(arg_dict, index=[i]) for i in range(n_splits)])
    results['method'] = ['IPW'] * n_splits
    results['sim_nr'] = [sim_nr] * n_splits

    y_ohe = OneHotEncoder(sparse=False).fit_transform(y[:, np.newaxis])
    skf = StratifiedKFold(n_splits=n_splits)
    lr = LogisticRegression(class_weight='balanced')
    
    if c.ndim == 1:
        c = c[:, np.newaxis]
    
    tmp_scores = np.zeros(n_splits)
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        lr.fit(c[train_idx], y[train_idx])
        probas = lr.predict_proba(c[train_idx])
        weights = 1 / (probas * y_ohe[train_idx]).sum(axis=1)
        pipeline.fit(X[train_idx], y[train_idx], clf__sample_weight=weights)
        preds = pipeline.predict(X[test_idx])
        tmp_scores[i] = f1_score(y[test_idx], preds, average='macro')
    
    results['score'] = tmp_scores
    
    return results


def run_with_counterbalancing_random(X, y, c, pipeline, cv, arg_dict, verbose=False,
                              c_type='categorical', metric='corr', threshold=0.05,
                              use_pval=True, sim_nr=None):
    """ Run a classification analysis using without controlling for confounds.
    Parameters
    ----------
    X : numpy array
        Array of shape N (samples) x K (features) with floating point numbers
    y : numpy array
        Array of shape N (samples) x 1 with binary numbers {0, 1}
    c : numpy array
        Array of shape N (samples) x 1 with either binary {0, 1}
        or continuous (from normal dist, 0 mean, 1 variance) values
    pipeline : Pipeline-object
        A scikit-learn Pipeline-object
    n_splits : int
        Number of splits to generate in the K-fold routine
    arg_dict : dict
        Dictionary with arguments used in data generation
        (i.e. args fed to generate_data function)
    
    Returns
    -------
    results : pandas DataFrame
        DataFrame with data parameters (from arg-dict) and fold-wise scores.
    """

    results = pd.concat([pd.DataFrame(arg_dict, index=[i]) for i in range(n_splits)])
    results['method'] = ['CB'] * n_splits
    results['sim_nr'] = [sim_nr] * n_splits

    results_corr = pd.concat([pd.DataFrame(arg_dict, index=[i]) for i in range(arg_dict['K']*2)])
    results_corr['ki'] = np.tile(np.arange(arg_dict['K']), reps=2)
    results_corr['before_after'] = ['before'] * arg_dict['K'] + ['after'] * arg_dict['K']
    results_corr['sim_nr'] = [sim_nr] * arg_dict['K'] * 2
    
    corrs_xy_before = vectorized_corr(y, X)
    scorrs_xy_before = vectorized_semipartial_corr(y, c, X, which='2D')    
    scorrs_xc_before = vectorized_semipartial_corr(c, y, X, which='2D')
    
    skf = CounterbalancedStratifiedSplitRandom(X, y, c, n_splits=cv, c_type=c_type, verbose=verbose,
                                         metric=metric, use_pval=use_pval, threshold=threshold)
    try:
        skf.check_counterbalance_and_subsample()
    except:
        results['score'] = np.zeros(n_splits)
        corrs_xy_after = np.zeros_like(corrs_xy_before)
        scorrs_xy_after = np.zeros_like(scorrs_xy_before)
        scorrs_xc_after = np.zeros_like(scorrs_xy_before)
        
        results_corr['corr_xy'] = np.concatenate((corrs_xy_before, corrs_xy_after))
        results_corr['scorr_xy'] = np.concatenate((scorrs_xy_before, scorrs_xy_after))
        results_corr['scorr_xc'] = np.concatenate((scorrs_xc_before, scorrs_xc_after))
        
        results_corr['subsample_perc'] = 100
        return results, results_corr

    Xn, yn, cn = X[skf.subsample_idx], y[skf.subsample_idx], c[skf.subsample_idx]    
    corrs_xy_after = vectorized_corr(yn, Xn)
    scorrs_xy_after = vectorized_semipartial_corr(yn, cn, Xn, which='2D')
    scorrs_xc_after = vectorized_semipartial_corr(cn, yn, Xn, which='2D')
    
    results_corr['corr_xy'] = np.concatenate((corrs_xy_before, corrs_xy_after))
    results_corr['scorr_xy'] = np.concatenate((scorrs_xy_before, scorrs_xy_after))
    results_corr['scorr_xc'] = np.concatenate((scorrs_xc_before, scorrs_xc_after))
    
    results_corr['subsample_perc'] = [(Xn.shape[0] - X.shape[0]) / X.shape[0]] * X.shape[1] * 2
    results['score'] = cross_val_score(estimator=pipeline, X=Xn, y=yn, cv=skf, scoring=scoring)
    return results, results_corr


def run_with_wholedataset_confound_regression(X, y, c, pipeline, cv, arg_dict, sim_nr=None):
    """ Run a classification analysis using without controlling for confounds.
    
    Parameters
    ----------
    X : numpy array
        Array of shape N (samples) x K (features) with floating point numbers
    y : numpy array
        Array of shape N (samples) x 1 with binary numbers {0, 1}
    c : numpy array
        Array of shape N (samples) x 1 with either binary {0, 1}
        or continuous (from normal dist, 0 mean, 1 variance) values
    pipeline : Pipeline-object
        A scikit-learn Pipeline-object
    n_splits : int
        Number of splits to generate in the K-fold routine
    arg_dict : dict
        Dictionary with arguments used in data generation
        (i.e. args fed to generate_data function)
        
    Returns
    -------
    results : pandas DataFrame
        DataFrame with data parameters (from arg-dict) and fold-wise scores.
    """
    
    results = pd.concat([pd.DataFrame(arg_dict, index=[i]) for i in range(n_splits)])
    results['method'] = ['WDCR'] * n_splits
    results['sim_nr'] = [sim_nr] * n_splits
    
    # Regress out c from X
    cr = ConfoundRegressor(X=X, confound=c, cross_validate=True)
    X_corr = cr.fit_transform(X)
    results['score'] = cross_val_score(estimator=pipeline, X=X_corr, y=y, cv=cv, scoring=scoring)

    return results


def run_with_foldwise_confound_regression(X, y, c, pipeline, cv, arg_dict, sim_nr=None):
    """ Run a classification analysis using without controlling for confounds.
    
    Parameters
    ----------
    X : numpy array
        Array of shape N (samples) x K (features) with floating point numbers
    y : numpy array
        Array of shape N (samples) x 1 with binary numbers {0, 1}
    c : nu1mpn_sampy array
        Array of shape N (samples) x 1 with either binary {0, 1}
        or continuous (from normal dist, 0 mean, 1 variance) values
    pipeline : Pipeline-object
        A scikit-learn Pipeline-object
    n_splits : intn_sampn_samp
        Number of splits to generate in the K-fold routine
    arg_dict : dict
        Dictionary with arguments used in data generation
        (i.e. args fed to generate_data function)

    Returns
    -------
    results : pandas DataFrame
        DataFrame with data parameters (from arg-dict) and fold-wise scores.
    """
 
    results = pd.concat([pd.DataFrame(arg_dict, index=[i]) for i in range(n_splits)])
    results['method'] = ['CVCR'] * n_splits
    results['sim_nr'] = [sim_nr] * n_splits

    skf = StratifiedKFold(n_splits=n_splits)
    scores = np.zeros(n_splits)
    
    cfr = ConfoundRegressor(X=X, confound=c, cross_validate=True)
    this_pipe = deepcopy(pipeline).steps
    this_pipe.insert(0, ('regress', cfr))
    this_pipe = Pipeline(this_pipe)
    results['score'] = cross_val_score(estimator=this_pipe, X=X, y=y, cv=cv, scoring=scoring)
    
    return results

class CounterbalancedStratifiedSplitRandom(object):

    def __init__(self, X, y, c, n_splits=5, c_type='categorical',
                 metric='corr', use_pval=False, threshold=0.05, verbose=False):

        self.X = X
        self.y = y
        self.c = c
        self.z = None
        self.n_splits = n_splits
        self.c_type = c_type
        self.metric = metric
        self.use_pval = use_pval
        self.threshold = threshold
        self.seed = None
        self.verbose = verbose

    def _validate_fold(self, y, c):

        if self.c_type == 'continuous':

            if self.metric == 'corr':
                stat, pval = pearsonr(c, y)
            elif self.metric == 'tstat':
                stat, pval = ttest_ind(c[y == 0], c[y == 1])
            else:
                raise ValueError("Please choose either 'corr' or 'tstat'!")

            if self.use_pval:
                #print(pearsonr(self.y[self.subsample_idx], self.c[self.subsample_idx]))
                return pval > self.threshold
            else:
                return np.abs(stat) < self.threshold

        elif self.c_type == 'categorical':
            bincounts = np.zeros((np.unique(y).size, np.unique(c).size))
            for i, y_class in enumerate(np.unique(y)):
                bincounts[i, :] = np.bincount(c[y == y_class])

            counterbalanced = np.all(bincounts[0, :] == bincounts[1, :])
            return counterbalanced

    def _subsample_continuous(self, iteration=0):

        # First, let's do a t-test to check for differences between
        # c | y=0 and c | y=1; thus, only binary c for now
        self.subsample_idx = np.arange(self.y.size)
        amount = int(1 + np.floor(iteration / 10000))
        this_c = self.c[self.subsample_idx]
        this_y = self.y[self.subsample_idx]

        c_y0 = this_c[this_y == 0]
        c_y1 = this_c[this_y == 1]
        idx_c_y0 = self.subsample_idx[this_y == 0]
        idx_c_y1 = self.subsample_idx[this_y == 1]
        
        idx_c_y0 = np.random.choice(idx_c_y0, size=idx_c_y0.size - amount, replace=False)    
        idx_c_y1 = np.random.choice(idx_c_y1, size=idx_c_y1.size - amount, replace=False)
        self.subsample_idx = np.sort(np.concatenate((idx_c_y0, idx_c_y1)))
        
        #if iteration % 100 == 0:
        #    print(pearsonr(self.y[self.subsample_idx], self.c[self.subsample_idx]))
        
    def _subsample_categorical(self):

        c_unique = np.unique(self.c)
        y_unique = np.unique(self.y)
        counts = np.zeros((y_unique.size, c_unique.size))

        # Count how many times a c appears for a given y
        # so, count(c | y_i)
        for i, y_class in enumerate(y_unique):
            this_c = self.c[self.y == y_class]
            counts[i, :] = np.array([(c == this_c).sum() for c in c_unique])

        # ... yielding a len(y_unique) x len(c_unique) matrix
        # Now, take the minimum across rows
        min_counts = counts.min(axis=0)

        if np.all(min_counts == 0):
            msg = ("Wow, your data is really messed up ... There is no way to "
                   "subsample it, because the minimum proportion of all values"
                   "of c across all values of y is 0 ...")
            raise ValueError(msg)

        # Which are exactly the number of trials (per c) which you need to
        # subsample, which is done below:
        final_idx = []
        for i, y_class in enumerate(y_unique):

            this_idx = self.subsample_idx[self.y == y_class]
            this_c = self.c[self.y == y_class]

            for ii, c in enumerate(c_unique):
                final_idx.append(np.random.choice(this_idx[this_c == c],
                                                  int(min_counts[ii]),
                                                  replace=False))

        # The concatenated indices now represent the indices needed to
        # properly subsample the data to make it counterbalanced
        self.subsample_idx = np.sort(np.concatenate(final_idx))

    def _subsample(self, iteration):

        if self.c_type == 'continuous':
            self._subsample_continuous(iteration=iteration)
        elif self.c_type == 'categorical':
            self._subsample_categorical()
        else:
            raise ValueError("Please pick c_type='categorical' or "
                             "c_type='continuous'")

        if len(self.subsample_idx) < (2 * len(np.unique(self.y))):
            msg = ("Probably subsampled too much (only have %i samples now); "
                   "this dataset can't be meaningfully "
                   "counterbalanced" % len(self.subsample_idx))
            raise ValueError(msg)

    def _find_counterbalanced_seed(self, max_attempts=10):
        """ Find a seed of Stratified K-Fold that gives counterbalanced
        classes """

        y_tmp = self.y[self.subsample_idx]
        c_tmp = self.c[self.subsample_idx]
        X_tmp = self.X[self.subsample_idx]

        to_stratify = y_tmp if self.z is None else self.z
        if self.c_type == 'categorical':
            lowest_strat_count = np.min(np.bincount(to_stratify))

            if lowest_strat_count < self.n_splits:
                raise ValueError("You have too few samples of each c-y "
                                 "combination to completely counterbalance all "
                                 "your folds with n_splits=%i; highest number of "
                                 "splits you can use is %i" % (self.n_splits,
                                                               lowest_strat_count))

        seeds = np.random.randint(0, high=1e7, size=max_attempts, dtype=int)

        for i, seed in enumerate(seeds):

            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                  random_state=seed)

            for (train_idx, test_idx) in skf.split(X_tmp, y=to_stratify):

                this_y, this_c = y_tmp[train_idx], c_tmp[train_idx]
                good_split = self._validate_fold(this_y, this_c)
                if not good_split:
                    break

            if good_split:

                if self.verbose:
                    print("Picking seed %i" % seed)

                self.seed = seed
                return True

        return False

    def check_counterbalance_and_subsample(self):

        self.subsample_idx = np.arange(self.y.size)

        if self.c_type == 'continuous':
            found_split = self._find_counterbalanced_seed()
            i = 0
            while not found_split:
                self._subsample(iteration=i)
                found_split = self._find_counterbalanced_seed()
                i += 1
        elif self.c_type == 'categorical':
            self._subsample()
            recode_dict = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
            this_c = self.c[self.subsample_idx]
            this_y = self.y[self.subsample_idx]
            self.z = [recode_dict[(yi, ci)] for yi, ci in zip(this_c, this_y)]
            found_split = self._find_counterbalanced_seed()

        if self.verbose:
            new_N, old_N = len(self.subsample_idx), self.y.size
            print("Size of y after subsampling: %i (%.1f percent reduction in "
                  "samples)" % (new_N, (old_N - new_N) / old_N * 100))

    def split(self, X, y, groups=None):
        """ The final idx to output are subsamples of the subsample_idx... """

        if self.seed is None:
            raise ValueError("Run '.check_counterbalance_and_subsample' "
                             "before you run '.split'!")

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.seed)

        to_stratify = y if self.z is None else self.z
        for (train_idx, test_idx) in skf.split(X=X, y=to_stratify):
            yield ((train_idx, test_idx))

def run_with_counterbalancing_random(X, y, c, pipeline, cv, arg_dict, verbose=False,
                              c_type='categorical', metric='corr', threshold=0.05,
                              use_pval=True, sim_nr=None):
    """ Run a classification analysis using without controlling for confounds.
    Parameters
    ----------
    X : numpy array
        Array of shape N (samples) x K (features) with floating point numbers
    y : numpy array
        Array of shape N (samples) x 1 with binary numbers {0, 1}
    c : numpy array
        Array of shape N (samples) x 1 with either binary {0, 1}
        or continuous (from normal dist, 0 mean, 1 variance) values
    pipeline : Pipeline-object
        A scikit-learn Pipeline-object
    n_splits : int
        Number of splits to generate in the K-fold routine
    arg_dict : dict
        Dictionary with arguments used in data generation
        (i.e. args fed to generate_data function)
    
    Returns
    -------
    results : pandas DataFrame
        DataFrame with data parameters (from arg-dict) and fold-wise scores.
    """

    results = pd.concat([pd.DataFrame(arg_dict, index=[i]) for i in range(n_splits)])
    results['method'] = ['CB'] * n_splits
    results['sim_nr'] = [sim_nr] * n_splits

    results_corr = pd.concat([pd.DataFrame(arg_dict, index=[i]) for i in range(arg_dict['K']*2)])
    results_corr['ki'] = np.tile(np.arange(arg_dict['K']), reps=2)
    results_corr['before_after'] = ['before'] * arg_dict['K'] + ['after'] * arg_dict['K']
    results_corr['sim_nr'] = [sim_nr] * arg_dict['K'] * 2
    
    corrs_xy_before = vectorized_corr(y, X)
    scorrs_xy_before = vectorized_semipartial_corr(y, c, X, which='2D')    
    scorrs_xc_before = vectorized_semipartial_corr(c, y, X, which='2D')
    
    skf = CounterbalancedStratifiedSplitRandom(X, y, c, n_splits=cv, c_type=c_type, verbose=verbose,
                                               use_pval=use_pval, metric=metric, threshold=threshold)
    try:
        skf.check_counterbalance_and_subsample()
    except:
        print("WRONG")
        results['score'] = np.zeros(n_splits)
        corrs_xy_after = np.zeros_like(corrs_xy_before)
        scorrs_xy_after = np.zeros_like(scorrs_xy_before)
        scorrs_xc_after = np.zeros_like(scorrs_xy_before)
        
        results_corr['corr_xy'] = np.concatenate((corrs_xy_before, corrs_xy_after))
        results_corr['scorr_xy'] = np.concatenate((scorrs_xy_before, scorrs_xy_after))
        results_corr['scorr_xc'] = np.concatenate((scorrs_xc_before, scorrs_xc_after))
        
        results_corr['subsample_perc'] = 100
        return results, results_corr

    Xn, yn, cn = X[skf.subsample_idx], y[skf.subsample_idx], c[skf.subsample_idx]    
    corrs_xy_after = vectorized_corr(yn, Xn)
    scorrs_xy_after = vectorized_semipartial_corr(yn, cn, Xn, which='2D')
    scorrs_xc_after = vectorized_semipartial_corr(cn, yn, Xn, which='2D')
    
    results_corr['corr_xy'] = np.concatenate((corrs_xy_before, corrs_xy_after))
    results_corr['scorr_xy'] = np.concatenate((scorrs_xy_before, scorrs_xy_after))
    results_corr['scorr_xc'] = np.concatenate((scorrs_xc_before, scorrs_xc_after))
    
    results_corr['subsample_perc'] = [(Xn.shape[0] - X.shape[0]) / X.shape[0]] * X.shape[1] * 2
    results['score'] = cross_val_score(estimator=pipeline, X=Xn, y=yn, cv=skf, scoring=scoring)
    return results, results_corr

if __name__ == '__main__':

    # We do it three times for robustness
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    scoring = 'f1_macro'

    simulations = 1
    n_splits = 10
    pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='linear'))])

    # Specify arguments for data generations (we'll set corr_cy and confound_r2 later)
    data_args = dict(N=200, K=5, c_type='continuous', y_type='binary',
                     corr_cy=None, signal_r2=0.004, confound_r2=None,
                     verbose=False, tolerance=0.001)

    # The values for corr_cy to loop over
    corr_cy_vec = [0.65]

    # The confound_r2 values to loop over
    confound_r2_vecs = [np.arange(0, corr_cy**2 - 0.05, 0.05)
                        for corr_cy in corr_cy_vec]

    signal_r2_vec = [0.004, 0.1]

    # The 'reference performance' to keep track of (scores on data generated without
    # any influence of the confound)
    reference_performance = np.zeros((simulations, len(signal_r2_vec)))

    results_gen_sim = []
    results_corr_gen_sim = []

    # Loop over simulations
    for sim in np.arange(simulations):

        print("Simulation: %i" % (sim + 1))
        cv = StratifiedKFold(n_splits=n_splits)
        # Loop over values for corr_cy
        for i, signal_r2 in enumerate(signal_r2_vec):
            data_args.update(signal_r2=signal_r2)
            data_args.update(confound_r2=None)
            data_args.update(corr_cy=0)

            dgen = DataGenerator(**data_args)
            Xref, yref = dgen.generate().return_vals()
            reference_performance[sim, i] = cross_val_score(pipeline, Xref, yref, cv=cv).mean()

            print("Signal r2: %.3f" % signal_r2)
            for ii, corr_cy in enumerate(corr_cy_vec):
                data_args.update(corr_cy=corr_cy)
                data_args.update(confound_r2=None)

                confound_r2_vec = confound_r2_vecs[ii]

                # Loop over values for confound_r2
                for iii, confound_r2 in tqdm(enumerate(confound_r2_vec)):
                    data_args.update(confound_r2=confound_r2)
                    dgen = DataGenerator(**data_args)
                    X, y, c = dgen.generate().return_vals()
                    results_gen_sim.append(run_without_confound_control(X, y, c, pipeline, cv, data_args, sim_nr=sim))
                    results_gen_sim.append(run_with_wholedataset_confound_regression(X, y, c, pipeline, cv, data_args, sim_nr=sim))
                    results_gen_sim.append(run_with_foldwise_confound_regression(X, y, c, pipeline, cv, data_args, sim_nr=sim))
                    res, corrs = run_with_counterbalancing_random(X, y, c, pipeline, n_splits, c_type='continuous',
                                                           arg_dict=data_args, sim_nr=sim)
                    results_gen_sim.append(res)
                    results_corr_gen_sim.append(corrs)

    results_gen_sim_df = pd.concat(results_gen_sim)
    results_corrs_gen_sim_df = pd.concat(results_corr_gen_sim)
    results_gen_sim_df.to_csv('results_gen_sim_random.tsv', sep='\t', index=False)
    results_corrs_gen_sim_df.to_csv('results_corrs_random.tsv', sep='\t', index=False)

