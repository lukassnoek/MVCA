import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pandas as pd

from tqdm import tqdm
from nistats.hemodynamic_models import spm_hrf
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr
from tqdm import tqdm_notebook
from confounds import ConfoundRegressor


class FmriData:
    """ Generates fMRI data.

    Parameters
    ----------
    P : int
        Number of conditions
    I : int
        Number of instances per condition
    I_dur : int
        Duration of trial in seconds (fixed across conditions, P)
    ISIs = list
        List of possible ISIs
    TR : int
        Time to repetition (must be int for simplicity)
    K : tuple
        Voxel dimensions
    ar_rho1 : float
        AR(p) autocorrelation of noise
    smoothness : int
        Spatial smoothing factor (in sigma of voxel dimensions)
    noise_factor : int/float
        How "strong" the noise is (scaling factor for noise covariance matrix V)
    cond_means : list
        List of length P with condition means of activation
    cond_stds : list
        List of length P with condition standard deviation of activation
    confound_corr : float
        Correlation of confound with conditions (assuming 2-class condition for simplicity). If None,
        no confound is simulated.
    """

    def __init__(self, P=2, I=20, I_dur=2, ISIs=(4, 5, 6, 7), TR=2, K=(3, 3),
                 ar1_rho=0.5, smoothness=2, noise_factor=1, cond_means=None, cond_stds=None,
                 conf_params=None, single_trial=True):
        """ Initializes FmriData object. """

        self.P = P
        self.I = I
        self.I_dur = I_dur
        self.ISIs = ISIs
        self.TR = TR
        self.K = K
        self.ar1_rho = ar1_rho
        self.smoothness=smoothness
        self.noise_factor = noise_factor
        self.cond_means = cond_means
        self.cond_stds = cond_stds
        self.single_trial = single_trial
        self.conf_params = conf_params

        self.X = None
        self.y = None
        self.conds = None
        self.conf = None
        self.V = None
        self.hrf = None

    def generate_data(self, X=None, single_trial=True):
        """ Generates timeseries fMRI data (X, y) with or without confound.

        Parameters
        ----------
        X : numpy array or None
            A numpy array of shape (timepoints x P) or (timepoints x (P*I)) for a
            single-trial design ("least-squares all", LSA) or None. When None,
            X is generated.
        single_trial : bool
            When False, the design has a separate regressor for each condition (p = 1 ... P).
            When True, the design has a separate regressor for each trial (i = 1 ... P * I)

        Returns
        -------
        X : numpy array
            A numpy array of shape (timepoints x P) or (timepoints x P*I)
        y : numpy array
            A numpy array of shape (timepoints x voxels)
        conds : numpy array
            A numpy array with condition labels (with P unique labels)
        true_betas : numpy array
            A numpy array with the true parameters used to generate the model
        """

        if X is None:  # Generate new X (else: use prespecified X)
            self._generate_X()

        self._generate_y()
        return self.X, self.y, self.conds, self.true_betas

    def _generate_conf(self, conds):

        weight = 0.0001
        desired_corr = self.conf_params['corr']
        conf = conds * weight + np.random.normal(0, 1, conds.size)
        this_corr = pearsonr(conf, conds)[0]
        while np.abs(this_corr - desired_corr) > 0.01:
            conf = conds * weight + np.random.normal(0, 1, conds.size)
            this_corr = pearsonr(conf, conds)[0]
            weight += 0.0001

        conf = (conf - conf.min()) / (conf.max() - conf.min())
        return conf

    def _generate_X(self):
        """ Generates X (design matrix). """

        single_trial = self.single_trial

        # Generate I trials for P conditions
        conds = np.tile(np.arange(self.P), self.I)
        conds = np.random.permutation(conds)  # shuffle trials

        if len(conds) % len(self.ISIs) != 0:
            raise ValueError("Please choose ISIs which can spread across trials evenly.")

        # Generate ISIs and shuffle
        ISIs = np.repeat(self.ISIs, len(conds) / len(self.ISIs))
        ISIs = np.random.permutation(ISIs)
        run_dur = (np.sum(ISIs) + self.I_dur * len(conds))  # run-duration

        osf = 10  # oversampling factor for onsets/hrf
        if single_trial:  # nr of regressors = conditions * trials
            X = np.zeros((run_dur * osf, self.P*self.I))
        else:  # nr regressors = nr conditions
            X = np.zeros((run_dur * osf, self.P))

        current_onset = 0  # start creating onsets
        for i, trial in enumerate(conds):

            if single_trial:
                X[current_onset:(current_onset + self.I_dur * osf), i] = 1
            else:
                X[current_onset:(current_onset + self.I_dur * osf), trial] = 1

            this_ITI = self.I_dur * osf + ISIs[i] * osf
            current_onset += this_ITI

        # Define HRF
        if self.hrf is None:
            hrf = spm_hrf(tr=self.TR, oversampling=self.TR*osf,
                          time_length=32.0, onset=0.0)
            hrf = hrf / np.max(hrf)  # scale HRF, peak = 1
        else:
            hrf = self.hrf

        # If confound model is 'additive', create a regressor based on confound
        # and add to X to take care of convolution (to be used later when "controlling"
        # for its influence)
        if self.conf_params is not None:
            conf = self._generate_conf(conds)
        else:
            conf = np.zeros(conds.size)

        conf_pred = np.zeros(run_dur * osf)
        conf_pred[X.sum(axis=1) != 0] = np.repeat(conf, repeats=osf)
        X = np.c_[X, conf_pred]

        # Convolve regressors with HRF
        X = np.hstack([np.convolve(X[:, i], hrf)[:run_dur*osf, np.newaxis]
                       for i in range(X.shape[1])])

        X = X[::self.TR*osf, :]  # downsample
        X = np.c_[np.ones(X.shape[0]), X]  # stack intercept

        self.X = X
        self.conds = conds
        self.conf = conf

    def _generate_y(self):
        """ Generate signals (y). """

        X, conds, single_trial = self.X, self.conds, self.single_trial
        if self.conf_params is None:
            conf_mean, conf_std = 0, 0
        else:
            conf_mean, conf_std = self.conf_params['mean'], self.conf_params['std']

        N = X.shape[0]  # N = (downsampled) timepoints

        # Create ar1 covariance matrix and generate noise
        if self.V is None:
            self.V = self._generate_V(N)

        noise = np.random.multivariate_normal(np.zeros(N), self.V,
                                              size=np.prod(self.K)).T
        # Create condition means/stds
        if single_trial:
            # Extract actual means/stds based on condition
            cond_means = np.array([self.cond_means[i] for i in conds])
            cond_stds = np.array([self.cond_stds[i] for i in conds])
        else:
            cond_means = self.cond_means
            cond_stds = self.cond_stds

        # Add conf effects
        cond_means = np.append(cond_means, conf_mean)
        cond_stds = np.append(cond_stds, conf_std)

        # Add intercept effects
        cond_means = np.append(0, cond_means)
        cond_stds = np.append(1, cond_stds)

        # Create true paramaters
        cov = np.eye(X.shape[1])
        cov *= cond_stds

        if single_trial:
            true_betas = np.random.multivariate_normal(cond_means.astype(float),
                                                       cov, size=np.prod(self.K)).T
        else:
            true_betas = np.random.multivariate_normal(cond_means, cov,
                                                       size=np.prod(self.K)).T

        # Create signal!
        y = X.dot(true_betas) + noise

        if self.smoothness is not None:  # Smooth if a smoothing kernel has been specified
            #y = gaussian_filter(y.reshape((y.shape[0],) + self.K),
            #                    self.smoothness).reshape((y.shape[0], np.prod(self.K)))
            for i in range(y.shape[0]):
                this_y = y[i, :].reshape(self.K)
                y[i, :] = gaussian_filter(this_y, self.smoothness).ravel()

        self.X = X
        self.y = y
        self.true_betas = true_betas

    def fit_glm(self, X=None, y=None, control_for_conf=False, method='LSA', remove_icept=True,
                aggressive=False):
        """ Fits a GLM (using generalized least squares).

        Parameters
        ----------
        X : numpy array or None
            If numpy array, X should be a (timepoints x conditions/trials) array.
            If None, the attribute `X` of the object will be used.
        y : numpy array or None
            If numpy array, y should be a (timepoints x voxels) array. If
            None, the attribute `y` of the object will be used.
        control_for_conf : bool
            Whether to control for the confound in the first-level model. Only
            relevant for when confound_mod='additive' was used in data generation.
        remove_icept : bool
            Whether to remove the intercept from the statistics (betas, t-values, etc.)
            that will be returned

        Returns
        -------
        est_betas : numpy array
            Array with *estimated* (as opposed to *true*) parameters.
        stderrs : numpy array
            Array with standard errors of the estimated parameters.
        tvals : numpy array
            Array with t-values of the estimated parameters (against baseline)
        """

        if X is None:
            X = self.X

        if y is None:
            y = self.y

        if control_for_conf:
            if aggressive:  # remove now
                conf = X[:, -1]
                gls_results = sm.GLS(y, X[:, -1, np.newaxis], sigma=self.V).fit()
                y = y - X[:, -1, np.newaxis].dot(gls_results.params)
                X = X[:, :-1]
            else:  # remove later (during model fit)
                X = X
        else:
            X = X[:, :-1]

        est_betas = np.zeros((X.shape[1], y.shape[1]))
        stderrs = np.zeros_like(est_betas)
        tvals = np.zeros_like(est_betas)

        for i in range(y.shape[1]):  # loop over voxels

            if method == 'LSA':
                gls_results = sm.GLS(y[:, i], X, sigma=self.V).fit()
                est_betas[:, i] = gls_results.params
                stderrs[:, i] = gls_results.bse
                tvals[:, i] = gls_results.tvalues
            elif method == 'LSS':

                if control_for_conf and not aggressive:
                    n_preds = X.shape[1] - 2
                else:
                    n_preds = X.shape[1] - 1

                for ii in range(n_preds):

                    icept = X[:, 0]
                    if control_for_conf and not aggressive:
                        conf = X[:, -1]
                        other_idx = np.ones(X.shape[1]).astype(bool)
                        other_idx[0] = False
                        other_idx[ii + 1] = False
                        other_preds = X[:, other_idx]
                        this_pred = X[:, ii + 1]
                        this_X = np.c_[icept, this_pred, other_preds, conf]
                    else:
                        other_idx = np.ones(X.shape[1]).astype(bool)
                        other_idx[0] = False
                        other_idx[ii + 1] = False
                        other_preds = X[:, other_idx].sum(axis=1)
                        this_pred = X[:, ii + 1]
                        this_X = np.c_[icept, this_pred, other_preds]

                    gls_results = sm.GLS(y[:, i], this_X, sigma=self.V).fit()
                    est_betas[ii+1, i] = gls_results.params[1]
                    stderrs[ii+1, i] = gls_results.bse[1]
                    tvals[ii+1, i] = gls_results.tvalues[1]

            else:
                raise ValueError("Choose either LSA or LSS")

        if control_for_conf and not aggressive:  # remove confound from estimated parameters
            est_betas = est_betas[:-1, :]
            stderrs = stderrs[:-1, :]
            tvals = tvals[:-1, :]

        if remove_icept:  # remove intercept from estimated parameters
            est_betas = est_betas[1:, :]
            stderrs = stderrs[1:, :]
            tvals = tvals[1:, :]

        return est_betas, stderrs, tvals

    def _generate_V(self, N):
        """ Generates a autocovariance matrix based on a AR(p) model. """
        rho = self.ar1_rho
        cov = rho ** scipy.linalg.toeplitz(np.arange(N))
        return cov * self.noise_factor

    def plot(self, X, y, conds, voxel=(0, 0)):
        """ Plots the design and signal of a particular voxel from a particular run. """
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5), sharex=True)

        ax[0].set_prop_cycle('color', [plt.cm.Dark2(i) for i in conds])
        ax[0].plot(X[:, 1:])
        ax[0].axhline(0, ls='--', c='k', lw=1)
        ax[0].set_xlim(0, X.shape[0])
        ax[0].set_xlabel('Time (in dynamics)', fontsize=15)
        ax[0].set_ylabel('Amplitude (a.u.)', fontsize=15)

        yresh = y.reshape((y.shape[0],) + self.K)
        ax[1].plot(yresh[:, voxel[0], voxel[1]])
        ax[1].axhline(0, ls='--', c='k', lw=1)
        ax[1].set_xlabel('Time (in dynamics)', fontsize=15)

        sns.despine()
        fig.tight_layout()
        fig.show()


def stack_runs(generator, R=10, control_for_conf=False, aggressive=False, smooth=None):

    all_betas, all_stderrs, all_tvals, all_conf = [], [], [], []
    for i in range(R):
        X, y, conds, true_params = generator.generate_data()

        if smooth is not None:
            y = np.hstack([gaussian_filter(y[:, i], sigma=smooth)[:, np.newaxis]
                           for i in range(y.shape[1])])

        betas, stderrs, tvals = fmri_gen.fit_glm(X, y, control_for_conf=control_for_conf,
                                                 aggressive=aggressive)
        all_betas.append(betas)
        all_stderrs.append(stderrs)
        all_tvals.append(tvals)
        all_conf.append([np.mean(generator.conf[conds == i]) for i in range(generator.P)])

    betas = np.vstack(all_betas)
    stderrs = np.vstack(all_stderrs)
    tvals = np.vstack(all_tvals)
    conf = np.concatenate(all_conf)

    if generator.single_trial:
        target = np.tile(conds, R)
        group = np.repeat(conds, R)
    else:
        target = np.tile(np.arange(generator.P), R)
        group = np.repeat(np.arange(generator.P), R)

    return betas, stderrs, tvals, target, group, conf

default_pipe = make_pipeline(StandardScaler(), SVC(kernel='linear'))
smooth_kernel = [0, 1, 2, 3, 4]
results = dict(acc=[], kernel=[], method=[], corr=[], analysis=[])
for i, sigma in tqdm(enumerate(smooth_kernel), desc='smooth'):

    for ii, corr in tqdm(enumerate(np.arange(0, 1.05, 0.2)), desc='corr'):
        iters = 25

        for ii in range(iters):
            fmri_gen = FmriData(P=2, I=40, I_dur=1, ISIs=[4,5], TR=2,
                            K=(4, 4), ar1_rho=0.5, smoothness=1,
                            noise_factor=1, cond_means=(0, 0),
                            cond_stds=(0.5, 0.5), single_trial=True,
                            conf_params=dict(corr=corr, mean=1, std=0))

            X, y, conds, true_params = fmri_gen.generate_data()
            y = np.hstack([gaussian_filter(y[:, i], sigma=sigma)[:, np.newaxis]
                           for i in range(y.shape[1])])

            betas, stders, tvals = fmri_gen.fit_glm(X, y)
            acc = cross_val_score(default_pipe, tvals, conds, cv=10).mean()
            results['kernel'].append(sigma)
            results['acc'].append(acc)
            results['method'].append('No control')
            results['corr'].append(corr)
            results['analysis'].append('Trial-wise')

            cfr_pipe = make_pipeline(ConfoundRegressor(X=tvals, confound=fmri_gen.conf),
                                     StandardScaler(), SVC(kernel='linear'))

            acc = cross_val_score(cfr_pipe, tvals, conds, cv=10).mean()
            results['kernel'].append(sigma)
            results['acc'].append(acc)
            results['method'].append('CVCR')
            results['corr'].append(corr)
            results['analysis'].append('Trial-wise')

            fmri_gen = FmriData(P=2, I=40, I_dur=1, ISIs=[4,5], TR=2,
                                K=(4, 4), ar1_rho=0.5, smoothness=1,
                                noise_factor=1, cond_means=(0, 0),
                                cond_stds=(0.5, 0.5), single_trial=True,
                                conf_params=None)

            X, y, conds, true_params = fmri_gen.generate_data()
            y = np.hstack([gaussian_filter(y[:, i], sigma=sigma)[:, np.newaxis]
                           for i in range(y.shape[1])])

            betas, stders, tvals = fmri_gen.fit_glm(X, y)
            acc = cross_val_score(default_pipe, tvals, conds, cv=10).mean()
            results['kernel'].append(sigma)
            results['acc'].append(acc)
            results['method'].append('Baseline')
            results['corr'].append(corr)
            results['analysis'].append('Trial-wise')
            ###################

            fmri_gen = FmriData(P=2, I=40, I_dur=1, ISIs=[4,5], TR=2,
                            K=(4, 4), ar1_rho=0.5, smoothness=1,
                            noise_factor=1, cond_means=(0, 0),
                            cond_stds=(0.5, 0.5), single_trial=False,
                            conf_params=dict(corr=corr, mean=1, std=0))

            _, _, tvals, target, group, conf = stack_runs(fmri_gen, smooth=sigma)
            acc = cross_val_score(default_pipe, tvals, target, cv=10, groups=group).mean()
            results['kernel'].append(sigma)
            results['acc'].append(acc)
            results['method'].append('No control')
            results['corr'].append(corr)
            results['analysis'].append('Run-wise')

            cfr_pipe = make_pipeline(ConfoundRegressor(X=tvals, confound=conf),
                                     StandardScaler(), SVC(kernel='linear'))

            acc = cross_val_score(cfr_pipe, tvals, target, cv=10, groups=group).mean()
            results['kernel'].append(sigma)
            results['acc'].append(acc)
            results['method'].append('CVCR')
            results['corr'].append(corr)
            results['analysis'].append('Run-wise')

            fmri_gen = FmriData(P=2, I=40, I_dur=1, ISIs=[4,5], TR=2,
                                K=(4, 4), ar1_rho=0.5, smoothness=1,
                                noise_factor=1, cond_means=(0, 0),
                                cond_stds=(0.5, 0.5), single_trial=False,
                                conf_params=None)

            _, _, tvals, target, group, conf = stack_runs(fmri_gen, smooth=sigma)
            acc = cross_val_score(default_pipe, tvals, target, cv=10, groups=group).mean()
            results['kernel'].append(sigma)
            results['acc'].append(acc)
            results['method'].append('Baseline')
            results['corr'].append(corr)
            results['analysis'].append('Run-wise')

df_CVCR_sm = pd.DataFrame(results)
df_CVCR_sm['corr'] = df_CVCR_sm['corr'].round(1)
df_CVCR_sm.to_csv('autocorr_results.tsv', sep='\t', index=False)

