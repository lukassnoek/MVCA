import numpy as np
import pandas as pd
from scipy.ndimage import gaussian filter
from sklearn.preprocessing import StandardScaler


smooth_kernel = [0, 1, 2, 3, 4]
results = dict(acc=[], kernel=[], method=[], corr=[], analysis=[])

for i, sk in tqdm_notebook(enumerate(smooth_kernel)):
    
    for corr in tqdm_notebook(np.arange(0, 1.05, 0.1)):
        
        iters = 100
        for ii in range(iters):
            fmri_gen = FmriData(P=2, I=40, I_dur=1, ISIs=[4,5], TR=2,
                            K=(4, 4), ar1_rho=0.5, smoothness=1,
                            noise_factor=1, cond_means=(0, 0),
                            cond_stds=(0.5, 0.5), single_trial=True,
                            conf_params=dict(corr=corr, mean=1, std=0))

            X, y, conds, true_params = fmri_gen.generate_data()
            y = np.hstack([gaussian_filter(y[:, i], sigma=sk)[:, np.newaxis]
                           for i in range(y.shape[1])])
    
            betas, stders, tvals = fmri_gen.fit_glm(X, y)
            acc = cross_val_score(default_pipe, tvals, conds, cv=10).mean()
            results['kernel'].append(sk)
            results['acc'].append(acc)
            results['method'].append('No control')
            results['corr'].append(corr)
            results['analysis'].append('Trial-wise')

            cfr_pipe = make_pipeline(ConfoundRegressor(X=tvals, confound=fmri_gen.conf),
                                     StandardScaler(), SVC(kernel='linear'))

            acc = cross_val_score(cfr_pipe, tvals, conds, cv=10).mean()
            results['kernel'].append(sk)
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
            y = np.hstack([gaussian_filter(y[:, i], sigma=sk)[:, np.newaxis]
                           for i in range(y.shape[1])])
    
            betas, stders, tvals = fmri_gen.fit_glm(X, y)
            acc = cross_val_score(default_pipe, tvals, conds, cv=10).mean()
            results['kernel'].append(sk)
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

            _, _, tvals, target, group, conf = stack_runs(fmri_gen, smooth=sk)
            acc = cross_val_score(default_pipe, tvals, target, cv=10, groups=group).mean()
            results['kernel'].append(sk)
            results['acc'].append(acc)
            results['method'].append('No control')
            results['corr'].append(corr)
            results['analysis'].append('Run-wise')

            cfr_pipe = make_pipeline(ConfoundRegressor(X=tvals, confound=conf),
                                     StandardScaler(), SVC(kernel='linear'))

            acc = cross_val_score(cfr_pipe, tvals, target, cv=10, groups=group).mean()
            results['kernel'].append(sk)
            results['acc'].append(acc)
            results['method'].append('CVCR')
            results['corr'].append(corr)
            results['analysis'].append('Run-wise')
            
            fmri_gen = FmriData(P=2, I=40, I_dur=1, ISIs=[4,5], TR=2,
                                K=(4, 4), ar1_rho=0.5, smoothness=1,
                                noise_factor=1, cond_means=(0, 0),
                                cond_stds=(0.5, 0.5), single_trial=False,
                                conf_params=None)

            _, _, tvals, target, group, conf = stack_runs(fmri_gen, smooth=sk)
            acc = cross_val_score(default_pipe, tvals, target, cv=10, groups=group).mean()
            results['kernel'].append(sk)
            results['acc'].append(acc)
            results['method'].append('Baseline')
            results['corr'].append(corr)
            results['analysis'].append('Run-wise')
            
df_CVCR_sm = pd.DataFrame(results)
df_CVCR_sm['corr'] = df_CVCR_sm['corr'].round(1)
