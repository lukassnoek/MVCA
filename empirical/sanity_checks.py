import pandas as pd
import numpy as np

behav = '/media/lukas/goliath/MVCA/PIOP1/PIOP1_behav_2017_with_brainsize.tsv'
behav = pd.read_csv(behav, sep='\t', index_col=0)
behav = behav.replace(9999, np.NaN)

print(behav[['brain_size', 'Gender', 'wm_score', 'Raven']].corr())