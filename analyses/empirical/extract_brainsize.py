from tqdm import tqdm
import numpy as np
import os.path as op
import pandas as pd
from glob import glob
import nibabel as nib

behav_file = '/media/lukas/goliath/MVCA/PIOP1/PIOP1_behav_2017.tsv'
df = pd.read_csv(behav_file, sep='\t', index_col=0)
vbm_dir = '/media/lukas/goliath/MVCA/PIOP1/derivatives/vbm/struc'
GM_files = sorted(glob(op.join(vbm_dir, '*struc_GM.nii.gz')))

n_vox = np.zeros(len(GM_files))
subs = []
for i, f in tqdm(enumerate(GM_files)):
    img = nib.load(f).get_data()
    n_vox[i] = (img > 0).sum()
    subs.append(op.basename(f).split('_')[0])

brain_size = pd.DataFrame(n_vox, index=subs, columns=['brain_size'])
df = pd.concat((df, brain_size), axis=1, join='outer')
df.to_csv(op.join(op.dirname(behav_file), 'PIOP1_behav_2017_with_brainsize.tsv'),
          sep='\t')

