import os
import os.path as op
from glob import glob
import numpy as np

vbm_dir = '/media/lukas/goliath/MVCA/PIOP1/derivatives/vbm'

# This is exactly the FSL-VBM command (fslvbm3b) that generates the
# mod_merg* file (in combination with fslmerge)
GM_files = sorted(glob(op.join(vbm_dir, 'struc', '*_GM_to_template_GM.nii.gz')))
sub_names = [op.basename(f).split('_')[0] for f in GM_files]
with open(op.join(vbm_dir, 'stats', 'sub_names.txt'), 'w') as fout:
    fout.write('\n'.join(sub_names))

tbss_dir = '/media/lukas/goliath/MVCA/PIOP1/derivatives/tbss/tbss_pipeline'

FA_files = sorted(glob(op.join(tbss_dir, 'FA', '*FA_FA.nii.gz')))
sub_names = [op.basename(f).split('_')[0] for f in FA_files]
with open(op.join(tbss_dir, 'stats', 'sub_names.txt'), 'w') as fout:
    fout.write('\n'.join(sub_names))


