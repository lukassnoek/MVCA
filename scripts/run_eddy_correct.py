import os
import os.path as op
from glob import glob
from tqdm import tqdm

base_dir = '/media/lukas/goliath/MVCA/ID1000/derivatives_tbss'
out_dir = op.join(base_dir, 'eddy_corrected')

files = sorted(glob(op.join(base_dir, 'raw', '*.nii.gz')))

for f in tqdm(files):
    new_fn = op.basename(f.split('.')[0]) + '_eddycorrected.nii.gz'
    cmd = 'eddy_correct %s %s 0' % (f, op.join(out_dir, new_fn))
    os.system(cmd)