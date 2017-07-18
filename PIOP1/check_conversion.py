import os
import os.path as op
from glob import glob
from tqdm import tqdm

base_dir = '/media/lukas/goliath/MVCA/PIOP1/bids'
func_files = sorted(glob(op.join(base_dir, 'sub*', 'func', '*.nii.gz')))

for func_file in tqdm(func_files):
    outpath = op.join(op.dirname(base_dir), 'raw', 'conversion_check', op.basename(func_file))
    cmd = 'fslroi %s %s 4 1' % (func_file, outpath)
    os.system(cmd)

