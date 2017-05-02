import os
import numpy as np
import os.path as op
from glob import glob
from scipy.io import loadmat

base_dir = '/media/lukas/goliath/MVCA/ID1000'
raw_dir = op.join(base_dir, 'raw')

scan_infos = sorted(glob(op.join(raw_dir, 'ID*', '*DTI32', 'ScanInfo.mat')))

for scan_info in scan_infos:
    print(scan_info)
    si = loadmat(scan_info)
    bval = np.array(si['ScanInfo'][0][0][-1]).T
    bvec = np.array(si['ScanInfo'][0][0][-2]).T

    dwi_run = glob(op.join(op.dirname(scan_info), '*.nii.gz'))[0].split('run-')[-1].split('.')[0]

    subnr = op.basename(op.dirname(op.dirname(scan_info))).split('ID')[-1]
    bval_fn = op.join(op.dirname(scan_info), 'sub-%s_run-%s_dwi.bval' % (subnr, dwi_run))
    bvec_fn = op.join(op.dirname(scan_info), 'sub-%s_run-%s_dwi.bvec' % (subnr, dwi_run))
    np.savetxt(bval_fn, bval, delimiter=' ', fmt='%i')
    np.savetxt(bvec_fn, bvec, delimiter=' ', fmt='%.5f')

    bval_fn_bids = op.join(base_dir, 'bids', 'sub-%s' % subnr, 'dwi', op.basename(bval_fn))
    np.savetxt(bval_fn_bids, bval, delimiter=' ', fmt='%i')

    bvec_fn_bids = op.join(base_dir, 'bids', 'sub-%s' % subnr, 'dwi', op.basename(bvec_fn))
    np.savetxt(bvec_fn_bids, bvec, delimiter=' ', fmt='%.5f')

    bval_fn_bids = op.join(base_dir, 'derivatives_tbss', 'raw', op.basename(bval_fn))
    np.savetxt(bval_fn_bids, bval, delimiter=' ', fmt='%i')

    bvec_fn_bids = op.join(base_dir, 'derivatives_tbss', 'raw', op.basename(bvec_fn))
    np.savetxt(bvec_fn_bids, bvec, delimiter=' ', fmt='%.5f')

    bval_fn_bids = op.join(base_dir, 'derivatives_tbss', 'eddy_corrected', op.basename(bval_fn))
    np.savetxt(bval_fn_bids, bval, delimiter=' ', fmt='%i')

    bvec_fn_bids = op.join(base_dir, 'derivatives_tbss', 'eddy_corrected', op.basename(bvec_fn))
    np.savetxt(bvec_fn_bids, bvec, delimiter=' ', fmt='%.5f')