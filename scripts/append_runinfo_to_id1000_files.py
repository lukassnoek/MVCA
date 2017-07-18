import os
import os.path as op
from glob import glob

base_dir = '/media/lukas/goliath/MVCA/ID1000/raw'
subs = sorted(glob(op.join(base_dir, 'ID????')))

for sub in subs:

    t1s = sorted(glob(op.join(sub, '*T13D*', '*T13D.nii.gz')))
    dtis = sorted(glob(op.join(sub, '*DTI32*', '*DWI.nii.gz')))

    for i, t1 in enumerate(t1s):
        new = t1.split('.nii.gz')[0] + '_run-%i.nii.gz' % (i+1)
        cmd = 'mv %s %s' % (t1, new)
        os.system(cmd)

    for i, dti in enumerate(dtis):
        new = dti.split('.nii.gz')[0] + '_run-%i.nii.gz' % (i + 1)
        cmd = 'mv %s %s' % (dti, new)
        os.system(cmd)


