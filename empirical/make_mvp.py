import os.path as op
import numpy as np
from skbold.core import MvpBetween

base_dir = '/media/lukas/goliath/MVCA/PIOP1'

subject_list = np.genfromtxt(op.join(base_dir, 'derivatives', 'vbm', 'stats', 'sub_names.txt'), dtype=str)

src = {}
src['4D_anat_VBM'] = {'path': op.join(base_dir, 'derivatives', 'vbm', 'stats', 'GM_mod_merg_s3.nii.gz'),
                      'subjects': subject_list}
subject_idf = 'sub-????'
remove_zeros = True

mvp = MvpBetween(source=src, subject_idf=subject_idf, remove_zeros=remove_zeros,
                 subject_list=subject_list)
mvp.create()
mvp.write('/media/lukas/goliath/MVCA/empirical', 'mvp_vbm')

subject_list = np.genfromtxt(op.join(base_dir, 'derivatives', 'tbss',
                                     'tbss_pipeline', 'stats', 'sub_names.txt'), dtype=str)

src = {}
src['4D_anat_TBSS'] = {'path': op.join(base_dir, 'derivatives', 'tbss', 'tbss_pipeline',
                                       'stats', 'all_FA_skeletonised.nii.gz'),
                      'subjects': subject_list}
subject_idf = 'sub-????'
remove_zeros = True

mvp = MvpBetween(source=src, subject_idf=subject_idf, remove_zeros=remove_zeros,
                 subject_list=subject_list)
mvp.create()
mvp.write('/media/lukas/goliath/MVCA/empirical', 'mvp_tbss')


