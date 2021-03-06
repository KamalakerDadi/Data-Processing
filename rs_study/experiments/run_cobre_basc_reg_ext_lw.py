"""Processing pipeline example for resting state fMRI datasets
"""
import os
import itertools
import numpy as np
from nilearn.image import load_img


def _append_results(results, model, iteration, dim):
    """Gather results from a model which has attributes.

    Parameters
    ----------
    results : dict
        Should contain columns with empty array list for appending
        all the cross validation results for each iteration in
        cross_val_score
        {'atlas': [], 'classifier': [], 'measure': [], 'scores': []}
    model : object, instance of LearnBrainRegions

    Return
    ------
        results : dictionary
    """
    for atlas in model.models_:
        for measure in ['correlation', 'partial correlation', 'tangent']:
            for classifier in ['svc_l1', 'svc_l2', 'ridge']:
                results['iter_shuffle_split'].append(iteration)
                results['atlas'].append(atlas)
                results['measure'].append(measure)
                results['classifier'].append(classifier)
                results['scoring'].append('roc_auc')
                results['dataset'].append('COBRE')
                results['compcor_10'].append('yes')
                results['motion_regress'].append('yes')
                results['smoothing_fwhm'].append(6.)
                results['connectome_regress'].append('no')
                results['region_extraction'].append('yes')
                results['min_region_size_in_mm3'].append(1500)
                results['covariance_estimator'].append('LedoitWolf')
                score = model.scores_[atlas][measure][classifier]
                results['scores'].append(score)
                if atlas == 'basc':
                    rois_img = load_img(model.rois_[atlas])
                    n_regions = np.unique(rois_img.get_data())
                    n_regions = set(n_regions)
                    n_regions.remove(0)
                    results['n_regions'].append(len(n_regions))
                results['dimensionality'].append(dim)
                results['atlas_type'].append('pre-defined')
                results['version'].append('symmetric')

    return results


def draw_predictions(imgs=None, labels=None, groups=None, index=None,
                     train_index=None, test_index=None,
                     dimensionality=None,
                     scoring='roc_auc', models=None, atlases=None,
                     masker=None, connectomes=None,
                     confounds=None, confounds_mask_img=None,
                     connectome_regress_confounds=None):
    """
    """
    learn_brain_regions = LearnBrainRegions(
        model=models,
        atlases=atlases,
        masker=masker,
        regions_extract=False,
        connectome_convert=True,
        connectome_measure=connectomes,
        connectome_confounds=connectome_regress_confounds,
        compute_confounds='compcor_10',
        compute_confounds_mask_img=confounds_mask_img,
        compute_not_mask_confounds=None,
        covariance_estimator='LedoitWolf',
        verbose=2)
    print("Processing index={0}".format(index))

    train_data = [imgs[i] for i in train_index]

    # Fit the model to learn brain regions on the data
    learn_brain_regions.fit(train_data)

    # Tranform into subjects timeseries signals and connectomes
    # from a learned brain regions on training data. Now timeseries
    # and connectomes are extracted on all images
    learn_brain_regions.transform(imgs, confounds=confounds)

    # classification scores
    learn_brain_regions.classify(labels, cv=[(train_index, test_index)],
                                 groups=groups, scoring='roc_auc')
    print(learn_brain_regions.scores_)

    return learn_brain_regions

###########################################################################
# Data
# ----
# Load the datasets

from loader import load_cobre

data_path = '/media/kr245263/4C9A6E0E9A6DF53C/'
datasets_ = ['cobre']
groups = ['Bipolar', 'Schizoaffective']
data_store = dict()
cache = dict()
for path, dataset in zip(itertools.repeat(data_path), datasets_):
    print("Loading %s datasets from %s path" % (dataset, path))
    data_store[dataset] = load_cobre(get_missing=True,
                                     exclude_groups=groups)
    cache[dataset] = os.path.join(data_path, ('data_processing_' + dataset +
                                              '_analysis'))

# Data to process
name = 'cobre'
# class type for each subject is different
cache_path = cache[name]

from joblib import Memory, Parallel, delayed
mem = Memory(cachedir=cache_path)

func_imgs = data_store[name].func
phenotypic = data_store[name].dx_group
motion_confounds = data_store[name].motion_param
connectome_regress_confounds = None

from utils import data_info

shape, affine, _ = data_info(func_imgs[0])

###########################################################################
# Predefined Atlases
# ------------------
# Fetch the atlas
from nilearn import datasets as nidatasets

# By default we have atlas of version='SPM12'
basc = nidatasets.fetch_atlas_basc_multiscale_2015()

# Define atlases for LearnBrainRegions object as dict()
atlases = dict()

from region_extractor import connected_label_regions
relabelled_basc_36 = connected_label_regions(basc['scale036'],
                                             min_size=1500)
atlases['basc'] = relabelled_basc_36
###########################################################################
# Masker
# ------
# Masking the data

from nilearn import datasets

# Fetch grey matter mask from nilearn shipped with ICBM templates
gm_mask = datasets.fetch_icbm152_brain_gm_mask(threshold=0.2)

from nilearn.input_data import MultiNiftiMasker

masker = MultiNiftiMasker(mask_img=gm_mask, target_shape=shape,
                          target_affine=affine, smoothing_fwhm=6.,
                          standardize=True, detrend=True, mask_strategy='epi',
                          memory=mem, memory_level=2, n_jobs=5,
                          verbose=10)

##############################################################################
# Cross Validator
# ---------------

from sklearn.model_selection import StratifiedShuffleSplit

n_iter = 100
classes = phenotypic
_, labels = np.unique(classes, return_inverse=True)
cv = StratifiedShuffleSplit(n_splits=n_iter,
                            test_size=0.25, random_state=0)
##############################################################################
# Functional Connectivity Analysis model
# ---------------------------------------
from model import LearnBrainRegions

connectomes = ['correlation', 'partial correlation', 'tangent']

############################################################################
# Gather results - Data structure

columns = ['atlas', 'measure', 'classifier', 'scores', 'iter_shuffle_split',
           'n_regions', 'smoothing_fwhm', 'dataset', 'compcor_10',
           'motion_regress', 'dimensionality', 'connectome_regress', 'scoring',
           'region_extraction', 'covariance_estimator', 'min_region_size_in_mm3',
           'atlas_type', 'version']
results = dict()
for column_name in columns:
    results.setdefault(column_name, [])
print(results)

##############################################################################
# Run the analysis now
# --------------------
import pandas as pd

folder_name = name + str(n_iter) + '_basc_reg_ext__ledoitwolf'

dim = 36

iter_for_prediction = cv.split(func_imgs, classes)
for index, (train_index, test_index) in enumerate(iter_for_prediction):
    all_results = draw_predictions(
        imgs=func_imgs,
        labels=labels, groups=classes, index=index,
        train_index=train_index, test_index=test_index,
        scoring='roc_auc', models=None, atlases=atlases,
        masker=masker, connectomes=connectomes,
        confounds=motion_confounds,
        confounds_mask_img=gm_mask,
        connectome_regress_confounds=connectome_regress_confounds)
    print(index)
    results = _append_results(results, all_results, index, dim)

#################################################################################
# Frame the results into pandas Data Frame
# ----------------------------------------

results_csv = pd.DataFrame(results)
results_csv.to_csv(folder_name + '.csv')
print("Done................")
