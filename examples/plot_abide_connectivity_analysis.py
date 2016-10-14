"""Processing pipeline example for resting state fMRI datasets
"""
import os
import itertools
import numpy as np


def _append_results(results, model):
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
                results['atlas'].append(atlas)
                results['measure'].append(measure)
                results['classifier'].append(classifier)
                score = model.scores_[atlas][measure][classifier]
                results['scores'].append(score)
    return results


def draw_predictions(imgs=None, labels=None, index=None,
                     train_index=None, test_index=None,
                     scoring='roc_auc', models=None, atlases=None,
                     masker=None, connectomes=None,
                     connectome_regress_confounds=None):
    """
    """
    learn_brain_regions = LearnBrainRegions(
        model=models,
        atlases=atlases,
        masker=masker,
        connectome_convert=True,
        connectome_measure=connectomes,
        connectome_confounds=connectome_regress_confounds,
        n_comp=40,
        n_parcels=120,
        compute_confounds='compcor_10',
        compute_confounds_mask_img=gm_mask,
        compute_not_mask_confounds=None,
        verbose=2)
    print("Processing index={0}".format(index))

    train_data = [imgs[i] for i in train_index]

    # Fit the model to learn brain regions on the data
    learn_brain_regions.fit(train_data)

    # Tranform into subjects timeseries signals and connectomes
    # from a learned brain regions on training data. Now timeseries
    # and connectomes are extracted on all images
    learn_brain_regions.transform(imgs, confounds=None)

    # classification scores
    # By default it used two classifiers, LinearSVC ('l1', 'l2') and Ridge
    # Not so good documentation and implementation here according to me
    learn_brain_regions.classify(labels, cv=[(train_index, test_index)],
                                 scoring='roc_auc')
    print(learn_brain_regions.scores_)

    return learn_brain_regions

###########################################################################
# Data
# ----
# Load the datasets from Nilearn

from nilearn import datasets

abide_data = datasets.fetch_abide_pcp(pipeline='cpac')
func_imgs = abide_data.func_preproc
phenotypic = abide_data.phenotypic

# class type for each subject is different
class_type = 'DX_GROUP'
cache_path = 'data_processing_abide'

from sklearn.externals.joblib import Memory, Parallel, delayed
mem = Memory(cachedir=cache_path)

connectome_regress_confounds = None

from nilearn_utils import data_info
target_shape, target_affine, _ = data_info(func_imgs[0])

###########################################################################
# Masker
# ------
# Masking the data

# Fetch grey matter mask from nilearn shipped with ICBM templates
gm_mask = datasets.fetch_icbm152_brain_gm_mask(threshold=0.2)

from nilearn.input_data import MultiNiftiMasker

masker = MultiNiftiMasker(mask_img=gm_mask, target_shape=target_shape,
                          target_affine=target_affine, smoothing_fwhm=6.,
                          standardize=True, detrend=True, mask_strategy='epi',
                          memory=mem, memory_level=2, n_jobs=2,
                          verbose=5)
##############################################################################
# Cross Validator
# ---------------

from sklearn.cross_validation import StratifiedShuffleSplit

classes = phenotypic[class_type].values
_, labels = np.unique(classes, return_inverse=True)
cv = StratifiedShuffleSplit(labels, n_iter=20, test_size=0.25, random_state=0)

##############################################################################
# Functional Connectivity Analysis model
# ---------------------------------------
from learn import LearnBrainRegions

models = ['ica', 'dictlearn']
connectomes = ['correlation', 'partial correlation', 'tangent']

###############################################################################
# Gather results - Data structure

columns = ['atlas', 'measure', 'classifier', 'scores']
gather_results = dict()
for label in columns:
    gather_results.setdefault(label, [])

##############################################################################
# Run the analysis now
# --------------------

# You can use Parallel if you want here!

for model in models:
    meta_results = Parallel(n_jobs=10, verbose=2)(
        delayed(draw_predictions)(
            imgs=func_imgs,
            labels=labels, index=index,
            train_index=train_index, test_index=test_index,
            scoring='roc_auc', models=model, atlases=None,
            masker=masker, connectomes=connectomes,
            connectome_regress_confounds=connectome_regress_confounds)
        for index, (train_index, test_index) in enumerate(cv))
    for i, meta_result_ in enumerate(meta_results):
        # This needs to be changed according to connectomes and classifiers
        # selected in the analysis.
        gather_results = _append_results(gather_results, meta_result_)

##############################################################################
# Frame the results into pandas Data Frame
# ----------------------------------------
import pandas as pd

results = pd.DataFrame(gather_results)
results.to_csv(cache_path + '.csv')
