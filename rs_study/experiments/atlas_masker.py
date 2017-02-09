"""Function to embed standard parameters for Maps/Labels masker
"""
import warnings

import numpy as np

from nilearn._utils import check_niimg, check_niimg_3d, check_niimg_4d
from nilearn.input_data import (NiftiMapsMasker, NiftiLabelsMasker,
                                NiftiMasker, MultiNiftiMasker,
                                NiftiSpheresMasker)
from nilearn.input_data.masker_validation import check_embedded_nifti_masker
from nilearn._utils.class_inspect import get_params
from nilearn._utils.compat import _basestring


def check_embedded_atlas_masker(estimator, atlas_type=None, img=None,
                                multi_subject=True, seeds=None, radius=None,
                                t_r=None, low_pass=None, high_pass=None):
    """Base function to return masker type and its parameters

    Accepts all Nilearn masker types but returns only Maps or Labels masker.
    The idea being that this function returns an object with essential
    parameters embedded in it such as repetition time, low pass, high pass,
    standardize, detrend for resting state fMRI data analysis.

    Mostly useful for pipelined analysis.

    Parameters
    ----------
    estimator : object, instance of all masker types
        Accepts any instance masker type from nilearn.input_data

    img : maps_img or labels_img
        Used in initialization of instance masker object depending upon
        the length/type of the image. If maps_img related then used in
        NiftiMapsMasker instance or labels_img then NiftiLabelsMasker.

    seeds : List of triplet of coordinates in native space
        Used in NiftiSpheresMasker initialization.

    atlas_type : str {'maps', 'labels'}
        'maps' implies NiftiMapsMasker
        'labels' implies NiftiLabelsMasker

    multi_subject : bool, optional
        Indicates whether to return masker of multi subject type.
        List of subjects.

    Returns
    -------
    masker : NiftiMapsMasker, NiftiLabelsMasker
        Depending upon atlas type.

    params : dict
        Masker parameters
    """
    if atlas_type is not None:
        if atlas_type not in ['maps', 'labels', 'spheres', 'auto']:
            raise ValueError(
                "You provided unsupported masker type for atlas_type={0} "
                "selection. Choose one among them ['maps', 'labels', 'spheres']"
                "for atlas type masker. Otherwise atlas_type=None for general "
                "Nifti or MultiNifti Maskers.".format(atlas_type))

    if not isinstance(estimator, (NiftiMasker, MultiNiftiMasker, NiftiMapsMasker,
                                  NiftiLabelsMasker, NiftiSpheresMasker)):
        raise ValueError("Unsupported 'estimator' instance of masker is "
                         "provided".format(estimator))

    if atlas_type == 'spheres' and seeds is None:
        raise ValueError("'seeds' must be specified for atlas_type='spheres'."
                         "See documentation nilearn.input_data.NiftiSpheresMasker.")

    if (atlas_type == 'maps' or atlas_type == 'labels') and img is None:
        raise ValueError("'img' should not be None for atlas_type={0} related "
                         "instance of masker. Atlas related maskers is created "
                         "by provided a valid atlas image. See documenation in "
                         "nilearn.input_data for specific "
                         "masker related either maps or labels".format(atlas_type))

    if atlas_type == 'auto' and img is not None:
        img = check_niimg(img)
        if len(img.shape) > 3:
            atlas_type = 'maps'
        else:
            atlas_type = 'labels'

    if atlas_type == 'maps' and img is not None:
        img = check_niimg_4d(img)

    if atlas_type == 'labels' and img is not None:
        img = check_niimg_3d(img)

    new_masker = check_embedded_nifti_masker(estimator,
                                             multi_subject=multi_subject)
    mask = getattr(new_masker, 'mask_img', None)
    estimator_mask = getattr(estimator, 'mask_img', None)
    if mask is None and estimator_mask is not None:
        new_masker.mask_img = estimator.mask_img

    if atlas_type is None:
        return new_masker
    else:
        masker_params = new_masker.get_params()
        new_masker_params = dict()
        _ignore = set(('mask_strategy', 'mask_args', 'n_jobs', 'target_affine',
                       'target_shape'))
        for param in masker_params:
            if param in _ignore:
                continue
            if hasattr(new_masker, param):
                new_masker_params[param] = getattr(new_masker, param)
        # Append atlas extraction related parameters
        if t_r is not None:
            new_masker_params['t_r'] = t_r

        if low_pass is not None:
            new_masker_params['low_pass'] = low_pass

        if high_pass is not None:
            new_masker_params['high_pass'] = high_pass

        if atlas_type is not None:
            if len(img.shape) > 3 and atlas_type == 'maps':
                new_masker_params['maps_img'] = img
                new_masker = NiftiMapsMasker(**new_masker_params)
            elif len(img.shape) == 3 and atlas_type == 'labels':
                new_masker_params['labels_img'] = img
                new_masker = NiftiLabelsMasker(**new_masker_params)

            return new_masker

        if seeds is not None and atlas_type == 'spheres':
            if radius is not None:
                new_masker_params['radius'] = radius
            new_masker_params['seeds'] = seeds
            new_masker = NiftiSpheresMasker(**new_masker_params)

        return new_masker
