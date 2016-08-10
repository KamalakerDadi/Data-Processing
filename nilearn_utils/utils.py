"""Nilearn data utilities
"""

import numpy as np

from nilearn.image import load_img
from nilearn.input_data import NiftiMasker, MultiNiftiMasker

from nilearn._utils.niimg import _safe_get_data
from nilearn._utils.class_inspect import get_params


def data_info(img):
    """Tool to report the image data shape, affine, voxel size

    Parameters
    ----------
    img : Nifti like image/object

    Returns
    -------
    shape, affine, vox_size
    """
    img = load_img(img)
    img_data = _safe_get_data(img)

    if len(img.shape) > 3:
        shape = img.shape[:3]
    else:
        shape = img.shape

    affine = img.get_affine()
    vox_size = np.prod(np.diag(abs(affine[:3])))

    return shape, affine, vox_size


def params_masker(instance, estimator):
    """Get params of NiftiMasker or MultiNiftiMasker instance

    Parameters
    ----------
    instance : NiftiMasker or MultiNiftiMasker called object or instance

    """
    params = get_params(estimator, instance)

    if isinstance(instance, MultiNiftiMasker):
        multi_masker = True
