"""Utilities developed based on Nilearn
"""

import collections
import warnings

import numpy as np
import pandas as pd

from sklearn.datasets.base import Bunch

from nilearn.image.resampling import coord_transform

from nilearn._utils.compat import get_affine, _basestring
from nilearn._utils import check_niimg
from nilearn._utils.niimg import _safe_get_data


def load_cut_coords(path):
    """Load data from csv and make coordinates to list of triplets

    Parameters
    ----------
    path : str
        Path to data.
        NOTE: only .csv files

    Returns
    -------
    coords : list
        List contains coordinates in triplets (x, y, z).

    data : pandas.DataFrame
        Loaded data from csv file
    """
    data = pd.read_csv(path)
    data = data.drop('Unnamed: 0', axis=1)

    coords = []
    for x, y, z in zip(data['x'], data['y'], data['z']):
        coord_ = (x, y, z)
        coords.append(coord_)

    return coords, data


def find_region_names_using_cut_coords(coords, atlas_img, labels=None):
    """Given list of MNI space coordinates, get names of the brain regions.

    Names of the brain regions are returned by getting nearest coordinates
    in the given `atlas_img` space iterated over the provided list of
    `coords`. These new image coordinates are then used to grab the label
    number (int) and name assigned to it. Last, these names are returned.

    Parameters
    ----------
    coords : Tuples of coordinates in a list
        MNI coordinates.

    atlas_img : Nifti-like image
        Path to or Nifti-like object. The labels (integers) ordered in
        this image should be sequential. Example: [0, 1, 2, 3, 4] but not
        [0, 5, 6, 7]. Helps in returning correct names without errors.

    labels : str in a list
        Names of the brain regions assigned to each label in atlas_img.
        NOTE: label with index 0 is assumed as background. Example:
            harvard oxford atlas. Hence be removed.

    Returns
    -------
    new_labels : int in a list
        Labels in integers generated according to correspondence with
        given atlas image and provided coordinates.

    names : str in a list
        Names of the brain regions generated according to given inputs.
    """
    if not isinstance(coords, collections.Iterable):
        raise ValueError("coords given must be a list of triplets of "
                         "coordinates in native space [(1, 2, 3)]. "
                         "You provided {0}".format(type(coords)))

    if isinstance(atlas_img, _basestring):
        atlas_img = check_niimg(atlas_img)

    affine = get_affine(atlas_img)
    atlas_data = _safe_get_data(atlas_img, ensure_finite=True)
    check_labels_from_atlas = np.unique(atlas_data)

    if labels is not None:
        names = []
        if not isinstance(labels, collections.Iterable):
            labels = np.asarray(labels)

    if isinstance(labels, collections.Iterable) and \
            isinstance(check_labels_from_atlas, collections.Iterable):
        if len(check_labels_from_atlas) != len(labels):
            warnings.warn("The number of labels provided does not match "
                          "with number of unique labels with atlas image.",
                          stacklevel=2)

    coords = list(coords)
    nearest_coordinates = []

    for sx, sy, sz in coords:
        nearest = np.round(coord_transform(sx, sy, sz, np.linalg.inv(affine)))
        nearest = nearest.astype(int)
        nearest = (nearest[0], nearest[1], nearest[2])
        nearest_coordinates.append(nearest)

    assert(len(nearest_coordinates) == len(coords))

    new_labels = []
    for coord_ in nearest_coordinates:
        # Grab index of current coordinate
        index = atlas_data[coord_]
        new_labels.append(index)
        if labels is not None:
            names.append(labels[index])

    if labels is not None:
        return new_labels, names
    else:
        return new_labels
