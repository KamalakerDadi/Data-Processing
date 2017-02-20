"""Wrapper for Region Extractor with caching using joblib Memory
"""
import numbers
import collections
import numpy as np

from scipy import ndimage

from sklearn.externals.joblib import Memory, Parallel, delayed

from nilearn.regions import connected_regions, RegionExtractor
from nilearn.image import threshold_img, new_img_like

from nilearn._utils.cache_mixin import cache
from nilearn.regions.region_extractor import _threshold_maps_ratio
from nilearn._utils import check_niimg, check_niimg_3d
from nilearn._utils.niimg import _safe_get_data
from nilearn._utils.compat import _basestring
from nilearn._utils.niimg_conversions import concat_niimgs


def _region_extractor_cache(maps_img, mask_img=None, min_region_size=2500,
                            threshold=1., thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            memory=Memory(cachedir=None), memory_level=0):
    """Region Extraction with caching built upon helper functions

    See nilearn.regions.RegionExtractor for documentation and related
    """

    if thresholding_strategy == 'ratio_n_voxels':
        print("[Thresholding] using maps ratio with threshold={0}"
              " ".format(threshold))
        threshold_maps = _threshold_maps_ratio(maps_img, threshold)
    else:
        if thresholding_strategy == 'percentile':
            threshold = "{0}%".format(threshold)
            print("[Thresholding] using threshold_img with threshold={0}"
                  " ".format(threshold))
        threshold_maps = threshold_img(maps_img, mask_img=mask_img,
                                       threshold=threshold)
    print("Done thresholding")
    print("[Region Extraction] with {0}".format(extractor))
    regions_img, _ = cache(connected_regions, memory,
                           memory_level=memory_level,
                           func_memory_level=1)(threshold_maps,
                                                min_region_size=min_region_size,
                                                extract_type=extractor)
    return regions_img


def _remove_small_regions(input_data, mask_data, index,
                          affine, min_size):
    """Remove small regions in volume from input_data of specified min_size.
       min_size should be specified in mm^3 (region size in volume)

    Parameters
    ----------
    input_data : numpy.ndarray
        Values inside the regions defined by labels contained in input_data
        are summed together to get the size and compare with given min_size.
        For example, see scipy.ndimage.label

    mask_data : numpy.ndarray
        Data should contains labels assigned to the regions contained in given
        input_data to tag values to work on. The data must be of same shape of
        input_data.

    index : numpy.ndarray
        A sequence of label numbers of the regions to be measured corresponding
        to input_data. For example, sequence can be generated using
        np.arange(n_labels + 1)

    affine : numpy.ndarray
        Affine of input_data is used to convert size in voxels to size in
        volume of region in mm^3.

    min_size : float in mm^3
        Size of regions in input_data which falls below the specified min_size
        of volume in mm^3 will be discarded.

    Returns
    -------
    out : numpy.ndarray
        Data returned will have regions removed specified by min_size
        Otherwise, if criterion is not met then same input data will be
        returned.
    """
    # with return_counts argument is introduced from numpy 1.9.0.
    # _, region_sizes = np.unique(input_data, return_counts=True)

    # For now, to count the region sizes, we use return_inverse from
    # np.unique and then use np.bincount to count the region sizes.

    _, region_indices = np.unique(input_data, return_inverse=True)
    region_sizes = np.bincount(region_indices)
    size_in_vox = min_size / np.abs(np.linalg.det(affine[:3, :3]))
    labels_kept = region_sizes > size_in_vox
    if not np.all(labels_kept):
        # Put to zero the indices not kept
        rejected_labels_mask = np.in1d(input_data,
                                       np.where(np.logical_not(labels_kept))[0]
                                       ).reshape(input_data.shape)
        # Avoid modifying the input:
        input_data = input_data.copy()
        input_data[rejected_labels_mask] = 0
        # Reorder the indices to avoid gaps
        input_data = np.searchsorted(np.unique(input_data), input_data)
    return input_data


def connected_label_regions(labels_img, min_size=None, connect_diag=True,
                            labels=None):
    """ Extract connected regions from a brain atlas image defined by labels
    (integers)

    For each label in an parcellations, separates out connected
    components and assigns to each separated region a unique label.

    Parameters
    ----------
    labels_img : Nifti-like image
        A 3D image which contains regions denoted as labels. Each region
        is assigned with integers.

    min_size : float, in mm^3 optional (default None)
        Minimum region size in volume required to keep after extraction.
        Removes small or spurious regions.

    connect_diag : bool (default True)
        If 'connect_diag' is True, two voxels are considered in the same region
        if they are connected along the diagonal (26-connectivity). If it is
        False, two voxels are considered connected only if they are within the
        same x, y, or z direction.

    labels : 1D numpy array or list of str, (default None), optional
        Each string in a list or array denote the name of the brain atlas
        regions given in labels_img input. If provided, same names will be
        re-assigned corresponding to each connected component based extraction
        of regions relabelling. The total number of names should match with the
        number of labels assigned in the image.
        NOTE: The order of the names given in labels should be appropriately
        matched with the unique labels (integers) assigned to each region
        given in labels_img.

    Returns
    -------
    new_labels_img : Nifti-like image
        A new image comprising of regions extracted on an input labels_img.
    new_labels : list, optional
        If labels are provided, new labels assigned to region extracted will
        be returned. Otherwise, only new labels image will be returned.

    """
    labels_img = check_niimg_3d(labels_img)
    labels_data = _safe_get_data(labels_img, ensure_finite=True)
    affine = labels_img.get_affine()

    check_unique_labels = np.unique(labels_data)

    if min_size is not None and not isinstance(min_size, numbers.Number):
        raise ValueError("Expected 'min_size' to be specified as integer. "
                         "You provided {0}".format(min_size))
    if not isinstance(connect_diag, bool):
        raise ValueError("'connect_diag' must be specified as True or False. "
                         "You provided {0}".format(connect_diag))
    if np.any(check_unique_labels < 0):
        raise ValueError("The 'labels_img' you provided has unknown/negative "
                         "integers as labels {0} assigned to regions. "
                         "All regions in an image should have positive "
                         "integers assigned as labels."
                         .format(check_unique_labels))

    unique_labels = set(check_unique_labels)

    # check for background label indicated as 0
    if np.any(check_unique_labels == 0):
        unique_labels.remove(0)

    if labels is not None:
        if (not isinstance(labels, collections.Iterable) or
                isinstance(labels, _basestring)):
            labels = [labels, ]
        if len(unique_labels) != len(labels):
            raise ValueError("The number of labels: {0} provided as input "
                             "in labels={1} does not match with the number "
                             "of unique labels in labels_img: {2}. "
                             "Please provide appropriate match with unique "
                             "number of labels in labels_img."
                             .format(len(labels), labels, len(unique_labels)))
        new_names = []

    if labels is None:
        this_labels = [None] * len(unique_labels)
    else:
        this_labels = labels

    new_labels_data = np.zeros(labels_data.shape, dtype=np.int)
    current_max_label = 0
    for label_id, name in zip(unique_labels, this_labels):
        this_label_mask = (labels_data == label_id)
        # Extract regions assigned to each label id
        if connect_diag:
            structure = np.ones((3, 3, 3), dtype=np.int)
            regions, this_n_labels = ndimage.label(
                this_label_mask.astype(np.int), structure=structure)
        else:
            regions, this_n_labels = ndimage.label(this_label_mask.astype(np.int))

        if min_size is not None:
            index = np.arange(this_n_labels + 1)
            this_label_mask = this_label_mask.astype(np.int)
            regions = _remove_small_regions(regions, this_label_mask,
                                            index, affine, min_size=min_size)
            this_n_labels = regions.max()

        cur_regions = regions[regions != 0] + current_max_label
        new_labels_data[regions != 0] = cur_regions
        current_max_label += this_n_labels
        if name is not None:
            new_names.extend([name] * this_n_labels)

    new_labels_img = new_img_like(labels_img, new_labels_data, affine=affine)
    if labels is not None:
        new_labels = new_names
        return new_labels_img, new_labels

    return new_labels_img


def _region_extractor_labels_image(atlas, extract_type='connected_components',
                                   min_region_size=0):
    """ Function takes atlas image denoted as labels for each region
        and then imposes region extraction algorithm on the image to
        split them into regions apart.

    Parameters
    ----------
    atlas : 3D Nifti-like image
        An image contains labelled regions.

    extract_type : 'connected_components', 'local_regions'
        See nilearn.regions.connected_regions for full documentation

    min_region_size : in mm^3
        Minimum size of voxels in a region to be kept.

    """
    atlas_img = check_niimg(atlas)
    atlas_data = _safe_get_data(atlas_img)
    affine = atlas_img.get_affine()

    n_labels = np.unique(np.asarray(atlas_data))

    reg_imgs = []
    for label_id in n_labels:
        if label_id == 0:
            continue
        print("[Region Extraction] Processing with label {0}".format(label_id))
        region = (atlas_data == label_id) * atlas_data
        reg_img = new_img_like(atlas_img, region)
        regions, _ = connected_regions(reg_img, extract_type=extract_type,
                                       min_region_size=min_region_size)
        reg_imgs.append(regions)
    regions_extracted = concat_niimgs(reg_imgs)

    return regions_extracted, n_labels
