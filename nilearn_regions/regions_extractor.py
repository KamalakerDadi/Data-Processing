"""Wrapper for Region Extractor with caching using joblib Memory
"""
import numpy as np

from sklearn.externals.joblib import Memory, Parallel, delayed

from nilearn.regions import connected_regions, RegionExtractor
from nilearn.image import threshold_img, new_img_like

from nilearn._utils.cache_mixin import cache
from nilearn.regions.region_extractor import _threshold_maps_ratio
from nilearn._utils import check_niimg
from nilearn._utils.niimg import _safe_get_data
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
