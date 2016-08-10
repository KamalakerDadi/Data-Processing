"""A wrapper for clustering techniques from scikit-learn
"""

import numpy as np

from sklearn.cluster import MiniBatchKMeans, FeatureAgglomeration
from sklearn.feature_extraction import image
from sklearn.externals.joblib import Memory, delayed, Parallel

from nilearn.decomposition.base import BaseDecomposition
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn._utils.cache_mixin import CacheMixin


def _minibatch_kmeans_fit_method(data, n_clusters, init, random_state,
                                 verbose):
    """MiniBatchKMeans algorithm

    Parameters
    ----------
    data : array_like, shape=(n_samples, n_voxels)
        Masked subjects data

    n_clusters : int
        Number of clusters to parcellate.

    init : {'k-means++', 'random'}, default 'k-means++'
        Method for initialization
        k-means++ selects initial cluster centers for k-mean clustering in a
        smart way to speed up convergence. See section Notes in k_init for more
        details.
        random choose k observations (rows) at random from data for the initial
        centroids. If an ndarray is passed, it should be of shape
        (n_clusters, n_features) and gives the initial centers.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is given,
        it fixes the seed. Defaults to the global numpy random number generator.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

    Returns
    -------
    labels : Labels of each point
    """
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, init=init,
                             random_state=random_state, verbose=verbose)
    kmeans.fit(data)
    labels = kmeans.labels_

    return labels


def _feature_agglomeration_fit_method(data, n_clusters, connectivity, linkage):
    """Feature Agglomeration algorithm to fit on the data.

    Parameters
    ----------
    data : array_like, shape=(n_samples, n_voxels)
        Masked subjects data

    n_clusters : int
        Number of clusters to parcellate.

    connectivity : ndarray
        Connectivity matrix
        Defines for each feature the neighbouring features following a given
        structure of the data.

    linkage : str
        which linkage criterion to use.
        'ward' or 'linkage' or 'average'

    Returns
    -------
    labels : ndarray
        Labels to the data
    """
    ward = FeatureAgglomeration(n_clusters=n_clusters, connectivity=connectivity,
                                linkage=linkage)
    ward.fit(data)

    return ward.labels_


class Clustering(BaseDecomposition, CacheMixin):
    """CLustering techniques to decompose fMRI data into brain clusters.

    More specifically, MiniBatchKMeans and Feature Agglomeration algorithms
    can be used to learn parcellations from rs brain data. The alogrithms
    and its parameters are leveraged from scikit-learn. Parameters such as
    `linkage` for Feature Agglomeration, `init` for MiniBatchKMeans.

    Parameters
    ----------
    algorithm : str, {'minibatchkmeans', 'featureagglomeration'}
        An algorithm to choose between for brain parcellations.

    n_clusters : int, default=50
        Number of parcellations to divide the brain data into.

    linkage : str, {'ward', 'complete', 'average'}, default is 'ward'
        Which linkage criterion to use.
        - ward minimizes the variance of the clusters being merged.
        - average uses the average of the distances of each feature of the two
          sets.
        - complete or maximum linkage uses the maximum distances between all
          features of the two sets.
        Suitable for FeatureAgglomeration type of clustering method.

    init : str, {'k-means++', 'random'} default: 'k-means++'
        Method for initialization.
        - k-means++ selects initial cluster centers for k-mean clustering in
          a smart way to speed up convergence. See section Notes in k_init for
          more details.
        - random choose k observations (rows) at random from data for the
          initial centroids.
        Suitable for MiniBatchKMeans type of clustering method.

    connectivity : ndarray or callable, optional, Default is None.
        Connectivity matrix.
        Defines for each feature the neighboring features following a given
        structure of the data. This can be a connectivity matrix itself or a
        callable that transforms the data into a connectivity matrix, such as
        derived from kneighbors_graph.
        If connectivity is None, voxel-to-voxel connectivity matrix will be used.
        sklearn.feature_extraction.image.image_to_graph

    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    mask : Niimg-like object or MultiNiftiMasker instance
        Mask/Masker used for masking the data.
        If mask image if provided, it will be used in the MultiNiftiMasker.
        If an instance of MultiNiftiMasker is provided, then this instance
        parameters will be used in masking the data by overriding the default
        masker parameters.
        If None, mask will be automatically computed by a MultiNiftiMasker
        with default parameters.

    memory : instance of joblib.Memory or str
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level : integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

    Returns
    -------
    kmeans_labels_ : ndarray
        Labels to the parcellations if minibatchkmeans is selected.

    ward_labels_ : ndarray
        Labels to the parcellations of data if FeatureAgglomeration
        is selected.

    """
    VALID_ALGORITHMS = ["minibatchkmeans", "featureagglomeration"]

    def __init__(self, algorithm, n_clusters=50, linkage='ward',
                 init='k-means++', connectivity=None, random_state=0,
                 mask=None, target_affine=None, target_shape=None,
                 low_pass=None, high_pass=None, t_r=None,
                 smoothing_fwhm=None, standardize=False,
                 detrend=False, memory=Memory(cachedir=None),
                 memory_level=0, n_jobs=1, verbose=1):
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.init = init
        self.connectivity = connectivity
        self.random_state = random_state
        self.mask = mask
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, imgs, y=None, confounds=None):
        """ Fit the clustering technique to fmri images
        Parameters
        ----------
        X : List of Niimg-like objects
            Data from which clusters will be returned.
        """
        if self.algorithm is None:
            raise ValueError("Clustering algorithm must be specified in "
                             "['minibatchkmeans', 'featureagglomeration'].")

        valid_algorithms = self.VALID_ALGORITHMS
        if self.algorithm not in valid_algorithms:
            raise ValueError("Invalid algorithm={0} is provided. Please one "
                             "among them {1}".format(self.algorithm,
                                                     valid_algorithms))

        BaseDecomposition.fit(self, imgs)
        if not hasattr(imgs, '__iter__'):
            imgs = [imgs]
        data = self.masker_.fit_transform(imgs)
        data_ = np.vstack(data)
        if self.verbose:
            print("--- Fitting the data with Clustering algorithms ---")
        self._fit_method(data_)

        return self

    def _fit_method(self, data):
        """Helper function which applies clustering method on the masked data
        """
        mask_img = self.masker_.mask_img_

        if self.algorithm == 'minibatchkmeans':
            labels = self._cache(_minibatch_kmeans_fit_method,
                                 func_memory_level=1)(
                data.T, self.n_clusters, self.init, self.random_state,
                self.verbose)
            self.kmeans_labels_ = labels

        elif self.algorithm == 'featureagglomeration':
            mask = mask_img.get_data().astype(np.bool)
            shape = mask.shape
            if self.connectivity is None:
                self.connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                                        n_z=shape[2], mask=mask)
            labels = self._cache(_feature_agglomeration_fit_method,
                                 func_memory_level=1)(
                data, self.n_clusters, self.connectivity, self.linkage)

            self.ward_labels_ = labels
