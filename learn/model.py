"""Pipeline for data model analysis
"""
import collections
import itertools
import warnings
import numpy as np

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import RidgeClassifier
from sklearn.covariance import LedoitWolf
from sklearn.utils.extmath import randomized_svd

from nilearn import masking
from nilearn import signal
from nilearn.input_data import MultiNiftiMasker, NiftiMapsMasker
from nilearn.decomposition import DictLearning, CanICA
from nilearn.regions import RegionExtractor
from nilearn.image import (high_variance_confounds, resample_img,
                           new_img_like)

from nilearn import _utils
from nilearn._utils.compat import _basestring, izip, get_affine
from nilearn._utils.class_inspect import get_params
from nilearn._utils.niimg_conversions import _check_same_fov

from parcel import Parcellations
from nilearn_regions import (_region_extractor_cache,
                             _region_extractor_labels_image)
from nilearn_utils import data_info
from base_atlas_masker import check_embedded_atlas_masker
from nilearn_connectome import ConnectivityMeasure

MODELS_CATALOG = ["dictlearn", "ica", "kmeans", "ward"]

ESTIMATOR_CATALOG = dict(
    svc_l1=LinearSVC(penalty='l1', dual=False, random_state=0),
    svc_l2=LinearSVC(penalty='l2', random_state=0),
    ridge=RidgeClassifier())


def compute_confounds(imgs, mask_img, n_confounds=5, get_randomized_svd=False,
                      compute_not_mask=False):
    """
    """
    confounds = []
    if not isinstance(imgs, collections.Iterable) or \
            isinstance(imgs, _basestring):
        imgs = [imgs, ]

    img = _utils.check_niimg_4d(imgs[0])
    shape = img.shape[:3]
    affine = get_affine(img)

    if isinstance(mask_img, _basestring):
        mask_img = _utils.check_niimg_3d(mask_img)

    if not _check_same_fov(img, mask_img):
        mask_img = resample_img(
            mask_img, target_shape=shape, target_affine=affine,
            interpolation='nearest')

    if compute_not_mask:
        print("Non mask based confounds extraction")
        not_mask_data = np.logical_not(mask_img.get_data().astype(np.int))
        whole_brain_mask = masking.compute_multi_epi_mask(imgs)
        not_mask = np.logical_and(not_mask_data, whole_brain_mask.get_data())
        mask_img = new_img_like(img, not_mask.astype(np.int), affine)

    for img in imgs:
        print("[Confounds Extraction] {0}".format(img))
        img = _utils.check_niimg_4d(img)
        print("[Confounds Extraction] high ariance confounds computation]")
        high_variance = high_variance_confounds(img, mask_img=mask_img,
                                                n_confounds=n_confounds)
        if compute_not_mask and get_randomized_svd:
            signals = masking.apply_mask(img, mask_img)
            non_constant = np.any(np.diff(signals, axis=0) != 0, axis=0)
            signals = signals[:, non_constant]
            signals = signal.clean(signals, detrend=True)
            print("[Confounds Extraction] Randomized SVD computation")
            U, s, V = randomized_svd(signals, n_components=n_confounds,
                                     random_state=0)
            if high_variance is not None:
                confound_ = np.hstack((U, high_variance))
            else:
                confound_ = U
        else:
            confound_ = high_variance
        confounds.append(confound_)

    return confounds


def _model_fit(imgs, model):
    """ Model to fit on the fMRI images

    imgs : list of Nifti images

    model : instance of model {'DictLearning', 'CanICA'}
        Deals with components_ inversion to image
    """
    model_ = clone(model)
    model_.fit(imgs)

    masker_ = model_.masker_
    components_img_ = masker_.inverse_transform(model_.components_)

    return components_img_, masker_


def _cluster_model_fit(imgs, model):
    """ Clustering model to fit on the fMRI images

    imgs : list of Nifti images

    model : instance of model {'KMeans', 'Ward'}
        Deals with labels_ inversion to image
    """
    model_ = clone(model)
    model_.fit(imgs)

    masker_ = model_.masker_
    if model_.algorithm == 'minibatchkmeans':
        labels_img = masker_.inverse_transform(model_.kmeans_labels_)
    elif model_.algorithm == 'featureagglomeration':
        # Avoid 0 label
        labels_img = masker_.inverse_transform(model_.ward_labels_ + 1)

    return labels_img, masker_


class LearnBrainRegions(BaseEstimator, TransformerMixin):
    """Learn brain regions on brain parcellations coming from data driven methods

    Data driven parcellation methods such as (kmeans, ward) or decomposition
    methods such as CanICA, DictLearning.

    Parameters
    ----------
    model : str {'kmeans', 'ward', 'ica', 'dictlearn'}
        Model should be used imposed on data for brain parcellations.
        Choose as a list or dict, if more than one model computation is
        required to be done.

    masker : instance/object of masker MultiNiftiMasker
        Masker parameters and mask image provided with masker will be used
        in data driven learning models.

    atlases : dict, {'atlase_name': NiftiImage or filename to NiftiImage}
        Predefined atlases to be added in the analysis pipeline. These atlases
        are used directly from timeseries extraction. No additional regions
        extraction will be done.

    connectome_convert : bool, default True
        If True, connectome matrices will be estimated on models whichever have
        been specified. Valid kind of measures are 'correlation',
        'partial correlation', 'tangent'.

    connectome_measure : list or dict of strings
        Different kind of measures, can be passed as a list or dict.
        Must be Iterable.
        See documentation of nilearn.connectome.ConnectivityMeasure

    connectome_confounds : csv file or array-like, optional
        Confounds to regress out unwanted effect from the connectivity
        coefficients. This is passed to signal.clean.
        See signal.clean documentation for more details.

    n_comp : int, default to 10
        Number of components to decompose. Used for 'dictlearn' or 'ica'

    n_parcels : int, default to 30
        Number of clusters/parcellations. Used for 'kmeans' or 'ward'

    smoothing_fwhm : float, default to 4.
        smoothing applied on the data.

    regions_extract : bool, default is True
        If True, fit() will return regions extracted on a n_components in
        attribute rois_. If False, components extracted will be found in
        attribute parcellations_.

    min_region_size : int, default to 2500 in mm^3
        Each region is determined by certain volume of voxels. Given size is
        converted to volume of voxels by its affine volume of a region which
        falls under given size will be kept.

    threshold : number default to 1.
        Please see related documentation of nilearn.regions.RegionExtractor

    thresholding_strategy : str {'ratio_n_voxels', 'img_value', 'percentile'}

    extractor : {'connected_components', 'local_regions'}
        Type of region extractor, simple label based extractor and other one
        random walker extractor.

    compute_confounds : None, 'compcor_5', 'compcor_10', default None
        Compute high variance signal confounds of n=5 or 10. None implies
        no confounds are computed and no confounds regression. These confounds
        are used in transform() and stacked with other confounds which are
        given at transform level. Otherwise leave to default None.

    compute_not_mask_confounds : None, 'compcor_5', 'compcor_10', default None
        To compute high variance confounds signal using CompCor and also
        computes signals using RandomizedSVD. Same number is given to both
        depending upon input. compcor_5 implies 5 confounds and compcor_10
        implies 10 confounds.
        Out of mask: For example, confounds from white matter or csf mask will
        be computed if grey matter mask is provided.
        These confounds will then be stacked with additional confounds if given
        and are used at transform() level for confounds signal cleaning.

    output_file : str
        Path name to store the accuracy scores onto csv file
    """

    def __init__(self, model, masker, atlases=None, connectome_convert=True,
                 connectome_measure=None, connectome_confounds=None,
                 n_comp=10, n_parcels=30,
                 smoothing_fwhm=4., regions_extract=True,
                 min_region_size=2500, threshold=1.,
                 thresholding_strategy='ratio_n_voxels',
                 extractor='local_regions', compute_confounds=None,
                 compute_not_mask_confounds=None, verbose=0,
                 output_file=None):
        self.model = model
        self.masker = masker
        self.atlases = atlases
        self.connectome_measure = connectome_measure
        self.connectome_convert = connectome_convert
        self.connectome_confounds = connectome_confounds
        self.n_comp = n_comp
        self.n_parcels = n_parcels
        self.smoothing_fwhm = smoothing_fwhm
        self.regions_extract = regions_extract
        self.min_region_size = min_region_size
        self.threshold = threshold
        self.thresholding_strategy = thresholding_strategy
        self.extractor = extractor
        self.compute_confounds = compute_confounds
        self.compute_not_mask_confounds = compute_not_mask_confounds
        self.verbose = verbose
        self.output_file = output_file

    def fit(self, imgs, y=None):
        """Fit the data to atlas decomposition and regions extraction

        Parameters
        ----------
        X : Nifti-like images, list

        y : None
            Fit for nothing. only for scikit learn compatibility
        """
        PARCELLATIONS = dict()
        if imgs is None or len(imgs) == 0:
            raise ValueError("You should provide a list of data e.g. Nifti1Image"
                             " or Nifti1Image filenames. None/Empty is provided")

        if not isinstance(imgs, collections.Iterable) \
                or isinstance(imgs, _basestring):
            imgs = [imgs, ]

        # Load data
        if self.verbose > 0:
            print("[%s.fit] Loading data from %s" % (
                self.__class__.__name__,
                _utils._repr_niimgs(imgs)[:200]))

        if not isinstance(self.masker, MultiNiftiMasker):
            raise ValueError("An instance of MultiNiftiMasker should be "
                             "provided from nilearn.input_data.MultiNiftiMasker")

        masker = clone(self.masker)

        valid_models = MODELS_CATALOG
        if isinstance(self.model, _basestring):
            self.model = [self.model]

        if isinstance(self.model, collections.Iterable):
            for model in self.model:
                if model not in valid_models:
                    raise ValueError("Invalid model='{0}' is chosen. Please "
                                     "choose one or more among them {1} "
                                     .format(self.model, valid_models))
                if model == 'dictlearn':
                    if self.verbose > 0:
                        print("[Dictionary Learning] Fitting the model")
                        dict_learn = DictLearning(
                            mask=masker, n_components=self.n_comp,
                            random_state=0, n_epochs=1,
                            memory=masker.memory, memory_level=masker.memory_level,
                            n_jobs=masker.n_jobs, verbose=masker.verbose)
                        # Fit Dict Learning model
                        dict_learn_img, masker_ = _model_fit(imgs, dict_learn)
                        # Gather results
                        PARCELLATIONS[model] = dict_learn_img
                        if self.verbose > 0:
                            print("[Dictionary Learning] Done")
                elif model == 'ica':
                    if self.verbose > 0:
                        print("[CanICA] Fitting the model")
                        canica = CanICA(n_components=self.n_comp, mask=masker,
                                        threshold=3., verbose=masker.verbose,
                                        random_state=0, memory=masker.memory,
                                        memory_level=masker.memory_level,
                                        n_jobs=masker.n_jobs)
                        # Fit CanICA model
                        canica_img, masker_ = _model_fit(imgs, canica)
                        # Gather results
                        PARCELLATIONS[model] = canica_img
                        if self.verbose > 0:
                            print("[CanICA Learning] Done")
                elif model == 'kmeans':
                    if self.verbose > 0:
                        print("[MiniBatchKMeans] Fitting the model")
                    kmeans = Parcellations(
                        algorithm='minibatchkmeans', n_parcels=self.n_parcels,
                        mask=masker, init='k-means++', verbose=masker.verbose,
                        memory=masker.memory, memory_level=masker.memory_level,
                        n_jobs=masker.n_jobs, random_state=0)
                    # Fit MiniBatchKmeans model
                    kmeans_img, masker_ = _cluster_model_fit(imgs, kmeans)
                    # Gather results
                    PARCELLATIONS[model] = kmeans_img
                    if self.verbose > 0:
                        print("[MiniBatchKMeans] Learning Done")
                elif model == 'ward':
                    if self.verbose > 0:
                        print("[Feature Agglomeration] Fitting the model")
                    ward = Parcellations(
                        algorithm='featureagglomeration',
                        n_parcels=self.n_parcels, mask=masker, linkage='ward',
                        verbose=masker.verbose, memory=masker.memory,
                        memory_level=masker.memory_level, random_state=0)
                    # Fit Feature Agglomeration ward linkage model
                    ward_img, masker_ = _cluster_model_fit(imgs, ward)
                    # Gather results
                    PARCELLATIONS[model] = ward_img
                    if self.verbose > 0:
                        print("[Feature Agglomeration (Ward)] Learning Done")
        # Gather all parcellation results into attribute parcellations_
        self.parcellations_ = PARCELLATIONS
        # If regions need to be extracted
        if self.regions_extract:
            if self.verbose > 0:
                print("[Region Extraction] Preparing images")
            self._regions_extract(masker_)

        return self

    def _regions_extract(self, masker):
        """Region Extraction
        """
        ROIS = dict()
        if not hasattr(self, 'model'):
            raise ValueError("Model selection is missing in Region Extraction")

        if hasattr(self, 'parcellations_'):
            parcellations = self.parcellations_
        else:
            raise ValueError("Could not find attribute 'parcellations_' "
                             "for fitting [Region Extraction]")

        for model in self.model:
            if self.verbose > 0:
                print("Model selected '{0}' for Region Extraction "
                      .format(model))

            parcel_img = parcellations[model]

            if model == 'kmeans' or model == 'ward':
                ROIS[model] = parcel_img
            else:
                try:
                    regions_img_ = _region_extractor_cache(
                        parcel_img, mask_img=masker.mask_img_,
                        min_region_size=self.min_region_size,
                        threshold=self.threshold,
                        thresholding_strategy=self.thresholding_strategy,
                        extractor=self.extractor)
                except:
                    print("Excepted as error. Running with 'connected_comp'")
                    regions_img_ = _region_extractor_cache(
                        parcel_img, mask_img=masker.mask_img_,
                        min_region_size=self.min_region_size,
                        threshold=self.threshold,
                        thresholding_strategy=self.thresholding_strategy,
                        extractor='connected_components')
                ROIS[model] = regions_img_

        # Gather all ROIS results into attribute rois_
        self.rois_ = ROIS

    def _check_fitted(self):
        """Checks if fit() method is executed or not
        """
        if hasattr(self, 'rois_') and hasattr(self, 'model'):
            if len(self.rois_) == len(self.model):
                for model in self.model:
                    if self.rois_[model] is None:
                        raise ValueError("{0} image from attribute rois_ is "
                                         "missing from the fit estimator. "
                                         "You must call fit() with list of fMRI"
                                         " images".format(model))
        else:
            raise ValueError("Could not find attribute 'rois_' or 'model'")

    def transform(self, imgs, confounds=None):
        """Signal extraction from regions learned on the images.

        Parameters
        ----------
        imgs : Nifti like images, list

        confounds : csv file or array-like, optional
            Contains signals like motion, high variance confounds from
            white matter, csf. This is passed to signal.clean
        """
        self._check_fitted()
        SUBJECTS_TIMESERIES = dict()
        models = []
        for model in self.model:
            models.append(model)

        # Getting Masker to transform fMRI images in Nifti to timeseries signals
        # based on atlas learning
        if self.masker is None:
            raise ValueError("Could not find masker attribute. Masker is missing")

        if not isinstance(imgs, collections.Iterable) or \
                isinstance(imgs, _basestring):
            imgs = [imgs, ]

        mask_img = self.masker.mask_img

        if self.compute_confounds not in ['compcor_5', 'compcor_10', None]:
            warnings.warn("Given invalid input compute_confounds={0}. Given "
                          "input is diverting to compute_confounds=None"
                          .format(self.compute_confounds))
            self.compute_confounds = None

        if self.compute_not_mask_confounds not in ['compcor_5', 'compcor_10', None]:
            warnings.warn("Invalid input type of 'compute_not_mask_confounds'={0}"
                          "is provided. Switching to None"
                          .format(self.compute_not_mask_confounds))
            self.compute_not_mask_confounds = None

        if confounds is None and self.compute_confounds is None and \
                self.compute_not_mask_confounds is None:
            confounds = [None] * len(imgs)

        if self.compute_confounds is not None:
            if self.compute_confounds == 'compcor_5':
                n_confounds = 5
            elif self.compute_confounds == 'compcor_10':
                n_confounds = 10

            confounds_ = self.masker.memory.cache(compute_confounds)(
                imgs, mask_img, n_confounds=n_confounds)

        if confounds_ is not None:
            if confounds is not None:
                confounds = np.hstack((confounds, confounds_))
            else:
                confounds = confounds_

        if confounds is not None and isinstance(confounds, collections.Iterable):
            if len(confounds) != len(imgs):
                raise ValueError("Number of confounds given doesnot match with "
                                 "the given number of subjects. Add missing "
                                 "confound in a list.")

        if self.atlases is not None and not \
                isinstance(self.atlases, dict):
            raise ValueError("If 'atlases' are provided, it should be given as "
                             "a dict. Example, atlases={'name': your atlas image}")

        if self.atlases is not None and \
                isinstance(self.atlases, dict):
            for key in self.atlases.keys():
                if self.verbose > 0:
                    print("Found Predefined atlases of name:{0}. Added to "
                          "set of models".format(key))
                self.parcellations_[key] = self.atlases[key]
                self.rois_[key] = self.atlases[key]
                models.append(key)

        self.models_ = models

        for model in self.models_:
            subjects_timeseries = []
            if self.verbose > 0:
                print("[Timeseries Extraction] {0} atlas image is selected"
                      .format(model))
            atlas_img = self.rois_[model]
            masker = check_embedded_atlas_masker(self.masker, atlas_type='auto',
                                                 img=atlas_img, t_r=2.53,
                                                 low_pass=0.1, high_pass=0.01)

            for img, confound in izip(imgs, confounds):
                if self.verbose > 0:
                    print("Confound found:{0} for subject:{1}".format(confound,
                                                                      img))

                signals = masker.fit_transform(img, confounds=confound)
                subjects_timeseries.append(signals)

            if subjects_timeseries is not None:
                SUBJECTS_TIMESERIES[model] = subjects_timeseries
            else:
                warnings.warn("Timeseries signals extraction are found empty "
                              "for model:{0}".format(model))

        self.subjects_timeseries_ = SUBJECTS_TIMESERIES

        if self.connectome_convert:
            if self.connectome_measure is not None:
                if isinstance(self.connectome_measure, collections.Iterable):
                    catalog = self.connectome_measure
                else:
                    if isinstance(self.connectome_measure, _basestring):
                        catalog = [self.connectome_measure, ]
            else:
                warnings.warn("Given connectome_convert=True but connectome "
                              "are given as None. Taking connectome measure "
                              "kind='correlation'", stacklevel=2)
                catalog = ['correlation']

            self.connectome_measures_ = catalog

            connectivities = self._connectome_converter(
                catalog=self.connectome_measures_,
                confounds=self.connectome_confounds)

        return self

    def fit_transform(self, imgs, confounds=None):
        """ Perform both fit() and transform() for pipeline consistency
        """
        return self.fit(imgs).transform(imgs, confounds=confounds)

    def _check_transformed(self):
        """Checks if transformed to signals or not, before estimate connectomes.
        """
        if hasattr(self, 'subjects_timeseries_'):
            if len(self.subjects_timeseries_) == len(self.models_):
                for model in self.models_:
                    if self.subjects_timeseries_[model] is None:
                        raise ValueError("subjects_timeseries_ for model {0} "
                                         "is missing from the transform() "
                                         "estimator. Please check if related "
                                         "model has fitted and exists in model"
                                         "selection list by printing caller "
                                         "name.model")
        else:
            raise ValueError("Could not find attribute 'subjects_timeseries_'. "
                             "You must call tranform() before converting to "
                             "connectome matrices.")

    def _connectome_converter(self, catalog=None, confounds=None):
        """Estimates functional connectivity matrices.

        Depending upon the models selected for brain parcellations, all models
        dependent subjects timeseries will be used as input for estimating
        the functional network interactions.

        This function first checks whether fit() followed by transform() or
        fit_transform() has been called or not to find related subjects
        timeseries signals.

        Parameters
        ----------
        confounds : csv file, numpy array like
            Confounds to regress out the effect such as gender, age, etc
            from group level connectivity coefficients.
        """
        CONNECTOMES = dict()
        if not hasattr(self, 'subjects_timeseries_'):
            raise ValueError("Could not find attribute 'subjects_timeseries_'. "
                             "Make sure to call transform() for subjects "
                             "timeseries signals to connectome matrices.")

        if catalog is None:
            warnings.warn("Catalog for connectivity measure is None. Taking "
                          "kind='correlation'", stacklevel=2)
            catalog = ['correlation']
            self.connectome_measures_ = catalog

        if len(self.subjects_timeseries_) == len(self.models_):
            for model in self.models_:
                coefs = dict()
                if self.verbose > 0:
                    print("[Timeseries signals] Loading data of model '{0}' "
                          .format(model))
                subjects_timeseries = self.subjects_timeseries_[model]
                for measure in catalog:
                    if self.verbose > 0:
                        print("[Connectivity Measure] kind='{0}'".format(measure))
                    # By default Ledoit Wolf covariance estimator
                    connections = ConnectivityMeasure(
                        cov_estimator=LedoitWolf(assume_centered=True),
                        kind=measure)
                    # By default vectorize is True
                    if self.connectome_confounds is not None and self.verbose > 0:
                        print("[Connectivity Coefficients] Regression")
                    conn_coefs = connections.fit_transform(subjects_timeseries,
                                                           confounds=confounds)
                    coefs[measure] = conn_coefs

                if coefs is not None:
                    CONNECTOMES[model] = coefs
                else:
                    warnings.warn("Conn coefs are found empty for model {0}"
                                  .format(model))
        self.connectomes_ = CONNECTOMES

        return self

    def classify(self, labels, cv, estimators=None, scoring='roc_auc'):
        """Prediction scores with estimators, LinearSVC, Ridge

        In LinearSVC, we use two penalities 'l1' and 'l2'.

        labels : numpy.ndarray

        cv :

        estimators : estimator or estimators in dictionary
            Classifier used in cross_val_score. It should be in this form.
            dictionary consists of estimator name and its object.
            Example: {'svc_l2': LinearSVC(penalty='l1')}
            If dictionary type is not given, then we take default estimator
            which is LinearSVC with penalty='l2'

        scoring : str
            Accepted types for cross_val_score.
            See documentation of sklearn.cross_validation.cross_val_score
        """
        SCORES = dict()
        if not hasattr(self, 'connectomes_'):
            raise ValueError("The attribute 'connectomes_' is not located.")

        if not hasattr(self, 'connectome_measures_'):
            raise ValueError("The attribute 'connectome_measures_' is not"
                             " located.")

        if self.connectome_measures_ is None:
            connectome_catalog = ['correlation']
        else:
            connectome_catalog = self.connectome_measures_

        if estimators is None:
            estimators = ESTIMATOR_CATALOG

        if estimators is not None:
            if not isinstance(estimators, dict):
                warnings.warn("Given estimator is not a dictionary type."
                              " Taking default estimator", stacklevel=2)
                estimators = {'svc_l2': LinearSVC(penalty='l2', random_state=0)}

        if len(self.connectomes_) == len(self.models_):
            for model in self.models_:
                scores = dict()
                for kind in connectome_catalog:
                    estimator_scores = dict()
                    if self.verbose > 0:
                        print("[Classify] with kind={0} of {1} atlas "
                              .format(kind, model))
                    connectivity_coefs = self.connectomes_[model][kind]
                    for est_key in estimators.keys():
                        if self.verbose > 0:
                            print("[Classifier] {0}".format(est_key))
                        estimator = estimators[est_key]
                        cv_scores = cross_val_score(estimator,
                                                    connectivity_coefs,
                                                    labels,
                                                    scoring=scoring,
                                                    cv=cv,
                                                    n_jobs=self.masker.n_jobs)
                        estimator_scores[est_key] = cv_scores

                    if estimator_scores is not None:
                        scores[kind] = estimator_scores
                    else:
                        warnings.warn("cv_scores are found empty for kind{0}"
                                      .format(kind), stacklevel=2)
                if scores is not None:
                    SCORES[model] = scores

            self.scores_ = SCORES

        return self
