"""
Downloading NeuroImaging datasets: functional datasets (task + resting-state)
"""
import warnings
import os
import re
import json
import numpy as np
import numbers

import nibabel
from sklearn.datasets.base import Bunch
from sklearn.utils import deprecated

from .utils import (_get_dataset_dir, _fetch_files, _get_dataset_descr,
                    _read_md5_sum_file, _tree, _filter_columns)
from .._utils import check_niimg
from .._utils.compat import BytesIO, _basestring, _urllib, get_affine
from .._utils.numpy_conversions import csv_to_array


def fetch_cobre(n_subjects=10, data_dir=None, url=None, verbose=1):
    """Fetch COBRE datasets preprocessed using NIAK 0.12.4 pipeline.

    Downloads and returns preprocessed resting state fMRI datasets and
    phenotypic information such as demographic, clinical variables,
    measure of frame displacement FD (an average FD for all the time
    frames left after censoring).

    For each subject, this function also returns .mat files which contains
    all the covariates that have been regressed out of the functional data.
    The covariates such as motion parameters, mean CSF signal, etc. It also
    contains a list of time frames that have been removed from the time series
    by censoring for high motion.

    NOTE: The number of time samples vary, as some samples have been removed
    if tagged with excessive motion. This means that data is already time
    filtered. See output variable 'description' for more details.

    .. versionadded 0.2.3

    Parameters
    ----------
    n_subjects: int, optional
        The number of subjects to load from maximum of 146 subjects.
        By default, 10 subjects will be loaded. If n_subjects=None,
        all subjects will be loaded.

    data_dir: str, optional
        Path to the data directory. Used to force data storage in a
        specified location. Default: None

    url: str, optional
        Override download url. Used for test only (or if you setup a
        mirror of the data). Default: None

    verbose: int, optional
       Verbosity level (0 means no message).

    Returns
    -------
    data: Bunch
        Dictionary-like object, the attributes are:

        - 'func': string list
            Paths to Nifti images.
        - 'mat_files': string list
            Paths to .mat files of each subject.
        - 'phenotypic': ndarray
            Contains data of clinical variables, sex, age, FD.
        - 'description': data description of the release and references.

    Notes
    -----
    More information about datasets structure, See:
    https://figshare.com/articles/COBRE_preprocessed_with_NIAK_0_12_4/1160600
    """
    if url is None:
        # Here we use the file that provides URL for all others
        url = "https://figshare.com/api/articles/4197885/1/files"

    dataset_name = 'cobre'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    fdescr = _get_dataset_descr(dataset_name)

    # First, fetch the file that references all individual URLs
    files = _fetch_files(data_dir,
                         [("files", url + "?offset=0&limit=300", {})],
                         verbose=verbose)[0]
    files = json.load(open(files, 'r'))
    # Index files by name
    files_ = {}
    for f in files:
        files_[f['name']] = f
    files = files_

    # Fetch the phenotypic file and load it
    csv_name = 'phenotypic_data.tsv.gz'
    csv_file = _fetch_files(
        data_dir, [(csv_name, files[csv_name]['downloadUrl'],
                    {'md5': files[csv_name].get('md5', None),
                     'move': csv_name})],
        verbose=verbose)[0]

    # Load file in filename to numpy arrays
    names = ['id', 'current_age', 'gender', 'handedness', 'subject_type',
             'diagnosis', 'frames_ok', 'fd', 'fd_scrubbed']
    csv_array = np.recfromcsv(csv_file, names=names,
                              skip_header=True, delimiter='\t')

    # Check number of subjects
    max_subjects = len(csv_array)
    if n_subjects is None:
        n_subjects = max_subjects

    if n_subjects > max_subjects:
        warnings.warn('Warning: there are only %d subjects' % max_subjects)
        n_subjects = max_subjects

    _, labels = np.unique(csv_array['subject_type'], return_inverse=True)

    n_sz = np.ceil(float(n_subjects) / max_subjects * labels.sum())
    n_ct = np.floor(float(n_subjects) / max_subjects *
                    np.logical_not(labels).sum())

    # First, restrict the csv files to the adequate number of subjects
    sz_ids = csv_array[csv_array['subject_type'] == 'Patient']['id'][:n_sz]
    ct_ids = csv_array[csv_array['subject_type'] == 'Control']['id'][:n_ct]
    ids = np.hstack([sz_ids, ct_ids])
    csv_array = csv_array[np.in1d(csv_array['id'], ids)]

    # Call fetch_files once per subject.
    func = []
    tsv = []
    for i in ids:
        f = 'fmri_' + '00' + str(i) + '.nii.gz'
        t = 'fmri_' + '00' + str(i) + '.tsv.gz'
        f, t = _fetch_files(
            data_dir,
            [(f, files[f]['downloadUrl'], {'md5': files[f].get('md5', None),
                                           'move': f}),
             (t, files[t]['downloadUrl'], {'md5': files[t].get('md5', None),
                                           'move': t})
             ],
            verbose=verbose)
        func.append(f)
        tsv.append(t)

    return Bunch(func=func, tsv_files=tsv, phenotypic=csv_array,
                 description=fdescr)
