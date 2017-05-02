"""A module to load timeseries dataset from csv paths

"""
import os
import glob
import collections
import pandas as pd
import numpy as np

from sklearn.datasets.base import Bunch

from nilearn.datasets.utils import _get_dataset_dir
from nilearn._utils.compat import _basestring


def load_abide(data_dir, site_id='all', read=False, verbose=1):
    """ Load ABIDE data timeseries extracted using MSDL atlas + compcor=10

    Parameters
    ----------
    data_dir : str
        Path to data. Base directory where it should contain folder named with
        'ABIDE'.

    site_id : str or list of str (case sensitive), optional='all'
        Site id within,
        'PITT', 'OLIN', 'OHSU', 'SDSU', 'TRINITY', 'UM_1', 'UM_2', 'USM',
        'YALE', 'CMU', 'LEUVEN_1', 'LEUVEN_2', 'KKI', 'NYU', 'STANFORD',
        'UCLA_1', 'UCLA_2', 'MAX_MUN', 'CALTECH', 'SBL'

        By default, data of all sites will be returned, site_id='all'.

        Total sites = 20

    read : bool
        Whether to read them or not using pandas.

    verbose : int
        Verbosity level

    Returns
    -------
    data : Bunch

    if read == False
        timeseries_paths : list of str
            Paths to csv contains timeseries data of each site.

        phenotypic_path : str
            Path to csv contains phenotypic data

    if read is set as True
        timeseries_data : list of numpy array
            Load them using pandas and convert to numpy arrays to be
            in compatible to nilearn and ConnectivityMeasure.

        file_ids : list of str
            Its file ids

        dx_groups : list of int
            Its DX_GROUP 1 is autism, 2 is control

        phenotypic_data : pandas Data
            Loaded phenotypic data
    """
    VALID_IDS = ['Pitt', 'Olin', 'OHSU', 'SDSU', 'Trinity', 'UM_1', 'UM_2',
                 'USM', 'Yale', 'CMU', 'Leuven_1', 'Leuven_2', 'KKI', 'NYU',
                 'Stanford', 'UCLA_1', 'UCLA_2', 'MaxMun', 'Caltech', 'SBL']
    dataset_name = 'ABIDE'

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    phenotypic_path = os.path.join(data_dir, 'Phenotypic_V1_0b_preprocessed1.csv')

    timeseries_name = 'timeseries'

    data_dir = _get_dataset_dir(timeseries_name, data_dir=data_dir,
                                verbose=verbose)
    paths = []

    if site_id == 'all':
        site_id = VALID_IDS

    if not isinstance(site_id, collections.Iterable):
        site_id = [site_id, ]

    if isinstance(site_id, collections.Iterable):
        for i, this_id in enumerate(site_id):
            print(i, this_id)
            if not isinstance(this_id, _basestring) \
                    or this_id not in VALID_IDS:
                raise ValueError('An invalid site_id={0} is provided. '
                                 'Valid site names are: {1}'
                                 .format(this_id, VALID_IDS))
            filepaths = glob.glob(os.path.join(data_dir, this_id + '*.csv'))
            paths.extend(filepaths)

    if read:
        phenotypic_data = pd.read_csv(phenotypic_path)
        timeseries_data = []
        file_ids = []
        dx_groups = []
        if len(paths) != 0:
            for path in paths:
                filename = os.path.splitext(os.path.split(path)[1])[0]
                this_id = filename.split('_timeseries')[0]
                file_ids.append(this_id)
                data = pd.read_csv(path)
                data = data.drop('Unnamed: 0', axis=1)
                timeseries_data.append(np.asarray(data))
                this_group = phenotypic_data[
                    phenotypic_data['FILE_ID'] == this_id]['DX_GROUP']
                dx_groups.append(this_group.values[0])
        return Bunch(timeseries_data=timeseries_data, file_ids=file_ids,
                     dx_groups=dx_groups, phenotypic_data=phenotypic_data)
    else:
        return Bunch(timeseries_paths=paths,
                     phenotypic_path=phenotypic_path)


def load_acpi(data_dir, site_id='all', read=False, verbose=1):
    """ Load ACPI data (timeseries) extracted using MSDL atlas + compcor=10

    Parameters
    ----------
    data_dir : str
        Path to data. Base directory where it should contain folder named with
        'ACPI'.

    site_id : int or list of int, optional='all'
        Site id within, [3, 9, 20, 190, 1, 5]

        By default, data of all sites will be returned, site_id='all'.

        Total sites = 6

    read : bool
        Whether to read them or not using pandas.

    verbose : int
        Verbosity level

    Returns
    -------
    data : Bunch

    if read == False
        timeseries_paths : list of str
            Paths to csv contains timeseries data of each site.

        phenotypic_path : str
            Path to csv contains phenotypic data

    if read is set as True
        timeseries_data : list of numpy array
            Load them using pandas and convert to numpy arrays to be
            in compatible to nilearn and ConnectivityMeasure.

        subject_ids : list of str
            Its subject ids

        dx_groups : list of int
            Its DX_GROUP (1 - MJUser, 0 - No MJ)

        phenotypic_data : pandas Data
            Loaded phenotypic data
    """
    VALID_IDS = [3, 9, 20, 190, 1, 5]
    dataset_name = 'ACPI'

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    phenotypic_path = os.path.join(data_dir, 'mta_1_phenotypic_data.csv')
    phenotypic_data = pd.read_csv(phenotypic_path)

    timeseries_name = 'timeseries'

    data_dir = _get_dataset_dir(timeseries_name, data_dir=data_dir,
                                verbose=verbose)
    paths = []

    if site_id == 'all':
        site_id = VALID_IDS

    if not isinstance(site_id, collections.Iterable):
        site_id = [site_id, ]

    if isinstance(site_id, collections.Iterable):
        for i, this_id in enumerate(site_id):
            if not isinstance(this_id, int) or this_id not in VALID_IDS:
                raise ValueError('An invalid site_id={0} is provided. '
                                 'Valid site names are: {1}'
                                 .format(this_id, VALID_IDS))
            file_ids = phenotypic_data[phenotypic_data['SITE_ID'] == this_id]
            file_ids = file_ids['SUBID'].values
            for this_file_id in file_ids:
                filepath = glob.glob(os.path.join(data_dir,
                                                  str(this_file_id)
                                                  + '_timeseries.csv'))
                paths.extend(filepath)

    if read:
        timeseries_data = []
        subject_ids = []
        dx_groups = []
        if len(paths) != 0:
            for path in paths:
                filename = os.path.splitext(os.path.split(path)[1])[0]
                this_id = int(filename.split('_timeseries')[0])
                subject_ids.append(this_id)
                data = pd.read_csv(path)
                data = data.drop('Unnamed: 0', axis=1)
                timeseries_data.append(np.asarray(data))
                this_group = phenotypic_data[
                    phenotypic_data['SUBID'] == this_id]['MJUser']
                dx_groups.append(this_group.values[0])
        return Bunch(timeseries_data=timeseries_data, subject_ids=subject_ids,
                     dx_groups=dx_groups, phenotypic_data=phenotypic_data)
    else:
        return Bunch(timeseries_paths=paths, phenotypic_path=phenotypic_path)


def load_adnidod(data_dir, read=False, verbose=1):
    """ Load ADNIDOD data (timeseries) extracted using MSDL atlas + (compcor=10
    and motion regressors).

    Parameters
    ----------
    data_dir : str
        Path to data. Base directory where it should contain folder named with
        'ADNIDOD'.

    read : bool
        Whether to read them or not using pandas.

    verbose : int
        Verbosity level

    Returns
    -------
    data : Bunch

    if read == False
        timeseries_paths : list of str
            Paths to csv contains timeseries data of each site.

        phenotypic_path : str
            Path to csv contains phenotypic data

    if read is set as True
        timeseries_data : list of numpy array
            Load them using pandas and convert to numpy arrays to be
            in compatible to nilearn and ConnectivityMeasure.

        scan_ids : list of str
            Its file ids

        dx_groups : list of int
            Its DX_GROUP (1 - PTSD, 0 - Control)

        phenotypic_data : pandas Data
            Loaded phenotypic data
    """
    dataset_name = 'ADNIDOD'

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    phenotypic_path = os.path.join(data_dir, 'adnidod_demographic.csv')
    phenotypic_data = pd.read_csv(phenotypic_path)

    timeseries_name = 'timeseries'

    data_dir = _get_dataset_dir(timeseries_name, data_dir=data_dir,
                                verbose=verbose)
    paths = glob.glob(os.path.join(data_dir, '*.csv'))

    if read:
        timeseries_data = []
        scan_ids = []
        dx_groups = []
        for path in paths:
            filename = os.path.splitext(os.path.split(path)[1])[0]
            this_id = filename.split('_timeseries')[0]
            scan_ids.append(this_id)
            data = pd.read_csv(path)
            data = data.drop('Unnamed: 0', axis=1)
            timeseries_data.append(np.asarray(data))
            this_group = phenotypic_data[
                phenotypic_data['ID_scan'] == this_id]['diagnosis']
            dx_groups.append(this_group.values[0])
        return Bunch(timeseries_data=timeseries_data, scan_ids=scan_ids,
                     dx_groups=dx_groups, phenotypic_data=phenotypic_data)
    else:
        return Bunch(timeseries_paths=paths, phenotypic_path=phenotypic_path)


def load_cobre(data_dir, read=False, verbose=1):
    """ Load COBRE data (timeseries) extracted using MSDL atlas + (compcor=10
    and motion regressors).

    Parameters
    ----------
    data_dir : str
        Path to data. Base directory where it should contain folder named with
        'COBRE'.

    read : bool
        Whether to read them or not using pandas.

    verbose : int
        Verbosity level

    Returns
    -------
    data : Bunch

    if read == False
        timeseries_paths : list of str
            Paths to csv contains timeseries data of each site.

        phenotypic_path : str
            Path to csv contains phenotypic data

    if read is set as True
        timeseries_data : list of numpy array
            Load them using pandas and convert to numpy arrays to be
            in compatible to nilearn and ConnectivityMeasure.

        scan_ids : list of str
            Its file ids

        dx_groups : list of int
            Its DX_GROUP Schizophrenia, Control, Bipolar, Schizoaffective

        phenotypic_data : pandas Data
            Loaded phenotypic data
    """
    dataset_name = 'COBRE'

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    phenotypic_path = os.path.join(data_dir,
                                   '1139_Cobre_Neuropsych_V2_20160607.csv')
    phenotypic_data = pd.read_csv(phenotypic_path)

    timeseries_name = 'timeseries'

    data_dir = _get_dataset_dir(timeseries_name, data_dir=data_dir,
                                verbose=verbose)
    paths = glob.glob(os.path.join(data_dir, '*.csv'))

    if read:
        timeseries_data = []
        file_ids = []
        dx_groups = []
        for path in paths:
            filename = os.path.splitext(os.path.split(path)[1])[0]
            this_id = filename.split('_timeseries')[0]
            file_ids.append(this_id)
            data = pd.read_csv(path)
            timeseries_data.append(np.asarray(data))
            this_group = phenotypic_data[
                (phenotypic_data['Unnamed: 0'] == this_id)]['Unnamed: 1']
            if np.any(this_group):
                dx_groups.append(this_group.values[0])
            else:
                dx_groups.append('Did not match')
        return Bunch(timeseries_data=timeseries_data, file_ids=file_ids,
                     dx_groups=dx_groups, phenotypic_data=phenotypic_data)
    else:
        return Bunch(timeseries_paths=paths, phenotypic_path=phenotypic_path)


def load_camcan(data_dir, session_id='all', read=False, verbose=1):
    """ Load CAMCAN data (timeseries) extracted using MSDL atlas + (compcor=10
    and motion regressors)

    Parameters
    ----------
    data_dir : str
        Path to data. Base directory where it should contain folder named with
        'camcan'.

    session_id : int or list of int, optional='all'
        Session within, [1, 2, 3, 4]

    read : bool
        Whether to read them or not using pandas.

    verbose : int
        Verbosity level

    Returns
    -------
    data : Bunch

    if read == False
        timeseries_paths : list of str
            Paths to csv contains timeseries data of each site.

        phenotypic_path : str
            Path to csv contains phenotypic data

    if read is set as True
        timeseries_data : list of numpy array
            Load them using pandas and convert to numpy arrays to be
            in compatible to nilearn and ConnectivityMeasure.

        subject_ids : list of str
            Its subject ids

        phenotypic_data : pandas Data
            Loaded phenotypic data
    """
    VALID_IDS = [1, 2, 3, 4]
    dataset_name = 'camcan'

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    phenotypic_path = os.path.join(data_dir, 'participant_data.csv')
    phenotypic_data = pd.read_csv(phenotypic_path)

    timeseries_name = 'timeseries'
    data_dir = _get_dataset_dir(timeseries_name, data_dir=data_dir,
                                verbose=verbose)
    paths = []
    timeseries_data = []
    subject_ids = []

    session_names = {1: 'cbuid280_sess1',
                     2: 'cbuid280_sess2',
                     3: 'cbuid280_sess3',
                     4: 'cbuid280_sess4'}

    if session_id == 'all':
        session_id = VALID_IDS

    if not isinstance(session_id, collections.Iterable):
        session_id = [session_id, ]

    if isinstance(session_id, collections.Iterable):
        for i, this_id in enumerate(session_id):
            print(this_id)
            if not isinstance(this_id, int) or this_id not in VALID_IDS:
                raise ValueError('An invalid session_id={0} is provided. '
                                 'Valid session ids are: {1}'
                                 .format(this_id, VALID_IDS))
            this_id_data = phenotypic_data[session_names[this_id]]
            this_id_data = this_id_data[this_id_data.notnull()]
            session_name = session_names[this_id]
            this_data_indices = this_id_data.index.values
            for index in this_data_indices:
                observation_id = phenotypic_data[
                    (phenotypic_data[session_name] == this_id_data[index])]['Observations']
                filepath = glob.glob(os.path.join(data_dir, 'sub-' + observation_id[index]
                                                  + '_timeseries.csv'))
                print(filepath)
                if len(filepath) != 0:
                    if read:
                        subject_ids.append(observation_id[index])
                        this_index_data = pd.read_csv(filepath[0])
                        timeseries_data.append(this_index_data)
                    else:
                        paths.extend(filepath)

    if read:
        return Bunch(timeseries_data=timeseries_data, subject_ids=subject_ids,
                     phenotypic_data=pd.read_csv(phenotypic_path))
    else:
        return Bunch(timeseries_paths=paths, phenotypic_path=phenotypic_path)


def load_camcan_all_without_sessions(data_dir, read=False, verbose=1):
    """Grab all timeseries paths of camcan data without any filtering.
    """
    dataset_name = 'camcan'

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    phenotypic_path = os.path.join(data_dir, 'participant_data.csv')
    phenotypic_data = pd.read_csv(phenotypic_path)

    timeseries_name = 'timeseries'
    data_dir = _get_dataset_dir(timeseries_name, data_dir=data_dir,
                                verbose=verbose)
    paths = os.path.join(data_dir, '*.csv')

    timeseries_paths = glob.glob(paths)

    if not read:
        return Bunch(timeseries_paths=timeseries_paths,
                     phenotypic_path=phenotypic_path)

    timeseries_data = []
    for path in timeseries_paths:
        data = pd.read_csv(path)
        data = data.drop('Unnamed: 0', axis=1)
        timeseries_data.append(data)

    return Bunch(timeseries_data=timeseries_data,
                 phenotypic_data=pd.read_csv(phenotypic_path))


def load_hcp(data_dir, session, session_type,
             atlas_name='msdl', verbose=1):
    """Load HCP timeseries data paths of "LR"

    Session we have 1 and 2 in integers which denotes REST1 and REST 2

    session_type we have is 'LR' and 'RL'
    """
    dataset_name = 'HCP'

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    paths = os.path.join(data_dir, '*')
    data_dir = os.path.join(data_dir, '*', atlas_name)

    if session == 1:
        if session_type == 'LR':
            filename_session = 'rfMRI_REST1_LR_raw'
        elif session_type == 'RL':
            filename_session = 'rfMRI_REST1_RL_raw'

    if session == 2:
        if session_type == 'LR':
            filename_session = 'rfMRI_REST2_LR_raw'
        elif session_type == 'RL':
            filename_session = 'rfMRI_REST2_RL_raw'

    paths = os.path.join(data_dir, filename_session)

    paths = glob.glob(paths)

    return paths


def load_hcp_confounds(data_dir, session, session_type, verbose=1):
    """Load confounds of HCP of "LR"

    Session we have 1 and 2 in integers which denotes REST1 and REST 2

    session_type we have is 'LR' and 'RL'
    """
    dataset_name = 'HCP'

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    paths = os.path.join(data_dir, '*')
    confound_name = 'confounds'
    data_dir = os.path.join(data_dir, '*', confound_name)

    if session == 1:
        if session_type == 'LR':
            filename_session = 'rfMRI_REST1_LR_Movement_Regressors.txt'
        elif session_type == 'RL':
            filename_session = 'rfMRI_REST1_RL_Movement_Regressors.txt'

    if session == 2:
        if session_type == 'LR':
            filename_session = 'rfMRI_REST2_LR_Movement_Regressors.txt'
        elif session_type == 'RL':
            filename_session = 'rfMRI_REST2_RL_Movement_Regressors.txt'

    paths = os.path.join(data_dir, filename_session)

    paths = glob.glob(paths)

    return paths
