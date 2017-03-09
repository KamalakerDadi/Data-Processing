"""Datasets fetcher from a given data directory
"""

import os
import re
import glob
import pandas
import numpy as np

from sklearn.datasets.base import Bunch

from nilearn import datasets
from nilearn._utils.compat import BytesIO


def fetch_cobre(data_dir=None, verbose=0, autosort=True, paranoid=False,
                **kwargs):
    """ Fetch cobre data depending upon the parameters

    if paranoid=True, only paranoid datasets will be fetched, excluding other
    categorical disease datasets
    if autosort=True, picks datasets of only subjects with right handedness
    and Quality controlled

    Manual tweaking is necessary not automatically operational.
    """

    dataset_name = 'COBRE'
    data_dir = datasets.utils._get_dataset_dir(dataset_name, data_dir=data_dir,
                                               verbose=verbose)

    csv = os.path.join(data_dir, 'COBRE_phenotypic_data_1.csv')
    csv = pandas.read_csv(csv)
    csv = csv[csv['Subject Type'] != 'Disenrolled']

    if paranoid:
        csv = csv[csv['Diagnosis'] != '295.6']
        csv = csv[csv['Diagnosis'] != '295.7']
        csv = csv[csv['Diagnosis'] != '295.9']

    if autosort:
        kwargs['QC_AA_KR'] = 1

    mask = datasets.utils._filter_columns(csv, kwargs)
    csv = csv[mask]

    ids = ['00' + str(i) for i in csv['Subject ID'].values]
    data_url = ''
    anats = [os.path.join(str(id_), 'session_1', 'anat_1',
                          'wmprage.nii.gz') for id_ in ids]
    rests = [os.path.join(str(id_), 'session_1', 'rest_1',
                          'wrarest.nii.gz') for id_ in ids]
    files_anats = [(path, data_url, {}) for path in anats]
    files_rests = [(path, data_url, {}) for path in rests]

    files_anats = datasets.utils._fetch_files(data_dir, files_anats, verbose=verbose)
    files_rests = datasets.utils._fetch_files(data_dir, files_rests, verbose=verbose)

    return Bunch(anatomical=files_anats, functional=files_rests,
                 phenotypic=csv)


def fetch_adni(data_dir=None, verbose=0, ad_mci=False, **kwargs):
    """Fetch ADNI datasets which are sorted according to Image_ID

    description_file2.csv - sorted with Image_ID

    Order them according to EXAM DATE and take first subjects

    If ad_mci is True, only ad versus mci subjects will be fetched.
    """
    dataset_name = 'ADNI_longitudinal_rs_fmri_DARTEL'
    data_dir = datasets.utils._get_dataset_dir(dataset_name, data_dir=data_dir,
                                               verbose=verbose)
    csv = os.path.join(data_dir, 'description_file.csv')
    csv = pandas.read_csv(csv)
    if ad_mci:
        csv = csv[csv['DX_Group'] != 'Normal']

    # Order them according to the EXAM DATE
    # csv = csv.sort('EXAM_DATE')

    # Take the first subject filenames by ordered according to EXAM_DATE
    seen_subject_id = set()
    filenames = []
    motion_filenames = []
    new_description = dict()
    new_description.setdefault('Image_ID', [])
    new_description.setdefault('DX_Group', [])
    new_description.setdefault('Subject_ID', [])
    new_description.setdefault('ACQ_DATE', [])
    new_description.setdefault('EXAM_DATE', [])
    new_description.setdefault('FILENAME', [])
    for sub_id in csv['Subject_ID']:
        if sub_id not in seen_subject_id:
            sub_data = csv[csv['Subject_ID'] == sub_id]
            if len(sub_data) > 1:
                sub_data = sub_data[:1]
            index = sub_data.index[0]
            seen_subject_id.add(sub_id)
            path = os.path.join(
                data_dir, sub_data['Image_ID'][index], 'func', 'resampled*.nii')
            motion_path = os.path.join(data_dir, sub_data['Image_ID'][index],
                                       'func', 'rp*.txt')
            filename = glob.glob(path)
            motion_filename = glob.glob(motion_path)
            if filename:
                filenames.append(filename[0])
                if motion_filename:
                    motion_filenames.append(motion_filename[0])
                else:
                    print('%s motion file not found' % motion_filename)
                new_description['Image_ID'].append(sub_data['Image_ID'][index])
                new_description['DX_Group'].append(sub_data['DX_Group'][index])
                new_description['Subject_ID'].append(sub_data['Subject_ID'][index])
                new_description['ACQ_DATE'].append(sub_data['ACQ_DATE'][index])
                new_description['EXAM_DATE'].append(sub_data['EXAM_DATE'][index])
                new_description['FILENAME'].append(sub_data['FILENAME'][index])

    return Bunch(func=filenames, motion_param=motion_filenames,
                 phenotypic=new_description)


def fetch_acpi(data_dir=None, verbose=0, **kwargs):
    """ For now, by default only ants registered data with no global
    signal regression and no scrubbing.
    """
    dataset_name = os.path.join('acpi', 'ants_s0_g0')

    data_dir = datasets.utils._get_dataset_dir(dataset_name, data_dir=data_dir,
                                               verbose=verbose)
    csv = os.path.join(data_dir, 'mta_1_phenotypic_data.csv')

    csv = pandas.read_csv(csv)
    ids = csv['SUBID']
    func = []
    ids = []
    filepath = os.path.join(data_dir, '00%i-session_1', 'rest_1',
                            'func_preproc', 'func_preproc.nii.gz')
    files = [filepath % id_ for id_ in csv['SUBID']]
    for i, file_ in zip(csv['SUBID'], files):
        if os.path.isfile(file_):
            func.append(file_)
            ids.append(i)
    mask = np.in1d(csv['SUBID'], ids)
    return Bunch(functional=func, phenotypic=csv[mask])


def fetch_abide(data_dir=None, verbose=0, **kwargs):
    """
    """
    exclude_ids = ['UM_1_0050289', 'Yale_0050571', 'KKI_0050822',
                   'SDSU_0050204', 'CMU_a_0050664']
    strategy = 'nofilt_noglobal'
    pipeline = 'cpac'

    dataset_name = 'ABIDE_pcp'
    csv = 'Phenotypic_V1_0b_preprocessed1.csv'

    kwargs['qc_rater_1'] = b'OK'
    kwargs['qc_anat_rater_2'] = [b'OK', b'maybe']
    kwargs['qc_func_rater_2'] = [b'OK', b'maybe']
    kwargs['qc_anat_rater_3'] = b'OK'
    kwargs['qc_func_rater_3'] = b'OK'

    path_csv = os.path.join(data_dir, dataset_name, csv)

    with open(path_csv, 'r') as pheno_f:
        pheno = ['i' + pheno_f.readline()]

        for line in pheno_f:
            pheno.append(re.sub(r',(?=[^"]*"(?:[^"]*"[^"]*")*[^"]*$)', ";", line))

    # bytes (encode()) needed for python 2/3 compat with numpy
    pheno = '\n'.join(pheno).encode()
    pheno = BytesIO(pheno)
    pheno = np.recfromcsv(pheno, comments='$', case_sensitive=True)

    # First, filter subjects with no filename
    pheno = pheno[pheno['FILE_ID'] != b'no_filename']
    # Apply user defined filters
    user_filter = datasets.utils._filter_columns(pheno, kwargs)
    pheno = pheno[user_filter]

    for id_ in exclude_ids:
        pheno = pheno[pheno['FILE_ID'] != id_]

    data_dir = os.path.join(data_dir, dataset_name, pipeline, strategy)

    results = {}
    file_ids = [file_id.decode() for file_id in pheno['FILE_ID']]

    ext = '.nii.gz'
    derivative = 'func_preproc'
    files = []

    for file_id in file_ids:
        file_ = (file_id + '_' + derivative + ext)
        check_file = os.path.join(data_dir, file_)
        if os.path.isfile(check_file):
            files.append(check_file)
        else:
            print("File is missing %s" % file_)

    results['phenotypic'] = pheno
    results[derivative] = files

    return Bunch(**results)
