import glob
import os
import collections
from os.path import join, split
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch

data_dir = '/media/kr245263/4C9A6E0E9A6DF53C/COBRE/preproc'
csv_dir = '/media/kr245263/4C9A6E0E9A6DF53C/COBRE/assessment_data'


def get_scores(sub_ids=[], csv_dir=csv_dir, csv_file=None):
    if csv_file is None:
        csv_file = '1139_Cobre_Neuropsych_V2_20160607.csv'
    df = pd.read_csv(join(csv_dir, csv_file))
    labels = ['Subject_id', 'Dx_group', 'Visit']
    for i, l in enumerate(labels):
        df = df.rename(columns={'Unnamed: %d' % i: l})
    df = df.set_index('Subject_id')
    if not sub_ids:
        return df
    return df.loc[sub_ids]


def _get_missing_ids(scores, subject_id):
    missing_indices = np.where(scores['Dx_group'].isnull())
    missing_indices = missing_indices[0]
    missing_ids = []
    for i in range(len(missing_indices)):
        missing_ids.append(subject_id[missing_indices[i]])

    return missing_ids, missing_indices


def _get_missing_scores(sub_ids=[], csv_dir=csv_dir):
    csv_file = '1139_Demographics_20160607.csv'
    scores = get_scores(sub_ids, csv_file=csv_file)

    return scores


def _replace_missing_with_new_scores(scores, missing_scores, missing_ids,
                                     missing_indices):
    if len(missing_ids) != len(missing_indices):
        raise ValueError("Number of missing ids submitted are not matching "
                         "with number of indices submitted....")
    for i, ind in enumerate(missing_indices):
        if scores['Dx_group'].index[ind] == missing_ids[i]:
            scores['Dx_group'][ind] = missing_scores['Dx_group'][i]

    return scores


def get_clinical_scores(sub_ids=[]):
    """Return clincal score for regression
    Some relevants scores :
    - BACS_SC_T-Score
    - BACS_SC_RawScore
    - Animal_Fluency_RawScore
    - Animal_Fluency_T-Score
    - MatricsDomain_OverallCompositeScore_T-Score
    - MatricsDomain_ProcessingSpeed_T-Score
    - TMT_A_T-Score
    - WAIS_PSI
    - WAIS_PSI_Sum_of_ScaleScores
    - WAIS_Coding_RawScore
    - WAIS_SymbolSearch_RawScore
    - WASI_Verbal_T-Score
    - WASI_Similarities_T-Score
    """

    keys = ['BACS_SC_T-Score', 'BACS_SC_RawScore', 'Animal_Fluency_RawScore',
            'Animal_Fluency_T-Score',
            'MatricsDomain_OverallCompositeScore_T-Score',
            'MatricsDomain_ProcessingSpeed_T-Score', 'TMT_A_T-Score',
            'WAIS_PSI', 'WAIS_PSI_Sum_of_ScaleScores', 'WAIS_Coding_RawScore',
            'WAIS_SymbolSearch_RawScore', 'WASI_Verbal_T-Score',
            'WASI_Similarities_T-Score']

    sc = get_scores(sub_ids)
    scores = {}
    scores = {k: sc[k].values for k in keys}
    return scores


def get_excluded_subjects():
    df = pd.read_csv(join(data_dir, 'cobre_qc.csv'))
    return df[df['large mvt'] == 1].image.values


def load_cobre(data_dir=data_dir, exclude_groups=None, get_missing=False):
    """Loads cobre dataset and its corresponding
    clinical scores
    """
    sub_dirs = sorted(glob.glob(join(data_dir, 'A*')))
    sub_ids = [split(s)[-1] for s in sub_dirs]

    excluded_subjects = get_excluded_subjects()
    if exclude_groups is not None:
        preliminary_scores = get_scores(sub_ids)
        if not hasattr(exclude_groups, '__iter__') or \
                not isinstance(exclude_groups, collections.Iterable):
            exclude_groups = [exclude_groups]
        for group_name in exclude_groups:
            find_indexes = np.where(preliminary_scores['Dx_group'] == group_name)
            for index in find_indexes[0]:
                exclude_id = sub_ids[index]
                excluded_subjects = np.append(excluded_subjects, exclude_id)

    subject_id, func, anat, tissues, motion_files = [], [], [], [], []
    for i, s in enumerate(sub_dirs):
        if sub_ids[i] in excluded_subjects:
            continue
        # functional
        f = join(s, 'func', 'wrrest.nii')
        if not os.path.isfile(f):
            print('%s not found' % f)
            continue
        # motion file
        m = join(s, 'func', 'rp_rest.txt')
        if not m:
            print('%s motion file not found' % m)
            continue
        # anat
        a = glob.glob(join(s, 'anat', 'w*'))
        if not a:
            print('%s anat not found' % s)
            continue
        # tissues
        t = sorted(glob.glob(join(s, 'anat', 'mwc*')))

        func.append(f)
        anat.append(a[0])
        tissues.append(t)
        subject_id.append(sub_ids[i])
        motion_files.append(m)

    scores = get_scores(subject_id)
    if get_missing:
        missing_ids, missing_indices = _get_missing_ids(scores, subject_id)
        missing_scores = _get_missing_scores(missing_ids)
        scores = _replace_missing_with_new_scores(scores, missing_scores,
                                                  missing_ids, missing_indices)

    dataset = {'func': func,
               'anat': anat,
               'tissues': tissues,
               'subject_id': subject_id,
               'dx_group': scores['Dx_group'].values,
               'motion_param': motion_files}
    return Bunch(**dataset)
