"""New names fixed for plots in the paper
"""
from collections import OrderedDict


def new_names_classifier():
    """ New names assigned to each classifier
    """
    new_names = OrderedDict([('ridge', 'Ridge'),
                             ('svc_l1', r'SVC-$\ell_1$'),
                             ('svc_l2', r'SVC-$\ell_2$')])
    return new_names


def new_names_atlas():
    """New names assigned to each atlas
    """
    new_names = OrderedDict([('ica', 'ICA'),
                             ('kmeans', 'K-Means'),
                             ('dictlearn', 'Online Dictionary \n Learning'),
                             ('ward', 'Ward'),
                             ('aal_spm12', 'AAL'),
                             ('basc', 'BASC'),
                             ('harvard_oxford', 'Harvard Oxford')])
    return new_names


def new_names_measure():
    """New names assigned to each measure
    """
    new_names = OrderedDict([('correlation', 'Correlation'),
                             ('partial correlation', 'Partial \n Correlation'),
                             ('tangent', 'Tangent')])
    return new_names

