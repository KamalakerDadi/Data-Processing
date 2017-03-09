"""Load pipeline datasets (COBRE, ADNI, ACPI)
"""

import itertools
import datasets as p_datasets


def fetch(dataset_name, data_path):
    """Loads particular datasets depending upon the path and datset name.

    Parameters
    ----------
    dataset_name : {'cobre', 'adni', 'acpi'}

    data_path : path where data is stored or downloaded.
    """
    valid_names = ['cobre', 'adni', 'acpi', 'abide']
    if dataset_name not in valid_names:
        raise ValueError("Given dataset name {0} is invalid. Choose among "
                         "them {1}".format(dataset_name, valid_names))
    if dataset_name == 'cobre':
        data = p_datasets.fetch_cobre(data_dir=data_path)
    elif dataset_name == 'adni':
        data = p_datasets.fetch_adni(data_dir=data_path, ad_mci=True)
    elif dataset_name == 'acpi':
        data = p_datasets.fetch_acpi(data_dir=data_path)
    elif dataset_name == 'abide':
        data = p_datasets.fetch_abide(data_dir=data_path)

    return data
