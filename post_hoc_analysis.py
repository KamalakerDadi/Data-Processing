"""Post-hoc analysis scripts for resting state prediction scores
"""

import collections
import pandas as pd

import statsmodels.formula.api as smf

from nilearn._utils.compat import _basestring


def _categorize_data(data, columns):
    """Categorize data according to the given columns.

    Parameters
    ----------
    data : path to csv filename or pandas data frame
        Data to build for post-hoc analysis

    columns : list of column names
        List of columns names used in building new pandas
        data frame from the given input data.

    Returns
    -------
    data : pandas data frame
        New data contains only respectively with columns.
    """
    data_dict = {}
    if isinstance(data, _basestring):
        data = pd.read_csv(data)

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Given input 'data' is not pandas like Data "
                         "Frame. Provide csv filename or pandas Data Frame"
                         " directly.")

    if not isinstance(columns, collections.Iterable):
        columns = [columns, ]

    for column in columns:
        if column not in data.columns:
            raise ValueError("Given column name is not matched with column "
                             "names in given input data.")
        data_dict[column] = data[column]

    data = pd.DataFrame(data_dict)

    return data
