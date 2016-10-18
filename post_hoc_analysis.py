"""Post-hoc analysis scripts for resting state prediction scores
"""

import collections
import pandas as pd

from statsmodels.formula.api import ols, mixedlm

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


def _generate_formula(dep_variable, effect1, effect2, effect3,
                      target_in_effect1):
    """Generate formula for ols or mixedlm stats model

    Parameters
    ----------
    dep_variable : str
        First variable in the model

    effect1 : str
        First variable in the formula denoting the categorical effect against
        dep_variable.

    effect2 : str
        Second effect in the formula

    effect3 : str
        Third effect in the formula

    target_in_effect1 : str
        model name with respect to test_effect1.

    Returns
    -------
    formula : str
        Formula generated with given dependent and categorical variables.
    """
    formula = ("%s ~ C(%s, Sum('%s')) + " % (dep_variable, effect1, target_in_effect1)
               + " C(%s, Sum) " % effect2 + "+ C(%s, Sum) " % effect3)

    return formula


def analyze_effects(data, formula, model='ols', groups=None):
    """Measure the effects size of each categorical variables given in
    formula against dependent variable.

    Uses module smf.ols from stats model (statistics in Python)

    Parameters
    ----------
    data : pandas data frame

    formula : str
        Formula used in the specified model to fit model to the data.
        See documentation of statsmodels.formula.api or related examples.

    model : str, {'ols', 'mixedlm'}
        Imported from statsmodels.formula.api

    groups : str
        keyword argument passed to mixedlm model. Required specificall when
        you do mixedlm model tests.

    Returns
    -------
    model : instance of stats model whether 'ols' or 'mixedlm'
        fit() of the model on the data.
        print(model.summary()) to look at the summary of the fit() on data.
        params can be fetched as model.params
        pvalues can be fetched as model.pvalues
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Given input 'data' should be like pandas Data frame."
                         " You provided {0}".format(data))

    if model not in ['ols', 'mixedlm']:
        raise ValueError("model={0} you specified is not implemented. "
                         "Choose between 'ols' or 'mixedlm'".format(model))

    if model == 'ols':
        model_fit = ols(formula=formula, data=data).fit()
    elif model == 'mixedlm':
        model_fit = mixedlm(formula=formula, data=data,
                            groups=groups).fit()

    return model_fit
