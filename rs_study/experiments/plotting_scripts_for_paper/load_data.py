"""
"""
import pandas as pd


def _pandas_data_frame(path):
    """Load path to pandas data frame

    Parameters
    ----------
    path : str
        Path to csv file

    Returns
    -------
    data : pd.DataFrame
        Pandas data frame. If list of paths are provided
        list of pandas data frames are returned.
    """
    data = pd.read_csv(path)
    scores = data['scores'].str.strip('[ ]')
    del data['scores']
    data = data.join(scores)
    data.scores = data.scores.astype(float)

    return data


def _pandas_data_frame_list(paths):
    """Read path to pandas data frame.

    paths : list of str
        List of csv file paths

    Returns
    -------
    data : pd.DataFrame
        List of data frames
    """
    data = []
    if not isinstance(paths, (tuple, list)):
        paths = [paths, ]

    for path_ in paths:
        data.append(_pandas_data_frame(path_))

    return data


def _pandas_data_frame_list_of_paths_concat(paths):
    """Concat list of paths. Order of concat will be same as
    given order in paths as a list

    Parameters
    ----------
    paths : list of str
        List of csv file paths

    Returns
    -------
    data : pd.DataFrame
        Concatenated data
    """
    if not isinstance(paths, (list, tuple)):
        raise ValueError("You are looking for a wrong function."
                         "This function works only for given paths as"
                         "list to concat into pandas data frame.")

    data = _pandas_data_frame_list(paths)

    data = pd.concat(data)

    return data
