"""
"""
import glob

import pandas as pd

from post_hoc_analysis import _categorize_data

path_to_csvs = sorted(glob.glob('results_csv/results_*.csv'))
data_list = []

for csv in path_to_csvs:
    data_name = csv.split('/')[1].split('_')[1].split('.csv')[0]
    print(data_name)
    data = pd.read_csv(csv)

    scores = data['scores'].str.strip('[ ]')
    del data['scores']
    data = data.join(scores)
    data.scores = data.scores.astype(float)

    # Categorize data with columns to pandas data frame
    columns = ['scores', 'atlas', 'measure', 'classifier']
    data = _categorize_data(data, columns=columns)

    data['dataset'] = pd.Series([data_name] * len(data), index=data.index)

    data_list.append(data)


data = pd.concat(data_list)
