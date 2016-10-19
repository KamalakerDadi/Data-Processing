"""
"""
import glob

import pandas as pd

path_to_csvs = glob.glob('results_csv/*.csv')
labels = []

for csv in path_to_csvs:
    label = csv.split('/')[1].split('_')[1].split('.csv')[0]
    print("==== Printing mean scores - %s dataset ====" % label)
    labels.append(label)
    data = pd.read_csv(csv)

    ##########################################################################
    # Data preparation
    scores = data['scores'].str.strip('[ ]')
    del data['scores']
    data = data.join(scores)
    data.scores = data.scores.astype(float)

    data = data.groupby(['atlas', 'classifier', 'measure']).scores.agg(
        {'score_std': 'std', 'score_mean': 'mean'})
    # Print best result for each atlas
    data = data.reset_index()
    print(data[data.score_mean ==
               data.groupby('atlas').score_mean.transform(max)])
