"""
"""
import glob

import pandas as pd

path_to_csvs = sorted(glob.glob('../results_csv/results_*.csv'))
labels = []
mean_scores = []

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
    scores = (data[data.score_mean ==
                   data.groupby('atlas').score_mean.transform(max)])
    print(scores)
    scores['dataset'] = pd.Series([label] * len(scores), index=scores.index)

    mean_scores.append(scores)

mean_scores_merged = pd.concat(mean_scores)
