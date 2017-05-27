"""Report prediction scores of suggested pipeline methods for atlas, measure,
   classifier.
"""
import os
import pandas as pd

from collections import OrderedDict


#############################################################################
# Grab csv data paths for all datasets: structure of the data paths organization
# is as follows: covariance_estimator, datasets, atlases, whether networks or
# region_extraction and at the end we have csv files with scores for each atlas
# For example, scores_ica.csv for ICA.

covariance_estimator = 'LedoitWolf'

dataset_names = ['COBRE', 'ADNI', 'ADNIDOD', 'ACPI', 'ABIDE']
atlases = ['ICA', 'DictLearn', 'KMeans', 'BASC', 'Ward',
           'AAL', 'HarvardOxford']

extensions = {'ICA': 'scores_ica.csv',
              'DictLearn': 'scores_dictlearn.csv',
              'KMeans': 'scores_kmeans.csv',
              'BASC': 'scores_basc.csv',
              'Ward': 'scores_ward.csv',
              'AAL': 'scores_aal.csv',
              'HarvardOxford': 'scores_harvardoxford.csv'}

# set csv files data folder
base_path = os.path.join('../../prediction_scores', covariance_estimator)

# Loop over all datasets and atlases to grab csv paths and store them in
# dictionary for each dataset
dataset_paths = dict()

for dataset in dataset_names:
    paths = []
    for atlas in atlases:
        if atlas == 'AAL':
            atlas_path = os.path.join(atlas, extensions[atlas])
            path = os.path.join(base_path, dataset, atlas_path)
            if os.path.exists(path):
                paths.append(path)
        else:
            for folder_name in ['parcellations', 'networks', 'region_extraction',
                                'symmetric_split']:
                atlas_path = os.path.join(atlas, folder_name, extensions[atlas])
                path = os.path.join(base_path, dataset, atlas_path)
                if os.path.exists(path):
                    paths.append(path)
    dataset_paths[dataset] = paths
##############################################################################
# What we have upto here are dataset paths over all atlases for each dataset.
# Now, we load them using pandas. For that we import load_data which is module
# having set of functions to them based on list of paths or single path.

import load_data

data_list = []
for dataset in dataset_names:
    this_data_paths = dataset_paths[dataset]
    this_data = load_data._pandas_data_frame_list_of_paths_concat(this_data_paths)
    data_list.append(this_data)

data_all = pd.concat(data_list)
data_all = data_all.drop('Unnamed: 0', axis=1)
# So, we have now a data frame which contains the data of all the datasets. For
# each dataset over all the atlases.

# Choosing the optimal choice in dimensionality for each atlas
dic_model_dim = OrderedDict([('ica', 80),
                            ('kmeans', 100),
                            ('dictlearn', 60),
                            ('ward', 120),
                            ('aal_spm12', 116),
                            ('basc', 122),
                            ('harvard_oxford', 96)])

# Choosing the optimal choice in region extraction for each atlas after
# choosing to best dimensionality for each atlas
best_pref = OrderedDict([('ica', 'yes'),
                         ('kmeans', 'no'),
                         ('dictlearn', 'yes'),
                         ('ward', 'no'),
                         ('aal_spm12', 'no'),
                         ('basc', 'yes'),
                         ('harvard_oxford', 'no')])
# Build another data frame for optimal selections to print prediction scores for
# best pipeline methods
df = pd.DataFrame()
for key in dic_model_dim.keys():
    df = df.append(data_all[(data_all['atlas'] == key) &
                            (data_all['dimensionality'] == dic_model_dim[key]) &
                            (data_all['region_extraction'] == best_pref[key])])
###########################################################################
# Get prediction scores for good recommendations of pipeline methods
table_scores = pd.DataFrame()
rename = {'dataset': 'Dataset', 'atlas': 'Atlas', 'measure': 'Measure',
          'classifier': 'Classifier', 'dimensionality': 'Dimensionality',
          'region_extraction': 'Regions'}

# columns = ['Dataset', 'Atlas', 'Measure', 'Classifier',
#           'Dimensionality', 'Regions', 'ROC_AUC']

# Get new formal names for atlas, classifier, measures
from aliases import new_names_atlas, new_names_classifier, new_names_measure

# Filter one-by-one to grab
for dataset in df['dataset'].unique():
    data_dataset = df[(df['dataset'] == dataset)]
    # Mean scores and standard deviation using groupby
    mean_std_scores = data_dataset.groupby(
        ['atlas', 'classifier', 'measure', 'dimensionality',
         'region_extraction']
    ).scores.agg({'scores_std': 'std', 'scores_mean': 'mean'})
    mean_std_scores = mean_std_scores.reset_index()
    # Get the maximum scores grouping by atlas
    max_scores = mean_std_scores[mean_std_scores.scores_mean == mean_std_scores.groupby('atlas').scores_mean.transform(max)]
    # Pick one pipeline combination for results
    max_score = max_scores.scores_mean.max()
    pick_score_data = max_scores[max_scores['scores_mean'] == max_score]
    pick_score_data = pick_score_data.rename(index=str, columns=rename)
    # Atlas
    pick_score_data.loc[pick_score_data.index, 'Atlas'] = \
        new_names_atlas()[pick_score_data.Atlas.values[0]]
    # Classifier
    pick_score_data.loc[pick_score_data.index, 'Classifier'] = \
        new_names_classifier()[pick_score_data.Classifier.values[0]]
    # Measure
    pick_score_data.loc[pick_score_data.index, 'Measure'] = \
        new_names_measure()[pick_score_data.Measure.values[0]]
    pick_score_data['ROC_AUC'] = (pick_score_data['scores_mean'].round(2).map(str)
                                  + '$\pm$' + pick_score_data['scores_std'].round(2).map(str))
    del pick_score_data['scores_mean']
    del pick_score_data['scores_std']
    pick_score_data['Dataset'] = dataset
    table_scores = table_scores.append(pick_score_data)
###########################################################################
# csv
table_scores.to_csv('roc_auc_scores.csv')
table_scores.to_latex('roc_auc_scores.tex')
