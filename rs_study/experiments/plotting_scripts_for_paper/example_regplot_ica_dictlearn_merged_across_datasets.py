"""Example to demonstrate linear relationship through regression
using seaborn regplot between parameters 'dimensionality' as x variable
and 'scores' as y variable in regplot. Looking at the consistency in the
linear relationship across datasets.
"""
import os
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

# Gather data

import load_data
from my_palette import atlas_palette, color_palette

# Covariance estimator used in connectomes
covariance_estimator = 'LedoitWolf'

dataset_paths = dict()
dataset_names = ['COBRE', 'ADNI', 'ADNIDOD', 'ACPI']
atlases = ['ICA', 'DictLearn', 'KMeans', 'BASC']

extensions = {'ICA': 'scores_ica.csv',
              'DictLearn': 'scores_dictlearn.csv',
              'KMeans': 'scores_kmeans.csv',
              'BASC': 'scores_basc.csv'
              }

base_path = os.path.join('../prediction_scores', covariance_estimator)

for dataset in dataset_names:
    paths = []
    for atlas in atlases:
        atlas_path = os.path.join(atlas, 'region_extraction',
                                  extensions[atlas])
        path = os.path.join(base_path, dataset, atlas_path)
        if os.path.exists(path):
            paths.append(path)
    dataset_paths[dataset] = paths

data_list = []
for name in dataset_names:
    this_data_paths = dataset_paths[name]
    this_data = load_data._pandas_data_frame_list_of_paths_concat(this_data_paths)
    # Add dataset name to this_data
    data_list.append(this_data)

data = pd.concat(data_list)
data = data.drop('Unnamed: 0', axis=1)

sns.set(color_codes=True)
sns.set_style("whitegrid", {'axes.edgecolor': '.6', 'grid.color': '.6'})
sns.set_palette('dark')

scatter_kws = {'s': 3,
               'alpha': 0.3}
line_kws = {'lw': 2}

ncols = len(dataset_names)
fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(8, 4), squeeze=True,
                         sharey=True)
axes = axes.reshape(-1)

NAMES = {'ica': 'ICA',
         'dictlearn': 'DictLearn',
         'kmeans': 'KMeans',
         'basc': 'BASC'}
palette = color_palette(4)

for i, (name, ax) in enumerate(zip(dataset_names, axes)):
    for label, pal in zip(['ica', 'dictlearn', 'kmeans', 'basc'], palette):
        this_data = data[(data['dataset'] == name) &
                         (data['atlas'] == label) &
                         (data['classifier'] == 'svc_l2') &
                         (data['measure'] == 'tangent')]
        sns.regplot(x='n_regions', y='scores', data=this_data,
                    lowess=True, ax=ax, label=NAMES[label],
                    scatter_kws=scatter_kws,
                    line_kws=line_kws,
                    color=atlas_palette[label],
                    )
        if i == 0:
            ax.set_ylabel('Prediction scores', size=15)
        else:
            ax.set_ylabel('')

        if i == 3:
            ax.legend(scatterpoints=1, frameon=True, fontsize=12,
                      markerscale=5,
                      borderaxespad=0, handletextpad=.2, loc='upper right')
        ax.set_xlabel('')
        plt.text(.5, 1.02, name, transform=ax.transAxes, size=15, ha='center')

        ax.axis('tight')
        ax.set_ylim(.5, 1)

plt.text(.6, 0.03, 'Number of regions extracted', transform=fig.transFigure,
         size=15, ha='center')
plt.tight_layout(rect=[0, .1, 1, .96], pad=.1, w_pad=1)
plt.savefig('n_regions_vs_scores_merged.pdf')
plt.close()
