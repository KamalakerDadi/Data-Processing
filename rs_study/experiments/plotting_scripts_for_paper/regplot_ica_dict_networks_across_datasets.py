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
from my_palette import atlas_palette

# Covariance estimator used in connectomes
covariance_estimator = 'LedoitWolf'

dataset_paths = dict()
dataset_names = ['COBRE', 'ADNI']

base_path = os.path.join('../prediction_scores', covariance_estimator)

for dataset in dataset_names:
    path = os.path.join(base_path, dataset)
    path_ica = path + '/ICA/networks/scores_ica.csv'
    path_dictlearn = path + '/DictLearn/networks/scores_dictlearn.csv'
    paths_appended = [path_ica, path_dictlearn]
    dataset_paths[dataset] = paths_appended

data_list = []
for name in dataset_names:
    this_data_paths = dataset_paths[name]
    this_data = load_data._pandas_data_frame_list_of_paths_concat(this_data_paths)
    # Add dataset name to this_data
    data_list.append(this_data)

data = pd.concat(data_list)
data = data.drop('Unnamed: 0', axis=1)

# Average over all the folds:
# columns = ['atlas', 'dataset', 'measure', 'classifier', 'dimensionality']
# data = data.groupby(columns).mean().reset_index()

sns.set(color_codes=True)
sns.set_style("whitegrid", {'axes.edgecolor': '.6', 'grid.color': '.6'})
sns.set_palette('dark')

scatter_kws = {'s': 5}
line_kws = {'lw': 2}
ncols = len(dataset_names)
fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(4, 3.5), squeeze=True,
                         sharey=True)
axes = axes.reshape(-1)

NAMES = {'ica': 'ICA',
         'dictlearn': 'DictLearn'}

for i, (name, ax) in enumerate(zip(dataset_names, axes)):
    for label in ['ica', 'dictlearn']:
        this_data = data[(data['dataset'] == name) &
                         (data['atlas'] == label) &
                         (data['classifier'] == 'svc_l2') &
                         (data['measure'] == 'tangent')]
        sns.regplot(x='dimensionality', y='scores', data=this_data,
                    lowess=True, ax=ax, label=NAMES[label],
                    scatter_kws=scatter_kws,
                    line_kws=line_kws,
                    color=atlas_palette[label],
                    )
        if i == 0:
            ax.set_ylabel('Prediction scores', size=15)
            ax.legend(scatterpoints=1, frameon=True, fontsize=12, markerscale=3,
                      borderaxespad=0, handletextpad=.2, loc='lower right')
        else:
            ax.set_ylabel('')
        ax.set_xlabel('')
        plt.text(.5, 1.02, name, transform=ax.transAxes, size=15, ha='center')

        ax.axis('tight')
        ax.set_ylim(.4, 1)

plt.text(.6, 0.03, 'Number of components', transform=fig.transFigure,
         size=15, ha='center')
plt.tight_layout(rect=[0, .1, 1, .96], pad=.1, w_pad=1)
plt.savefig('networks_vs_scores_merged.pdf')
plt.close()
