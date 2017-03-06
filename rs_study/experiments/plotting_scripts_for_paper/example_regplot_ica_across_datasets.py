"""Example script to run strip plot comparisons and saving them to pdf
"""
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import load_data
from my_palette import atlas_palette

covariance_estimator = 'LedoitWolf'
dataset_paths = dict()
dataset_names = ['COBRE', 'ADNI', 'ACPI']

base_path = os.path.join('../prediction_scores', covariance_estimator)

for dataset in dataset_names:
    path = os.path.join(base_path, dataset)
    path_ica = path + '/ICA/region_extraction/scores_ica.csv'
    paths = [path_ica]
    dataset_paths[dataset] = paths

data_list = []
for name in dataset_names:
    this_data_path = dataset_paths[name]
    this_data = load_data._pandas_data_frame_list_of_paths_concat(this_data_path)
    data_list.append(this_data)

data = pd.concat(data_list)
data = data.drop('Unnamed: 0', axis=1)

sns.set(color_codes=True)
sns.set_style("whitegrid", {'axes.edgecolor': '.6', 'grid.color': '.6'})
sns.set_palette('dark')

scatter_kws = {'s': 5}
line_kws = {'lw': 2}

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5),
                         squeeze=True, sharey=True)
# axes = axes.reshape(-1)

NAMES = {'ica': 'ICA'}

for i, name in enumerate(dataset_names):
    this_data = data[(data['dataset'] == name) &
                     (data['classifier'] == 'svc_l2') &
                     (data['measure'] == 'tangent')]
    sns.regplot(x='n_regions', y='scores', data=this_data,
                lowess=True, ax=axes, label=name,
                scatter_kws=scatter_kws, line_kws=line_kws,
                # color=atlas_palette['ica']
                )
    axes.legend().remove()
    axes.legend(scatterpoints=1, frameon=True, fontsize=12, markerscale=1,
                borderaxespad=0, handletextpad=.05, loc='lower right')
    if i == 0:
        axes.set_ylabel('Prediction scores', size=15)
        # axes.legend(scatterpoints=1, frameon=True, fontsize=12, markerscale=3,
        #            borderaxespad=0, handletextpad=.05, loc='lower right')
    else:
        axes.set_ylabel('')
    axes.set_xlabel('')
    plt.text(.5, 1.02, name, transform=axes.transAxes, size=13, ha='center')

    axes.axis('tight')
    axes.set_ylim(.5, 1)

plt.text(.6, .025, 'Number of components', transform=fig.transFigure,
         size=15, ha='center')

plt.tight_layout(rect=[0, .05, 1, 1], pad=.1, w_pad=1)

plt.savefig('regplot_ica.pdf')
plt.close()

