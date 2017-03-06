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
dataset_names = ['COBRE', 'ADNI']

base_path = os.path.join('../prediction_scores', covariance_estimator)

for dataset in dataset_names:
    path = os.path.join(base_path, dataset)
    path_ica = path + '/ICA/region_extraction/scores_ica.csv'
    path_dict = path + '/DictLearn/region_extraction/scores_dictlearn.csv'
    path_kmeans = path + '/KMeans/region_extraction/scores_kmeans_120.csv'
    paths = [path_ica, path_dict, path_kmeans]
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


fig, axes = plt.subplots(nrows=1, ncols=len(dataset_names),
                         figsize=(4, 3.5),
                         squeeze=True, sharey=True)
axes = axes.reshape(-1)

NAMES = {'ica': 'ICA',
         'dictlearn': 'DictLearn',
         'kmeans': 'KMeans'}

for i, (name, ax) in enumerate(zip(dataset_names, axes)):
    this_data = data[(data['dataset'] == name) &
                     (data['classifier'] == 'svc_l2') &
                     (data['measure'] == 'tangent')]
    sns.stripplot(x='dimensionality', y='n_regions', data=this_data,
                  ax=ax, jitter=.25, size=1.5,
                  hue='atlas',  # color=atlas_palette[label],
                  palette=atlas_palette, split=True,
                  )
    if i == 1:
        ax.legend(scatterpoints=1, frameon=True, fontsize=12, markerscale=1,
                  handlelength=1, borderpad=.2,
                  borderaxespad=0, handletextpad=.05, loc=(.05, .5))
    else:
        ax.legend().remove()
    if i == 0:
        ax.set_ylabel('Number of regions extracted', size=15)
    else:
        ax.set_ylabel('')
    ax.set_xlabel('')
    plt.text(.5, 1.02, name, transform=ax.transAxes,
             size=13, ha='center')
    for x in (1, 3):
        ax.axvspan(x - .5, x + .5, color='.9', zorder=-1)

plt.text(.6, .025, 'Number of dimensions', transform=fig.transFigure,
         size=15, ha='center')

plt.tight_layout(rect=[0, .05, 1, 1])

plt.savefig('n_regions_vs_dimensionality.pdf')
plt.close()
