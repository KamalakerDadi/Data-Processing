"""
"""
import os
import seaborn as sns
import pandas as pd
import numpy as np

from scipy import stats
from matplotlib import pyplot as plt

# Gather data

import load_data
from my_palette import (atlas_palette, color_palette,
                        datasets_palette)

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
        if (dataset == 'ADNIDOD') and (atlas == 'ICA'):
            extension = 'scores_ica_120.csv'
        else:
            extension = extensions[atlas]
        atlas_path = os.path.join(atlas, 'region_extraction',
                                  extension)
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
this_data = data[(data['classifier'] == 'svc_l2') &
                 (data['measure'] == 'tangent')]

sns.set(color_codes=True)
sns.set_style("whitegrid", {'axes.edgecolor': '.6', 'grid.color': '.6'})
sns.set_palette('dark')


def demean(group):
    return group - group.mean()

this_data.pop('Unnamed: 0.1')
this_data.pop('smoothing_fwhm')
this_data.pop('reduction_n_components')
this_data.pop('min_region_size_in_mm3')
this_data.pop('atlas_type')
this_data.pop('covariance_estimator')
this_data.pop('connectome_regress')
this_data.pop('compcor_10')
this_data.pop('version')
this_data.pop('multi_pca_reduction')
this_data.pop('motion_regress')


plt.close('all')
atlas_names = {'ica': 'ICA',
               'dictlearn': 'DictLearn',
               'kmeans': 'KMeans',
               'basc': 'BASC'}

wilcoxon_tests = dict()
columns = ['atlas', 'dataset', 'pvalues', '-log10pvalues',
           'dimensionality_fixed', 'dimensionality_varied']

for column_name in columns:
    wilcoxon_tests.setdefault(column_name, [])

comparison_fixed_options = {'ica': 80,
                            'dictlearn': 80,
                            'kmeans': 120,
                            'basc': 122}

atlases = this_data['atlas'].unique()

for atlas in atlases:
    data2 = this_data[this_data['atlas'] == atlas]
    for dataset in data2['dataset'].unique():
        d_dataset = data2[(data2['dataset'] == dataset)]
        fixed_dim = comparison_fixed_options[atlas]
        for dim in d_dataset['dimensionality'].unique():
            if dim != fixed_dim:
                wilcoxon_tests['atlas'].append(atlas)
                wilcoxon_tests['dataset'].append(dataset)
                wilcoxon_tests['dimensionality_fixed'].append(fixed_dim)
                wilcoxon_tests['dimensionality_varied'].append(dim)
                data_fixed = d_dataset[(d_dataset['dimensionality'] == fixed_dim)]
                data_vary = d_dataset[(d_dataset['dimensionality'] == dim)]
                _, p = stats.wilcoxon(data_fixed['scores'], data_vary['scores'],
                                      correction=True)
                wilcoxon_tests['pvalues'].append(p)
                wilcoxon_tests['-log10pvalues'].append(-np.log10(p))

data_pvalues = pd.DataFrame(wilcoxon_tests)

ncols = len(this_data['atlas'].unique())
fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(10, 7), sharey=False,
                         squeeze=True, sharex=False)
# axes = axes.reshape(-1)
palette = sns.color_palette(n_colors=len(this_data['dataset'].unique()))

for i, ax in enumerate(axes):
    if i == 0:
        axx = ax.reshape(-1)
        for ii, (ax1, atlas) in enumerate(zip(axx, atlases)):
            each_atlas_pvalues = data_pvalues[(data_pvalues['atlas'] == atlas)]
            sns.boxplot(data=each_atlas_pvalues, y='pvalues',
                        x='dimensionality_varied',
                        ax=ax1, color='.8', fliersize=0)
            sns.stripplot(data=each_atlas_pvalues, y='pvalues',
                          x='dimensionality_varied',
                          ax=ax1, hue='dataset', size=4)
            if ii == 0:
                ax1.set_ylabel('P-values (Wilcoxon tests)', size=15)
            else:
                ax1.set_ylabel('')

            #if ii == 0:
            #    ax1.legend(scatterpoints=1, frameon=True, fontsize=15,
            #               markerscale=1, borderaxespad=0,
            #               handletextpad=.2, loc='upper left')
            #else:
            ax1.legend().remove()

            ax1.set_xlabel('')
            name = each_atlas_pvalues['dimensionality_fixed'].unique()[0]
            plt.text(.5, 1, atlas_names[atlas] + ':' + str(name) + 'vs Rest',
                     transform=ax1.transAxes, size=15, ha='center')
            y_ticklabels = ax1.get_yticks()
            for x in (1, 3, 5):
                ax1.axvspan(x - .5, x + .5, color='.9', zorder=-1)
        ax1.set_yticklabels(y_ticklabels)
    else:
        axx2 = ax.reshape(-1)
        for ii, (ax2, atlas) in enumerate(zip(axx2, atlases)):
            each_atlas_neg_pvalues = data_pvalues[(data_pvalues['atlas'] == atlas)]
            sns.boxplot(data=each_atlas_neg_pvalues, y='-log10pvalues',
                        x='dimensionality_varied',
                        ax=ax2, color='.8', fliersize=0)
            sns.stripplot(data=each_atlas_neg_pvalues, y='-log10pvalues',
                          x='dimensionality_varied',
                          ax=ax2, hue='dataset', size=4)
            if ii == 0:
                ax2.set_ylabel('$-\log_{10}(P) $', size=15)
            else:
                ax2.set_ylabel('')

            if ii == 0:
                ax2.legend(scatterpoints=1, frameon=True, fontsize=15,
                           markerscale=1, borderaxespad=0,
                           handletextpad=.2, loc='lower left',
                           ncol=2, bbox_to_anchor=(-0.3, -0.23),
                           columnspacing=0.5)
            else:
                ax2.legend().remove()

            ax2.set_xlabel('')
            ax2.set_xticklabels('')
            y_ticklabels2 = ax2.get_yticks()
            for x2 in (1, 3, 5):
                ax2.axvspan(x2 - .5, x2 + .5, color='.9', zorder=-1)
        ax2.set_yticklabels(y_ticklabels2)

xlabel_name = 'Detailed one-to-one comparison on dimensionality of each atlas'
plt.text(.6, .007, xlabel_name, transform=fig.transFigure,
         size=15, ha='center')
plt.tight_layout(rect=[0, .05, 1, 1])

plt.savefig('one_to_one_dim_comparison.pdf')
