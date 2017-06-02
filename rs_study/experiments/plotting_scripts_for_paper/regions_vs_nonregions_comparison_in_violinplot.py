import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from collections import OrderedDict

import load_data
from my_palette import (color_palette, atlas_palette,
                        datasets_palette)


def stripplot_mean_score(df, save_path, atlas=None, suffix=None, x=None,
                         y=None, hue=None, style='whitegrid', fontsize=14,
                         jitter=.2, figsize=(9, 3), leg_pos=2, axx=None):

    def change_label_name(row, label):
        row[label] = new_names[row[label]]
        return row

    ylabel = atlas
    aliases = {'kmeans': 'K-Means',
               'ica': 'GroupICA',
               'dictlearn': 'Dictionary Learning',
               'basc': 'BASC'}
    if atlas == 'kmeans':
        new_names = {'no': 'Without\n regions extracted',
                     'yes': 'With\n regions extracted'}
        df = df.apply(lambda x: change_label_name(x, y), axis=1)
    else:
        new_names = {'no': 'Without\n regions extracted',
                     'yes': 'With\n regions extracted'}
        df = df.apply(lambda x: change_label_name(x, y), axis=1)

    # change the name of the dataset to upper
    df['dataset'] = df['dataset'].str.upper()

    # make labels of the y axes shorter
    # df[y] = df[y].str.wrap(13)

    rc('xtick', labelsize=12)
    rc('ytick', labelsize=16)
    rc('axes', labelweight='bold')  # string.capitalize
    rc('legend', fontsize=fontsize)

    n_data = len(df['dataset'].unique())
    palette = color_palette(n_data)

    # draw a default vline at x=0 that spans the yrange
    axx.axvline(x=0, linewidth=4, zorder=0, color='0.6')

    sns.violinplot(data=df, x=x, y=y, fliersize=0, linewidth=2,
                   boxprops={'facecolor': '0.5', 'edgecolor': '.0'},
                   width=0.5, ax=axx)

    sns.stripplot(data=df, x=x, y=y, hue=hue, edgecolor='gray',
                  size=3, split=True, palette=datasets_palette, jitter=jitter,
                  ax=axx)

    axx.set_xlabel('')
    # axx.set_ylabel(aliases[ylabel], fontsize=15)
    axx.set_ylabel('')
    plt.text(.5, 1.02, aliases[key], transform=ax.transAxes, size=15, ha='center')

    # make the positive labels with "+"
    axx_xticklabels = []
    for x in axx.get_xticks():
        if x > 0:
            axx_xticklabels.append('+' + str(x))
        else:
            axx_xticklabels.append(str(x))
    axx.set_xticklabels(axx_xticklabels)
    # xticklabels=string.capitalize()
    # yticklabels=string.capitalize(axx.get_yticklabels())
    # axx.set_yticklabels(yticklabels)

    # n_atlas = len(df[y].unique())

    # background
    # for a in range(1, n_atlas):
    #    if a % 2:
    #        axx.axhspan(a - .5, a + .5, color='1', zorder=-1)

###############################################################################
# csv results

covariance_estimator = 'LedoitWolf'
dataset_paths = dict()
dataset_names = ['COBRE', 'ADNI', 'ADNIDOD', 'ACPI']
atlases = ['ICA', 'DictLearn', 'KMeans', 'BASC']

extensions = {'ICA': 'scores_ica.csv',
              'DictLearn': 'scores_dictlearn.csv',
              'KMeans': 'scores_kmeans.csv',
              'BASC': 'scores_basc.csv'}

base_path = os.path.join('../prediction_scores', covariance_estimator)

for dataset in dataset_names:
    paths = []
    for atlas in atlases:
        for folder in ['region_extraction', 'networks', 'parcellations']:
            atlas_path = os.path.join(atlas, folder, extensions[atlas])
            path = os.path.join(base_path, dataset, atlas_path)
            if os.path.exists(path):
                paths.append(path)
        dataset_paths[dataset] = paths
###############################################################################
# load and extract the data

data_list = []
for name in dataset_names:
    this_data_path = dataset_paths[name]
    this_data = load_data._pandas_data_frame_list_of_paths_concat(this_data_path)
    data_list.append(this_data)

data = pd.concat(data_list)
data = data.drop('Unnamed: 0', axis=1)

dic_model_dim = OrderedDict([('ica', 80),
                             ('kmeans', 120),
                             ('dictlearn', 60),
                             ('basc', 122)
                             ])

data2 = pd.DataFrame()
for key in dic_model_dim.keys():
    data2 = data2.append(data[(data['atlas'] == key) &
                              (data['dimensionality'] == dic_model_dim[key])])

atlases = ['ica', 'dictlearn', 'kmeans', 'basc']

fig, axes = plt.subplots(nrows=1, ncols=len(atlases), figsize=(10, 3),
                         squeeze=True, sharey=True)
axes = axes.reshape(-1)

for i, (key, ax) in enumerate(zip(atlases, axes)):

    this_data = data2[(data2['atlas'] == key)]
    # this_data = this_data.drop('Unnamed: 0.1', axis=1)

    ########################################################################
    # calculate mean data

    def demean(group):
        return group - group.mean()

    # Take the average over iter_shuffle_split
    df = this_data.groupby(['atlas', 'classifier', 'measure', 'dataset',
                            'region_extraction']).mean()
    df = df.reset_index()
    df.pop('smoothing_fwhm')
    df.pop('iter_shuffle_split')

    demeaned_scores = df.groupby(['classifier', 'measure', 'dataset',
                                  'dimensionality'])['scores'].transform(demean)

    df['demeaned_scores'] = demeaned_scores

    ########################################################################
    # plot to pdf, png and svg
    hue = 'dataset'

    # atlas
    x = 'demeaned_scores'
    y = 'region_extraction'
    stripplot_mean_score(df, key, x=x, atlas=key,
                         y=y, hue=hue, style='whitegrid', fontsize=12, jitter=0,
                         figsize=(5, 6), leg_pos='upper left', axx=ax)
    if i == 0:
        ax.legend(scatterpoints=1, frameon=True, fontsize=15, markerscale=1,
                  handlelength=0.8, borderpad=.2, borderaxespad=3,
                  handletextpad=.05, loc='lower left', ncol=4,
                  columnspacing=0.2, bbox_to_anchor=(-1.54, -0.55))
    else:
        ax.legend().remove()

    for x in (0, 1):
        if x == 0:
            ax.axhspan(2., 2, color='1', zorder=-1)
        else:
            ax.axhspan(x - .5, x + .5, color='0.75', zorder=-1)


plt.text(.6, .025, 'Impact on prediction accuracy', transform=fig.transFigure,
         size=15, ha='center')

plt.tight_layout(rect=[0, 0.05, 1, 1])

save_name_pdf = 'violin_plot_mean_score_region_extraction_ica_dict_kmeans.pdf'
plt.savefig(save_name_pdf)
