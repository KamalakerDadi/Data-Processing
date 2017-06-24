
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from collections import OrderedDict
import numpy as np

from my_palette import (color_palette, atlas_palette,
                        datasets_palette)


def stripplot_mean_score(df, save_path, dic_model_dim, suffix=None, x=None,
                         y=None, hue=None, style='whitegrid', fontsize=14,
                         jitter=.2, figsize=(9, 3), leg_pos=2,
                         leg_borderpad=None, leg_ncol=1, leg_bbox_to_anchor=None):
    marker_dict = {'COBRE': 'o',
                   'ADNI': 's',
                   'ADNIDOD': '>',
                   'ACPI': '^',
                   'ABIDE': 'v'}

    # make-up the data
    def change_label_name(row, label):
        row[label] = new_names[row[label]]
        return row

    # atlas labels
    if y == 'atlas':
        ylabel = "Atlas"
        #new_names = OrderedDict([('ica', 'ICA \n (' + str(dic_model_dim['ica'])
        #                          + ' regions)'),
        #                         ('kmeans', 'K-Means \n (' + str(dic_model_dim['kmeans'])
        #                          + ' regions)'),
        #                         ('dictlearn', 'Dictionary \n Learning \n '
        #                                       '(40 regions)'),
        #                         ('ward', 'Ward \n (120 regions)'),
        #                         ('aal_spm12', 'AAL \n (116 regions)'),
        #                         ('basc', 'BASC \n (122 regions)'),
        #                         ('harvard_oxford', 'HO \n (96 regions)')])

        new_names = OrderedDict([('ica', 'Group ICA'),
                                 ('kmeans', 'K-Means'),
                                 ('dictlearn', 'Online \n Dictionary \n Learning'),
                                 ('ward', 'Ward'),
                                 ('aal_spm12', 'AAL'),
                                 ('basc', 'BASC'),
                                 ('harvard_oxford', 'Harvard \n Oxford')])
        df = df.apply(lambda x: change_label_name(x, y), axis=1)

    # change the name of the dataset to upper
    df['dataset'] = df['dataset'].str.upper()

    # make labels of the y axes shorter
    # df[y] = df[y].str.wrap(13)

    rc('xtick', labelsize=12)
    rc('ytick', labelsize=12)
    rc('axes', labelweight='bold')  # string.capitalize
    rc('legend', fontsize=fontsize)

    # sns.set(color_codes=True)
    # sns.set_style("whitegrid", {'axes.edgecolor': '.6', 'grid.color': '.6'})
    # sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    # sns.set_palette('dark')

    n_datasets = len(df['dataset'].unique())
    palette = color_palette(n_datasets)

    fig, axx = plt.subplots(nrows=1, ncols=1, figsize=figsize, sharey=False)

    # draw a default vline at x=0 that spans the yrange
    axx.axvline(x=0, linewidth=4, zorder=0, color='0.6')

    plot = sns.boxplot(data=df, x=x, y=y, fliersize=0, linewidth=2,
                       boxprops={'facecolor': '0.8', 'edgecolor': '.0'},
                       width=0.8, ax=axx)
    width_of_boxplot = 0.8
    offset = width_of_boxplot / n_datasets
    n_offsets = [-offset * 2, -offset * 1, 0, offset * 1, offset * 2]

    for i, (off, dataset) in enumerate(zip(n_offsets,
                                           ['COBRE', 'ADNI', 'ADNIDOD', 'ACPI', 'ABIDE'])):
        # Plot dataset by dataset
        data = df[(df['dataset'] == dataset)]
        if y == 'atlas':
            # if dataset == 'ABIDE':
            #    atlases = df['atlas'].unique()
            # else:
            atlases = data['atlas'].unique()
            for i, atlas in enumerate(atlases):
                # if dataset == 'ABIDE':
                #    if atlas in ['AAL', 'Harvard Oxford', 'BASC',
                #                 'Online Dictionary \n Learning']:
                #        continue
                this_atlas = data[(data['atlas'] == atlas)]
                rank_a = len(this_atlas) * [i]
                y_a = np.add(rank_a, len(this_atlas) * [off])
                axx.scatter(x=this_atlas[x], y=y_a, data=this_atlas,
                            color=datasets_palette[dataset],
                            marker=marker_dict[dataset])
        elif y == 'measure':
            for ii, measure in enumerate(data['measure'].unique()):
                this_measure = data[data['measure'] == measure]
                rank_m = len(this_measure) * [ii]
                y_m = np.add(rank_m, len(this_measure) * [off])
                axx.scatter(x=this_measure[x], y=y_m, data=this_measure,
                            color=datasets_palette[dataset],
                            marker=marker_dict[dataset])
        elif y == 'classifier':
            for iii, classifier in enumerate(data['classifier'].unique()):
                this_classifier = data[data['classifier'] == classifier]
                rank_c = len(this_classifier) * [iii]
                y_c = np.add(rank_c, len(this_classifier) * [off])
                axx.scatter(x=this_classifier[x], y=y_c, data=this_classifier,
                            color=datasets_palette[dataset],
                            marker=marker_dict[dataset])

    # sns.stripplot(data=data, x=x, y=y, hue=hue, edgecolor='gray',
    #              size=8, split=True, palette=datasets_palette,
    #              jitter=jitter, marker=marker_dict[dataset], ax=axx)

    # axx.set_ylabel(ylabel, fontsize=14)
    axx.set_ylabel('')
    # plt.text(.5, 1.02, ylabel, transform=axx.transAxes, size=15, ha='center')
    # xlabel = 'Relative in prediction scores'
    # plt.text(.01, 3., xlabel, size=15, ha='center')
    # axx.set_xlabel('')
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

    n_atlas = len(df['atlas'].unique())

    # background
    for a in range(1, n_atlas):
        if a % 2:
            axx.axhspan(a - .5, a + .5, color='0.9', zorder=-1)
    # axx.set_xlim(-0.14, 0.11)

    cobre_marker1 = plt.Line2D([], [], color=datasets_palette['COBRE'],
                               marker=marker_dict['COBRE'], linestyle='')
    adni_marker2 = plt.Line2D([], [], color=datasets_palette['ADNI'],
                              marker=marker_dict['ADNI'], linestyle='')
    adnidod_marker3 = plt.Line2D([], [], color=datasets_palette['ADNIDOD'],
                                 marker=marker_dict['ADNIDOD'], linestyle='')
    acpi_marker4 = plt.Line2D([], [], color=datasets_palette['ACPI'],
                              marker=marker_dict['ACPI'], linestyle='')
    abide_marker5 = plt.Line2D([], [], color=datasets_palette['ABIDE'],
                               marker=marker_dict['ABIDE'], linestyle='')

    axx.legend([cobre_marker1, adni_marker2, adnidod_marker3, acpi_marker4, abide_marker5],
               ['COBRE', 'ADNI', 'ADNIDOD', 'ACPI', 'ABIDE'],
               loc=leg_pos, handletextpad=0.0, borderaxespad=0,
               fontsize=12, frameon=True, scatterpoints=1,
               markerscale=True, borderpad=leg_borderpad,
               ncol=leg_ncol, columnspacing=0.1,
               bbox_to_anchor=leg_bbox_to_anchor)

    # dont allow the labels go outside the figure area.
    plt.tight_layout(rect=[0, .09, 1, .92], pad=0.1, w_pad=1)
    if suffix is not None:
        save_name_pdf = 'impact_plot_' + str(x) + '_' + suffix + '.pdf'
        save_name_png = 'impact_plot_' + str(x) + '_' + suffix + '.png'
        save_name_svg = 'impact_plot_' + str(x) + '_' + suffix + '.svg'
    else:
        save_name_pdf = 'impact_plot_' + str(y) + '.pdf'
        save_name_png = 'impact_plot_' + str(y) + '.png'
        save_name_svg = 'impact_plot_' + str(y) + '.svg'

    if y == 'atlas':
        # ylabel = 'Choice of classifier'
        axx.set_xlabel('Relative in prediction scores',
                       fontsize=13, fontweight='normal')
        # plt.text(.5, 1.05, ylabel, transform=axx.transAxes, size=15, ha='center')
        plt.tight_layout(rect=[0.1, .1, 1, 1.], pad=0.3, w_pad=1)
        axx.legend([cobre_marker1, adni_marker2, adnidod_marker3, acpi_marker4, abide_marker5],
                   ['COBRE', 'ADNI', 'ADNIDOD', 'ACPI', 'ABIDE'],
                   loc=leg_pos, handletextpad=-0.5, borderaxespad=0,
                   fontsize=12, frameon=True, scatterpoints=1,
                   markerscale=True, borderpad=leg_borderpad,
                   ncol=leg_ncol, columnspacing=-0.2,
                   bbox_to_anchor=leg_bbox_to_anchor)
    else:
        plt.text(.5, 1.02, ylabel, transform=axx.transAxes, size=15, ha='center')

    # plt.savefig(os.path.join(save_path, save_name_pdf))
    # plt.savefig(os.path.join(save_path, save_name_png))
    # plt.savefig(os.path.join(save_path, save_name_svg))
    # print('The figure is saved %s' % os.path.join(save_path, save_name_pdf))

###############################################################################
# Gather data

import load_data

covariance_estimator = 'LedoitWolf'

dataset_paths = dict()
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

base_path = os.path.join('../prediction_scores', covariance_estimator)

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

data_list = []
for dataset in dataset_names:
    this_data_paths = dataset_paths[dataset]
    this_data = load_data._pandas_data_frame_list_of_paths_concat(this_data_paths)
    data_list.append(this_data)

data_all = pd.concat(data_list)
data_all = data_all.drop('Unnamed: 0', axis=1)

save_path = '../paper/figures/'
save_sufix = None
# extract the data from the csv that containes all the results

dic_model_dim = OrderedDict([('ica', 80),
                            ('kmeans', 120),
                            ('dictlearn', 60),
                            ('ward', 120),
                            ('aal_spm12', 116),
                            ('basc', 122),
                            ('harvard_oxford', 96)])

best_pref = OrderedDict([('ica', 'yes'),
                         ('kmeans', 'no'),
                         ('dictlearn', 'yes'),
                         ('ward', 'no'),
                         ('aal_spm12', 'no'),
                         ('basc', 'yes'),
                         ('harvard_oxford', 'no')])
df = pd.DataFrame()
for key in dic_model_dim.keys():
    df = df.append(data_all[(data_all['atlas'] == key) &
                            (data_all['dimensionality'] == dic_model_dim[key]) &
                            (data_all['region_extraction'] == best_pref[key])])
###############################################################################
# calculate mean data


def demean(group):
    return group - group.mean()

# Take the average over iter_shuffle_split
df = df.groupby(['classifier', 'measure', 'atlas', 'dataset']).mean()
df = df.reset_index()
df.pop('iter_shuffle_split')
demeaned_scores_atlas = df.groupby(['classifier', 'measure',
                                    'dataset'])['scores'].transform(demean)
demeaned_scores_measure = df.groupby(['classifier', 'atlas',
                                      'dataset'])['scores'].transform(demean)
demeaned_scores_classifier = df.groupby(['atlas', 'measure', 'dataset'])['scores'].transform(demean)

df['demeaned_scores_atlas'] = demeaned_scores_atlas
df['demeaned_scores_measure'] = demeaned_scores_measure
df['demeaned_scores_classifier'] = demeaned_scores_classifier
dic_atlas_n = {'aal_spm12': 1,
               'harvard_oxford': 2,
               'basc': 3,
               'ward': 4,
               'kmeans': 5,
               'ica': 6,
               'dictlearn': 7}

###############################################################################
# plot to pdf, png and svg
hue = 'dataset'

sns.set(color_codes=True)
sns.set_style("whitegrid", {'axes.edgecolor': '.6', 'grid.color': '.8'})
# sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sns.set_palette('dark')
# atlas
x = 'demeaned_scores_atlas'
y = 'atlas'
df['rank'] = df['atlas'].map(dic_atlas_n)
df.sort_values(by=['rank'], inplace=True)
stripplot_mean_score(df, save_path, dic_model_dim, suffix=save_sufix, x=x,
                     y=y, hue=hue, style='whitegrid', fontsize=12, jitter=0,
                     figsize=(5.5, 5), leg_pos='lower center', leg_borderpad=0.1,
                     leg_ncol=5, leg_bbox_to_anchor=(0.1, -0.2))

bbox = dict(boxstyle="round", fc="w", ec="0.9", alpha=0.5)
arrowprops = dict(arrowstyle='-[, widthB=2.8', lw=1.)
plt.annotate('Structural \n atlases', xy=(-0.22, 0.84),
             xytext=(-0.32, 0.76), xycoords='axes fraction',
             fontsize=10, ha='center', va='bottom', bbox=bbox,
             arrowprops=arrowprops, rotation=90)

arrowprops = dict(arrowstyle='-', lw=1.)
plt.annotate('Functional \n atlas', xy=(-0.15, 0.64),
             xytext=(-0.32, 0.555), xycoords='axes fraction',
             fontsize=10, ha='center', va='bottom', bbox=bbox,
             arrowprops=arrowprops, rotation=90)

arrowprops = dict(arrowstyle='-[, widthB=2.55', lw=1.)
plt.annotate('Clustering \n methods', xy=(-0.22, 0.44),
             xytext=(-0.32, 0.355), xycoords='axes fraction',
             fontsize=10, ha='center', va='bottom', bbox=bbox,
             arrowprops=arrowprops, rotation=90)
arrowprops = dict(arrowstyle='-[, widthB=2.9', lw=1.)
plt.annotate('Linear Decomposition \n methods', xy=(-0.24, 0.14),
             xytext=(-0.32, -0.0355), xycoords='axes fraction',
             fontsize=10, ha='center', va='bottom', bbox=bbox,
             arrowprops=arrowprops, rotation=90)
plt.text( -0.12, 6.21, '(*)', size=20, color='red')

plt.savefig(os.path.join(save_path, 'impact_plot_atlas.pdf'))
plt.savefig(os.path.join(save_path, 'impact_plot_atlas.svg'))
plt.savefig(os.path.join(save_path, 'impact_plot_atlas.png'))
