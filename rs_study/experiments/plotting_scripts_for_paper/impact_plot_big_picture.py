__author__ = 'darya'

import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from collections import OrderedDict

import load_data
from my_palette import (color_palette, atlas_palette,
                        datasets_palette)


def stripplot_mean_score(df, save_path, dic_model_dim, suffix=None, x=None,
                         y=None, hue=None, style='whitegrid', fontsize=14,
                         jitter=.2, figsize=(9, 3), leg_pos=2,
                         leg_borderpad=None, leg_ncol=1):
    # make-up the data
    def change_label_name(row, label):
        row[label] = new_names[row[label]]
        return row

    # classifier labels
    if y == 'classifier':
        ylabel = "Classifier"
        new_names = {'ridge': 'Ridge',
                     'svc_l1': r'SVM-$\ell_1$',
                     'svc_l2': r'SVM-$\ell_2$'}
        df = df.apply(lambda x: change_label_name(x, y), axis=1)

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

        new_names = OrderedDict([('ica', 'ICA'),
                                 ('kmeans', 'K-Means'),
                                 ('dictlearn', 'Online Dictionary \n Learning'),
                                 ('ward', 'Ward'),
                                 ('aal_spm12', 'AAL'),
                                 ('basc', 'BASC'),
                                 ('harvard_oxford', 'Harvard Oxford')])
        # new_names = {'dictlearn': 'Dictionary Learning',
        #             'ica': 'ICA',
        #             'kmeans':  'K-means',
        #             'ward':    'Ward',
        #             'ho_cort_symm_split': 'HO',
        #             'aal_spm12': 'AAL',
        #             'basc_scale122': 'Basc'}
        df = df.apply(lambda x: change_label_name(x, y), axis=1)

    # measure labels
    if y == 'measure':
        ylabel = "Measure"
        new_names = {'correlation': 'Correlation',
                     'partial correlation':     'Partial \n Correlation',
                     'tangent':  'Tangent'}
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

    fig, axx = plt.subplots(nrows=1, ncols=1, figsize=figsize, sharey=False)

    # draw a default vline at x=0 that spans the yrange
    axx.axvline(x=0, linewidth=4, zorder=0, color='0.6')

    sns.boxplot(data=df, x=x, y=y, fliersize=0, linewidth=2,
                boxprops={'facecolor': '0.5', 'edgecolor': '.0'},
                width=0.8)

    sns.stripplot(data=df, x=x, y=y, hue=hue, edgecolor='gray',
                  size=5, split=True, palette=datasets_palette, jitter=jitter)

    axx.set_xlabel('Impact of pipeline choices on \n prediction scores',
                   fontsize=15)
    #axx.set_ylabel(ylabel, fontsize=14)
    axx.set_ylabel('')
    #plt.text(.5, 1.02, ylabel, transform=axx.transAxes, size=15, ha='center')

    # make the positive labels with "+"
    axx_xticklabels = []
    for x in axx.get_xticks():
        if x > 0:
            axx_xticklabels.append('+' + str(x) + '$\%$')
        else:
            axx_xticklabels.append(str(x) + '$\%$')
    axx.set_xticklabels(axx_xticklabels)
    # xticklabels=string.capitalize()
    # yticklabels=string.capitalize(axx.get_yticklabels())
    # axx.set_yticklabels(yticklabels)

    n_atlas = len(df[y].unique())

    # background
    for a in range(1, n_atlas):
        if a % 2:
            axx.axhspan(a - .5, a + .5, color='1', zorder=-1)

    plt.legend(loc=leg_pos, handletextpad=0.0, borderaxespad=0,
               fontsize=15, frameon=True, scatterpoints=1,
               markerscale=True, borderpad=leg_borderpad,
               ncol=leg_ncol)

    # dont allow the labels go outside the figure area.
    plt.tight_layout(pad=0.1)
    if suffix is not None:
        save_name_pdf = 'mean_score_' + str(y) + '_' + suffix + '.pdf'
        save_name_png = 'mean_score_' + str(y) + '_' + suffix + '.png'
        save_name_svg = 'mean_score_' + str(y) + '_' + suffix + '.svg'
    else:
        save_name_pdf = 'mean_score_' + str(y) + '.pdf'
        save_name_png = 'mean_score_' + str(y) + '.png'
        save_name_svg = 'mean_score_' + str(y) + '.svg'

    plt.text(.5, 1.02, ylabel, transform=axx.transAxes, size=15, ha='center')

    plt.savefig(os.path.join(save_path, save_name_pdf))
    plt.savefig(os.path.join(save_path, save_name_png))
    plt.savefig(os.path.join(save_path, save_name_svg))
    print('The figure is saved %s' % os.path.join(save_path, save_name_pdf))

###############################################################################
# csv results
# to merge all the csv's from different experiments use merge_csv.py

csv_folder = '../prediction_scores'
csv_name = 'resultes_csv_merged.csv'
save_path = '../plotting_scripts_for_paper'
save_sufix = 'plot_mean_score_model_dimentin'

###############################################################################
# load and extract the data

data_all = pd.read_csv(os.path.join(csv_folder, csv_name), index_col=0)
data_all = data_all[data_all.columns[~data_all.columns.str.contains('Unnamed')]]


# extract the data from the csv that containes all the results

dic_model_dim = OrderedDict([('ica', 80),
                            ('kmeans', 120),
                            ('dictlearn', 80),
                            ('ward', 120),
                            ('aal_spm12', 116),
                            ('basc', 122),
                            ('harvard_oxford', 96)])

df = pd.DataFrame()
for key in dic_model_dim.keys():
    df = df.append(data_all[(data_all['atlas'] == key) &
                            (data_all['dimensionality'] == dic_model_dim[key])])

# clean the data
df.scores = df['scores'].str.strip('[ ]')
df.scores = df['scores'].astype(float)

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

df['demeaned_scores_atlas'] = demeaned_scores_atlas * 100
df['demeaned_scores_measure'] = demeaned_scores_measure * 100
df['demeaned_scores_classifier'] = demeaned_scores_classifier * 100

dic_atlas_n = {'aal_spm12': 1,
               'harvard_oxford': 2,
               'basc': 3,
               'ward': 4,
               'kmeans': 5,
               'ica': 6,
               'dictlearn': 7}

dic_measure_n = {'correlation': 1,
                 'partial correlation': 2,
                 'tangent': 3}

dic_classifier_n = {'svc_l1': 1,
                    'svc_l2': 2,
                    'ridge': 3}

###############################################################################
# plot to pdf, png and svg
hue = 'dataset'

# atlas
x = 'demeaned_scores_atlas'
y = 'atlas'
df['rank'] = df['atlas'].map(dic_atlas_n)
df.sort_values(by=['rank'], inplace=True)
stripplot_mean_score(df, save_path, dic_model_dim, suffix=save_sufix, x=x,
                     y=y, hue=hue, style='whitegrid', fontsize=12, jitter=0,
                     figsize=(5, 6), leg_pos='lower left', leg_borderpad=0.08)

# measure

x = 'demeaned_scores_measure'
y = 'measure'
df['rank'] = df['measure'].map(dic_measure_n)
df.sort_values(by=['rank'], inplace=True)
stripplot_mean_score(df, save_path, dic_model_dim, suffix=save_sufix, x=x,
                     y=y, hue=hue, style='whitegrid', fontsize=12, jitter=0,
                     figsize=(5, 4.5), leg_pos=(-0.02, -0.01), leg_borderpad=0.07)
                     #(-0.06, -0.04)


# classifier

x = 'demeaned_scores_classifier'
y = 'classifier'
df['rank'] = df['classifier'].map(dic_classifier_n)
df.sort_values(by=['rank'], inplace=True)
stripplot_mean_score(df, save_path, dic_model_dim, suffix=save_sufix, x=x,
                     y=y, hue=hue, style='whitegrid', fontsize=12, jitter=0,
                     figsize=(4, 3), leg_pos='lower left', leg_borderpad=0.1,
                     leg_ncol=1)
