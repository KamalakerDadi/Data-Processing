"""Plot to diff between regions and non-regions (networks)
"""


def change_label_name(row, label, new_names):
    row[label] = new_names[row[label]]
    return row

###############################################################################
# csv results
import os

covariance_estimator = 'LedoitWolf'
dataset_paths = dict()
dataset_names = ['COBRE', 'ADNI', 'ADNIDOD', 'ACPI', 'ABIDE']
atlases = ['ICA', 'DictLearn', 'KMeans', 'BASC']

extensions = {'ICA': 'scores_ica.csv',
              'DictLearn': 'scores_dictlearn.csv',
              'KMeans': 'scores_kmeans.csv',
              'BASC': 'scores_basc.csv'}

base_path = os.path.join('../../prediction_scores', covariance_estimator)

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
# load and extract the data to best choices
from collections import OrderedDict

import pandas as pd
import load_data

data_list = []
for name in dataset_names:
    this_data_path = dataset_paths[name]
    this_data = load_data._pandas_data_frame_list_of_paths_concat(this_data_path)
    data_list.append(this_data)

whole_data = pd.concat(data_list)
whole_data = whole_data.drop('Unnamed: 0', axis=1)

dic_model_dim = OrderedDict([('ica', 80),
                             ('kmeans', 120),
                             ('dictlearn', 60),
                             ('basc', 122)
                             ])

choices_data = pd.DataFrame()
for key in dic_model_dim.keys():
    choices_data = choices_data.append(
        whole_data[(whole_data['atlas'] == key) &
                   (whole_data['dimensionality'] == dic_model_dim[key])])
#############################################################################
# Calculate demeaned scores on choices data

atlases = ['ica', 'dictlearn', 'basc', 'kmeans']
to_append = []

for key in atlases:
    this_atlas_data = choices_data[(choices_data['atlas'] == key)]

    def demean(group):
        return group - group.mean()

    # Take the average over iter_shuffle_split
    df = this_atlas_data.groupby(['atlas', 'classifier', 'measure', 'dataset',
                                  'region_extraction']).mean()
    df = df.reset_index()
    df.pop('smoothing_fwhm')
    df.pop('iter_shuffle_split')
    # demean scores
    demeaned_scores = df.groupby(['classifier', 'measure', 'dataset',
                                  'dimensionality'])['scores'].transform(demean)

    df['demeaned_scores'] = demeaned_scores
    to_append.append(df)

data_to_plot = pd.concat(to_append)

# Distribution in difference between regions ('yes') and non-regions ('no')
data_set_index = data_to_plot.set_index(['atlas', 'classifier',
                                         'measure', 'dataset'])
# Separate out regions ('yes') and non-regions ('no')
regions = data_set_index[(data_set_index['region_extraction'] == 'yes')]
non_regions = data_set_index[(data_set_index['region_extraction'] == 'no')]

difference = regions['demeaned_scores'] - non_regions['demeaned_scores']
difference = difference.reset_index()

# sort the data by atlases = ['ica', 'dictlearn', 'basc', 'kmeans']
new_data = []
for atlas in atlases:
    new_data.append(difference[(difference['atlas'] == atlas)])

difference = pd.concat(new_data)
#############################################################################
# plotting goes here
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

import aliases
from my_palette import (color_palette, atlas_palette,
                        datasets_palette)
new_atlas_names = aliases.new_names_atlas()

sns.set(color_codes=True)
sns.set_style("whitegrid", {'axes.edgecolor': '.6', 'grid.color': '.6'})
sns.set_palette('dark')

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 4), sharex=True)
x = 'demeaned_scores'
y = 'atlas'
hue = 'dataset'

difference = difference.apply(lambda x: change_label_name(x, y, new_atlas_names), axis=1)
ax.axvline(x=0, linewidth=4, zorder=0, color='0.6')
sns.violinplot(data=difference, x=x, y=y, ax=ax,
               inner='box', color='0.8', scale='width')
sns.stripplot(data=difference, x=x, y=y, hue=hue, edgecolor='gray',
              size=5, split=True, palette=datasets_palette,
              jitter=.25, ax=ax)
ax.set_xlabel('')
ax.set_ylabel('')

ax.legend(scatterpoints=1, frameon=True, fontsize=10, markerscale=0.6,
          handlelength=0.8, borderpad=.2, borderaxespad=3,
          handletextpad=.05, loc='lower left', ncol=2,
          columnspacing=0.2, bbox_to_anchor=(-0.75, -0.35))

for x in (1, 3):
    ax.axhspan(x - .5, x + .5, color='0.9', zorder=-1)

# make the positive labels with "+"
ax_xticklabels = []

for x in ax.get_xticks():
    if x > 0:
        ax_xticklabels.append('+' + str(x))
    else:
        ax_xticklabels.append(str(x))
# Headline on top of figure
xlabel = 'Difference in accuracy measured \n by regions and on networks'
plt.text(.67, .02, xlabel, transform=fig.transFigure,
         size=11.5, ha='center')

# top left label
plt.text(.25, 1.02, 'networks$>$regions', transform=ax.transAxes,
         size=10, ha='center')
# top right label
plt.text(.8, 1.02, 'regions$>$networks', transform=ax.transAxes,
         size=10, ha='center')

plt.tight_layout(rect=[0, 0.07, 0.98, 1])

save_name_pdf = 'regions_networks_difference_distribution.pdf'
plt.savefig(save_name_pdf)
