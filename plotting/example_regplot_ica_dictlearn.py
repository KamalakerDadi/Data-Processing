"""Example to demonstrate linear relationship through regression
using seaborn regplot between parameters 'n_regions' as x variable
and 'scores' as y variable in regplot. Then using striplot
we choose hue='atlas' to demonstrate visual comparisons of behaviour
of regression plot across atlas extraction methods (ica vs dictlearn)
"""

import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

import load_data

path_ica = '../Experiments/COBRE/ICA/scores_ica.csv'
path_dict = '../Experiments/COBRE/DictLearn/scores_dictlearn.csv'

paths = [path_ica, path_dict]
data = load_data._pandas_data_frame_list_of_paths_concat(paths)

sns.set(color_codes=True)
sns.set_style("whitegrid", {'axes.edgecolor': '.6', 'grid.color': '.6'})
sns.set_palette('dark')

scatter_kws = {'s': 5}
line_kws = {'lw': 2}

fig, ax = matplotlib.pyplot.subplots(figsize=(4, 3.5), squeeze=True)

NAMES = {'ica': 'ICA',
         'dictlearn': 'Dict-learning',
         }

for label in ['ica', 'dictlearn']:
    this_data = data[(data['atlas'] == label) &
                     (data['classifier'] == 'svc_l2') &
                     (data['measure'] == 'tangent')]
    sns.regplot(x='n_regions', y='scores', data=this_data, lowess=True,
                ax=ax, label=NAMES[label],
                scatter_kws=scatter_kws,
                line_kws=line_kws,
                )

ax = plt.gca()
ax.axis('tight')
ax.set_ylim(.6, 1)

ax.legend(scatterpoints=1, frameon=True, fontsize=12, markerscale=3,
          borderaxespad=0, handletextpad=.2)
# ax.legend(loc="upper right")
ax.set_ylabel('Prediction scores', size=15)
ax.set_xlabel('Number of regions extracted', size=15)
plt.tight_layout(pad=.1)
plt.savefig('n_regions_vs_scores.pdf')
plt.close()
