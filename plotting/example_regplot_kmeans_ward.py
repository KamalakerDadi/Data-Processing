"""Example to demonstrate linear relationship through regression
using seaborn regplot between parameters 'dimensionality' as x variable
and 'scores' as y variable in regplot. Then using striplot
we choose hue='atlas' to demonstrate visual comparisons of behaviour
of regression plot across atlas extraction methods (kmeans vs ward)
"""

import seaborn as sns
import matplotlib

import load_data

path_kmeans = '../Experiments/COBRE/KMeans/scores_kmeans.csv'
path_ward = '../Experiments/COBRE/Ward/scores_ward.csv'

paths = [path_kmeans, path_ward]
text_name = 'KMeans vs Ward'
data = load_data._pandas_data_frame_list_of_paths_concat(paths)

sns.set_style("whitegrid", {'axes.edgecolor': '.6', 'grid.color': '.6'})
sns.set_palette('dark')
sns.set(color_codes=True)
scatter_kws = {'s': 5, 'c': ['g', 'b']}
line_kws = {'lw': 2, 'color': 'r'}
matplotlib.rc("legend", fontsize=10)

fig, ax = matplotlib.pyplot.subplots(figsize=(4, 3.5), squeeze=True)

sns.regplot(x='dimensionality', y='scores', data=data, lowess=True,
            ax=ax, label=['kmeans', 'ward'],
            scatter_kws=scatter_kws,
            line_kws=line_kws)
ax.legend(scatterpoints=2)
# ax.legend(loc="upper right")
ax.set_ylabel('Prediction scores', size=15)
ax.set_xlabel('Number of increase in clusters', size=15)
matplotlib.pyplot.text(.5, 1., text_name, transform=ax.transAxes,
                       size=15, ha='center')
matplotlib.pyplot.tight_layout(rect=[0, .05, 1, 1])
matplotlib.pyplot.savefig('clusters_vs_scores.pdf')
matplotlib.pyplot.close()
