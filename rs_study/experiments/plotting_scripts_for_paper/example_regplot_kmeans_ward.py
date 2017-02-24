"""Example to demonstrate linear relationship through regression
using seaborn regplot between parameters 'dimensionality' as x variable
and 'scores' as y variable in regplot. Then using striplot
we choose hue='atlas' to demonstrate visual comparisons of behaviour
of regression plot across atlas extraction methods (kmeans vs ward)
"""

import os
import seaborn as sns
from matplotlib import pyplot as plt

import load_data

# Covariance estimator used to build connectomes
covariance_estimator = 'LedoitWolf'
dataset = 'COBRE'

base_path = os.path.join('../prediction_scores', covariance_estimator, dataset)
path_kmeans = base_path + '/KMeans/parcellations/scores_kmeans.csv'
path_ward = base_path + '/Ward/parcellations/scores_ward.csv'

paths = [path_kmeans, path_ward]
text_name = 'KMeans vs Ward'
data = load_data._pandas_data_frame_list_of_paths_concat(paths)

sns.set(color_codes=True)
sns.set_style("whitegrid", {'axes.edgecolor': '.6', 'grid.color': '.6'})
sns.set_palette('dark')

scatter_kws = {'s': 5}
line_kws = {'lw': 2}

fig, ax = plt.subplots(figsize=(4, 3.5), squeeze=True)

NAMES = {'kmeans': 'KMeans',
         'ward': 'Ward',
         }

for label in ['kmeans', 'ward']:
    this_data = data[(data['atlas'] == label) &
                     (data['classifier'] == 'svc_l2') &
                     (data['measure'] == 'tangent')]
    sns.regplot(x='dimensionality', y='scores', data=this_data, lowess=True,
                ax=ax, label=NAMES[label],
                scatter_kws=scatter_kws,
                line_kws=line_kws)

ax = plt.gca()
ax.axis('tight')
ax.set_ylim(.53, 1)

ax.legend(scatterpoints=1, frameon=True, fontsize=12, markerscale=3,
          borderaxespad=0, handletextpad=.2)

ax.set_ylabel('Prediction scores', size=15)
ax.set_xlabel('Number of clusters', size=15)

plt.tight_layout(pad=.1)
plt.savefig('clusters_vs_scores.pdf')
plt.close()
