"""Example script to run strip plot comparisons and saving them to pdf
"""
import os
import seaborn as sns
import matplotlib.pyplot as plt

import load_data

covariance_estimator = 'LedoitWolf'
dataset_names = ['COBRE', 'ADNI']

base_path = os.path.join('../prediction_scores', covariance_estimator)

for dataset in dataset_names:
    path = os.path.join(base_path, dataset)
    path_ica = path + '/ICA/region_extraction/scores_ica.csv'
    path_dict = path + '/DictLearn/region_extraction/scores_dictlearn.csv'
    paths = [path_ica, path_dict]
    names = ['ICA', 'Dict-learning']
    data = load_data._pandas_data_frame_list(paths)
    sns.set_style("whitegrid", {'axes.edgecolor': '.6', 'grid.color': '.6'})
    sns.set_palette('dark')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 3.5),
                             squeeze=True, sharey=True)
    axes = axes.reshape(-1)
    for i, (ax, d, name) in enumerate(zip(axes, data, names)):
        sns.stripplot(x='dimensionality', y='n_regions', data=d, ax=ax,
                      jitter=.24, size=2.5)
        if i == 0:
            ax.set_ylabel('Number of regions extracted', size=15)
        else:
            ax.set_ylabel('')
        ax.set_xlabel('')
        plt.text(.5, 1.02, name, transform=ax.transAxes,
                 size=13, ha='center')
        for x in (1, 3):
            ax.axvspan(x - .5, x + .5, color='.9', zorder=-1)

    plt.text(.6, .025, 'Number of components', transform=fig.transFigure,
             size=15, ha='center')

    plt.tight_layout(rect=[0, .05, 1, 1])

    plt.savefig('n_regions_vs_dimensionality_' + dataset + '.pdf')
    plt.close()

