"""Example script to run strip plot comparisons and saving them to pdf
"""

import load_data

path_ica = '../Experiments/COBRE/ICA/scores_ica.csv'
path_dict = '../Experiments/COBRE/DictLearn/scores_dictlearn.csv'

paths = [path_ica, path_dict]
data = load_data._pandas_data_frame_list(paths)

import plots

plots.stripplot_to_pdf(data, 'this.pdf', x='dimensionality', y='n_regions',
                       fontsize=0.5)
