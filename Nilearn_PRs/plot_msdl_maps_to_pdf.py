"""
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from nilearn import datasets, plotting

msdl = datasets.fetch_atlas_msdl()

maps = msdl.maps
labels = msdl.labels
networks = msdl.networks

from nilearn.image import load_img, index_img, iter_img

maps_img = load_img(maps)
n_maps = maps_img.shape[3]

list_of_nodes = [[0, 1], [2], [3, 4, 5, 6], [7], [8], [9, 10, 11, 12],
                 [13], [14, 15, 16], [17, 18], [19, 20, 21], [22, 23, 24],
                 [25, 26], [27, 28, 29, 30, 31], [32], [33], [34, 35, 36],
                 [37, 38]]
list_of_imgs = []
list_of_titles = []

for node_ in list_of_nodes:
    cur_img = index_img(maps_img, node_)
    list_of_imgs.append(cur_img)
    title_ = []
    for i in node_:
        title_.append(msdl.labels[i])
    list_of_titles.append(title_)

cmap = plt.cm.bwr
color_list = cmap(np.linspace(0, 1, n_maps))
separate_colors = []

from nilearn.plotting import cm

for i in range(len(color_list)):
    cmap_ = cm.alpha_cmap(color_list[i])
    separate_colors.append(cmap_)


i = 0
nrows = 6
ncols = 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10),
                         squeeze=True)
cmaps = ['gist_earth', 'terrain', 'ocean', 'gist_stern',
         'brg', 'CMRmap', 'cubehelix', 'gnuplot', 'gnuplot2',
         'gist_ncar', 'nipy_spectral', 'jet', 'rainbow',
         'gist_rainbow', 'hsv', 'flag', 'prism']
for ii in range(nrows):
    for jj in range(ncols):
        if i < 17:
            cur_img = index_img(list_of_imgs[i], 0)
            cut_coords = plotting.find_xyz_cut_coords(cur_img)
            plotting.plot_prob_atlas(list_of_imgs[i],
                                     cmap=cmaps[i],
                                     view_type='filled_contours',
                                     threshold=0.1, alpha=0.8,
                                     axes=axes[ii, jj], figure=fig,
                                     draw_cross=False,
                                     cut_coords=cut_coords)
            plt.title(list_of_titles[i], fontsize=10,
                      horizontalalignment='center')
            i += 1
        else:
            axes[5, 2].remove()
            plt.savefig('maps.pdf')
            plt.close()
