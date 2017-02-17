""" A simple example to show how this function works.
"""

####################################################################
# Grab atlas from Nilearn

from nilearn import datasets

atlas_name = 'cort-maxprob-thr0-1mm'
harvard = datasets.fetch_atlas_harvard_oxford(atlas_name=atlas_name)

#####################################################################
# Required inputs for function find_region_names_using_cut_coords

dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)]
atlas_img = harvard.maps
labels = harvard.labels

######################################################################
# Grab function

from nilearn_utils import find_region_names_using_cut_coords

l, n = find_region_names_using_cut_coords(dmn_coords, atlas_img,
                                          labels=labels)

# where 'l' indicates new labels generated according to given atlas labels and
# coordinates
new_labels = l

# where 'n' indicates brain regions names labelled, generated according to given
# labels
region_names_involved = n

######################################################################
# Let's visualize
from nilearn.image import load_img
from nilearn import plotting
from nilearn.image import new_img_like

atlas_img = load_img(atlas_img)
affine = atlas_img.get_affine()
atlas_data = atlas_img.get_data()

for i, this_label in enumerate(new_labels):
    this_data = (atlas_data == this_label)
    this_data = this_data.astype(int)
    this_img = new_img_like(atlas_img, this_data, affine)
    plotting.plot_roi(this_img, cut_coords=dmn_coords[i],
                      title=region_names_involved[i])

plotting.show()
