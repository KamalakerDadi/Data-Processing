""" A simple example to show how this function works.
"""

####################################################################
# Load atlases from FSL installation directory

from loader import fetch_fsl_atlases

fsl_dir = '/usr/share/fsl/'
atlas_header = 'HarvardOxford'
atlas_name = 'cort-maxprob-thr0-1mm'

fsl_atlas = fetch_fsl_atlases(fsl_dir, atlas_header, atlas_name)
atlas_img, labels = fsl_atlas.atlas_img, fsl_atlas.labels

#####################################################################
# Load coordinates
from nilearn_utils import load_cut_coords

path = 'Data/coordinates.csv'
coords, data = load_cut_coords(path)

######################################################################
# Grab function

from nilearn_utils import find_region_names_using_cut_coords

labels, region_names = find_region_names_using_cut_coords(coords, atlas_img,
                                                          labels=labels)

######################################################################
# Save them to csv file back again

import pandas as pd

labels = pd.DataFrame(labels)
labels.columns = ['labels']
# Join with labels
data = data.join(labels)

region_names = pd.DataFrame(region_names)
region_names.columns = ['names']
# Join with region names
data = data.join(region_names)

# Join with type of atlas
data['atlas_type'] = pd.Series([atlas_name] * len(data))

# Join with name of atlas
data['atlas'] = pd.Series([atlas_header] * len(data))

# Finally, save to csv back again
data.to_csv('new_data.csv')
