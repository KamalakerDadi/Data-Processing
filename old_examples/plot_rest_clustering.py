"""
Clustering methods to learn a brain parcellation from rest fMRI
====================================================================

We use KMeans and spatially-constrained Ward-clustering to create a set of
parcels. These parcels are particularly interesting for learning functional
connectomes and discriminate between controls and disease targets.

NOTE: parcellations.py should be in the same path to run this example.
"""

##################################################################
# Download a rest dataset and turn it to a data matrix
# -----------------------------------------------------
#
# We we download 30 subjects of the ADHD dataset from Internet

from nilearn import datasets
dataset = datasets.fetch_adhd(n_subjects=30)

# Phenotypic
phenotypic = dataset.phenotypic
labels = phenotypic['adhd']

# get data info
from nilearn_utils import data_info

shape, affine, _ = data_info(dataset.func[0])
###################################################################
# Clustering algorithms
# ----------------------

algorithms = ['minibatchkmeans', 'featureagglomeration']

###################################################################
# Masker
# ------
# Masking the data

from nilearn import input_data

# Fetch grey matter mask from nilearn shipped from ICBM templates
gm_mask = datasets.fetch_icbm152_brain_gm_mask(threshold=0.2)
masker = input_data.MultiNiftiMasker(mask_img=gm_mask, smoothing_fwhm=6.,
                                     target_shape=shape, target_affine=affine,
                                     standardize=True, detrend=True,
                                     mask_strategy='epi')
######################################################################
# Perform Parcellations on the brain data
# ---------------------------------------
# KMeans & Ward

# Memory/ Cache
from sklearn.externals.joblib import Memory

# Import class/object names as Clustering
from parcel import Parcellations
# Parameters
n_parcels = 100

# kmeans
kmeans = Parcellations(algorithm='minibatchkmeans', n_parcels=n_parcels,
                       n_components=100,
                       mask=masker, init='k-means++', memory='nilearn_cache',
                       memory_level=2, n_jobs=2, random_state=0,
                       verbose=1)
print("Perform KMeans brain parcellations")
kmeans.fit(dataset.func)
masker = kmeans.masker_
kmeans_labels_img = masker.inverse_transform(kmeans.kmeans_labels_)

# ward
ward = Parcellations(algorithm='featureagglomeration', n_parcels=n_parcels,
                     n_components=100,
                     mask=masker, linkage='ward', memory='nilearn_cache',
                     memory_level=2, n_jobs=2, random_state=0,
                     verbose=1)
print("Perform Ward Agglomeration based brain parcellations")
ward.fit(dataset.func)
masker = ward.masker_
ward_labels_img = masker.inverse_transform(ward.ward_labels_ + 1)

##################################################################
# Visualize results
# ------------------
#
# First we display the labels of the kmeans clustering in the brain.
#
# To visualize results, we need to transform the parcellation's labels back
# to a neuroimaging volume. For this, we use the NiftiMasker's
# inverse_transform method.
from nilearn.plotting import plot_roi, plot_epi, show

from nilearn.image import mean_img
mean_func_img = mean_img(dataset.func[0])

plot_roi(masker.mask_img_, mean_func_img, title="Mask image GM",
         display_mode='xz')

plot_roi(kmeans_labels_img, mean_func_img, title="KMeans parcellation",
         display_mode='xz')

plot_roi(ward_labels_img, mean_func_img, title="Ward Parcellation",
         display_mode='xz')

show()
