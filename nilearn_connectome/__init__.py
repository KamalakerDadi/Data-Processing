"""A quick wrapper for nilearn.connectome.connectivity_matrices + in addition
   this has option for confounds and vectorize (since PR is not yer merged).
   It will be removed after PR #1236 is merged in Nilearn.
"""

from .connectome_matrices import ConnectivityMeasure
