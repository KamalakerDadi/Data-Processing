"""Data loader
"""
import os
from xml.etree import ElementTree

from sklearn.datasets.base import Bunch


def fetch_fsl_atlases(fsl_dir, atlas_header, atlas_name):
    """Load atlases and its labels from FSL installation directory

    Parameters
    ----------
    fsl_dir : str
        Path to FSL install directory. Example: /usr/share/fsl

    atlas_header : str
        Name of the preferred atlas to fetch data. Valid headers are
        'HarvardOxford' or 'Cerebellum'.

    atlas_name : str
        Name of the atlas types. For example, cortial or sub-cortical or
        cortical lateralized for 'HarvardOxford'. Flirt or Fnirt based
        cerebellum atlases.
        Refer to FSL installation directory for names.

    Returns
    -------
    atlas_img : Nifti-like image
        Path to atlas image from FSL installed root directory.

    labels : list
        Labels of regions in the atlas
    """
    root = os.path.join('data', 'atlases')
    base_path = os.path.join(root, atlas_header, atlas_header)

    atlas_file = base_path + '-' + atlas_name + '.nii.gz'

    if atlas_header == 'HarvardOxford':
        if 'cortl' in atlas_name:
            label_file = 'HarvardOxford-Cortical-Lateralized.xml'
        elif 'sub' in atlas_name:
            label_file = 'HarvardOxford-Subcortical.xml'
        elif 'cortl' in atlas_name:
            label_file = 'HarvardOxford-Cortical.xml'
        else:
            raise ValueError("Unknown atlas_name is provided for "
                             "atlas_header='HarvardOxford'")

    if atlas_header == 'Cerebellum':
        if 'flirt' in atlas_name:
            label_file = 'Cerebellum_MNIflirt.xml'
        elif 'fnirt' in atlas_name:
            label_file = 'Cerebellum_MNIfnirt.xml'
        else:
            raise ValueError("Unknown atlas_name is provided for "
                             "atlas_header='Cerebellum'")

    atlas_img = os.path.join(fsl_dir,  atlas_file)
    label_file = os.path.join(fsl_dir, root, label_file)

    names = {}
    names[0] = 'Background'
    for label in ElementTree.parse(label_file).findall('.//label'):
        names[int(label.get('index')) + 1] = label.text
    names = list(names.values())

    return Bunch(atlas_img=atlas_img, labels=names)
