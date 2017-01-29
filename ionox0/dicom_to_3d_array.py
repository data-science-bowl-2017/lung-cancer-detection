
# coding: utf-8

# In[ ]:

"""
https://www.kaggle.com/rmchamberlain/data-science-bowl-2017/dicom-to-3d-numpy-arrays/discussion
Converts the directory of DICOM files to a directory of .npy files
Writes out a JSON file mapping subject ID to 3D voxel spacing
"""
import json
from operator import itemgetter
import os

import dicom
import numpy as np

# Directory of original data
DICOM_DIR = '../luna_2016_data/subset9'
# Where to store npy arrays
NPY_DIR = 'stage1_npy'

def thru_plane_position(dcm):
    """Gets spatial coordinate of image origin whose axis
    is perpendicular to image plane.
    """
    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    position = tuple((float(p) for p in dcm.ImagePositionPatient))
    rowvec, colvec = orientation[:3], orientation[3:]
    normal_vector = np.cross(rowvec, colvec)
    slice_pos = np.dot(position, normal_vector)
    return slice_pos

def read_subject(subject_id):
    """ Read in the directory of a single subject and return a numpy array """
    directory = os.path.join(DICOM_DIR, subject_id)
    files = [os.path.join(directory, fname)
             for fname in os.listdir(directory) if fname.endswith('.dcm')]

    # Read slices as a list before sorting
    dcm_slices = [dicom.read_file(fname) for fname in files]

    # Extract position for each slice to sort and calculate slice spacing
    dcm_slices = [(dcm, thru_plane_position(dcm)) for dcm in dcm_slices]
    dcm_slices = sorted(dcm_slices, key=itemgetter(1))
    spacings = np.diff([dcm_slice[1] for dcm_slice in dcm_slices])
    slice_spacing = np.mean(spacings)

    # All slices will have the same in-plane shape
    shape = (int(dcm_slices[0][0].Columns), int(dcm_slices[0][0].Rows))
    nslices = len(dcm_slices)

    # Final 3D array will be N_Slices x Columns x Rows
    shape = (nslices, *shape)
    img = np.empty(shape, dtype='float32')
    for idx, (dcm, _) in enumerate(dcm_slices):
        # Rescale and shift in order to get accurate pixel values
        slope = float(dcm.RescaleSlope)
        intercept = float(dcm.RescaleIntercept)
        img[idx, ...] = dcm.pixel_array.astype('float32')*slope + intercept

    # Calculate size of a voxel in mm
    pixel_spacing = tuple(float(spac) for spac in dcm_slices[0][0].PixelSpacing)
    voxel_spacing = (slice_spacing, *pixel_spacing)

    return img, voxel_spacing

def convert_all_subjects():
    """ Converts all subjects in DICOM_DIR to 3D numpy arrays """
    subjects = os.listdir(DICOM_DIR)
    voxel_spacings = {}
    for subject in subjects:
        print('Converting %s' % subject)
        img, voxel_spacing = read_subject(subject)
        outfile = os.path.join(NPY_DIR, '%s.npy' % subject)
        np.save(outfile, img)
        voxel_spacings[subject] = voxel_spacing

    with open(os.path.join(NPY_DIR, 'voxel_spacings.json'), 'w') as fp:
        json.dump(voxel_spacings, fp)

if __name__ == '__main__':
    convert_all_subjects()

