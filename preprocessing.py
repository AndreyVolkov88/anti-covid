# common packages
import numpy as np
import os
from glob import glob
import copy
import cv2
from math import *
import matplotlib.pyplot as plt
from functools import reduce
# reading in dicom files
import pydicom
# skimage image processing packages
from skimage import measure, morphology
from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops
# scipy linear algebra functions
from scipy.linalg import norm
import scipy.ndimage
# plotly 3D interactive graphs
import plotly
from plotly.graph_objs import *
# set plotly credentials here

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def load_scan(path):
    slices = [pydicom.dcmread(s) for s in
              glob(path+'\*.dcm')]
    slices = [s for s in slices if 'SliceLocation' in s]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def get_patients(dir_path):
    patient_list=[]
    for path in glob(dir_path+'\*\*\*'):
        # patient_path = os.path.join(dir_path,path)
        patient_dicom = load_scan(path)
        patient_pixels = get_pixels_hu(patient_dicom)
        patient_list.append(patient_pixels)
    return patient_list

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image >= -700, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    # Improvement: Pick multiple background labels from around the patient
    # More resistant to “trays” on which the patient lays cutting the air around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to
    # something like morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets inside body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

def get_segmented_lung(patient_pixels):
    # get masks
    segmented_lungs = segment_lung_mask(patient_pixels,
                                        fill_lung_structures=False)
    segmented_lungs_fill = segment_lung_mask(patient_pixels,
                                             fill_lung_structures=True)
    # internal_structures = segmented_lungs_fill - segmented_lungs
    # isolate lung from chest
    copied_pixels = copy.deepcopy(patient_pixels)
    for i, mask in enumerate(segmented_lungs_fill):
        get_high_vals = mask == 0
        copied_pixels[i][get_high_vals] = 0
    seg_lung_pixels = copied_pixels
    return seg_lung_pixels

def image_processing(patient_image, n_slices, img_size):
    transformed_image = patient_image[:, 100:400, 70:430]
    transformed_image = cv2.resize(transformed_image, (transformed_image.shape[1], n_slices))
    stub = []
    for slice in transformed_image:
        stub.append(cv2.resize(slice, (img_size, img_size)))
    transformed_image = np.array(stub)
    return transformed_image

def process_data(dirpath, img_size, n_slices, segment=True):
    patients = get_patients(dirpath)
    data = []
    if segment:
        for patient in patients:
            img = get_segmented_lung(patient)
            data.append(image_processing(img, n_slices, img_size))
    else:
        for patient in patients:
            data.append(image_processing(patient, n_slices, img_size))
    return np.array(data)
