#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:31:36 2023

@author: c4ndypuff
"""

import numpy as np
import shutil
import cv2
import os
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans


#Function to import variables 
def pickle_to_file(file_name, data, protocol = pickle.HIGHEST_PROTOCOL):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol)

def extract_features(image):
    # Resize the image to 256x256.
    image = cv2.resize(image, (256, 256))
    # th, breast_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Calculate the mean value of breast tissue and background.
    breast_mask = (image > 128)
    breast_tissue = image[breast_mask]
    mean_breast_tissue = np.mean(breast_tissue)
    mean_background = np.mean(image[~breast_mask])

    # Calculate the standard deviation of the grayscale levels of the breast.
    std_breast_tissue = np.std(breast_tissue)

    # Return the extracted features as a dictionary.
    features = {'mean_breast_tissue': mean_breast_tissue,
                'mean_background': mean_background,
                'std_breast_tissue': std_breast_tissue}
                # 'threshold': th}
    return features


def extract_features_from_folder(folder_path, output_file, n_clusters=4):
    # Get a list of all image files in the folder.
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]

    # Define the data types for the fields in the structured array.
    dt = np.dtype([('filename', 'U100'),
                   ('mean_breast_tissue', np.float64),
                   ('mean_background', np.float64),
                   ('std_breast_tissue', np.float64),
                   # ('threshold', np.float64), #Se agreaga solo en el metodo de otsu 
                   ('label', int)])

    # Create an empty structured array to hold the results.
    results = np.empty(len(image_files), dtype=dt)

    # Loop through all images and extract features.
    for i, image_file in enumerate(tqdm(image_files, desc="Extracting features")):
        # Load the image.
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

        # Extract features from the image.
        features = extract_features(image)

        # Check if there are any NaN values in the features and replace them with 0.
        for key in features.keys():
            if np.isnan(features[key]):
                features[key] = 0

        # Add the features to the structured array.
        results[i] = (os.path.basename(image_file),
                      features['mean_breast_tissue'],
                      features['mean_background'],
                      features['std_breast_tissue'],
                       # features['threshold'], #Quita el signo de gato para agregar el valor de TH por Otsu 
                      -1)  # Initialize label as -1

    # Replace any remaining NaN values in the results with 0.
    results = np.nan_to_num(results)
    pickle_to_file('/home/c4ndypuff/Documentos/Features_Th_128.pkl', results)

    # # Calculate the mean and standard deviation over all the images.
    # mean_breast_tissue = np.mean(results['mean_breast_tissue'])
    # mean_background = np.mean(results['mean_background'])
    # std_breast_tissue = np.mean(results['std_breast_tissue'])
    # mean_th=np.mean(results['threshold'])

    # Create a 2D array of the features.
    X = np.column_stack((results['mean_breast_tissue'], results['mean_background'], results['std_breast_tissue']))

    # Cluster the images using KMeans.
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X)

    # Assign the cluster labels to the structured array.
    results['label'] = labels

    # Save the structured array to the output file.
    np.savetxt(output_file, results, delimiter=',', fmt='%s')

    # Print the path of the output file.
    print("Data saved to:", output_file)
    
    pickle_to_file('/home/c4ndypuff/Documentos/ClusterDataTh128.pkl', results)


folder_path = '/home/c4ndypuff/Documentos/DDSM_PNG'
output_file = '/home/c4ndypuff/Documentos/ClusterDataTh128.csv'
extract_features_from_folder(folder_path, output_file, n_clusters=4)
