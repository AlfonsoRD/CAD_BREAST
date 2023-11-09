#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:30:31 2023

@author: c4ndypuff
"""

import pandas as pd
import random
import pickle
import os
import shutil
from tqdm import tqdm 
import numpy as np
from sklearn.model_selection import train_test_split




def clear_folders(dicts):
    for folder in dicts:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


with open ('/home/c4ndypuff/Documentos/BackupOsmar/LabelsGroundTruth.pkl', 'rb') as f: 
    GT=pickle.load(f)

mal_cases=[]
norm_cases=[]
for exam in GT: 
    
    #Check if a image has a malignant case in left or right breast  
    if exam[1][2]==1 or exam[1][3]==1:
        
        #Check a image has a malignant detection of left breast, add path to CC and MLO view
        
        if exam[1][2]==1 and exam[1][3]==0: 
            
            #Add the path the view MLO and CC to mal_cases 
            a=exam[0] + '.LEFT_CC.png'
            b=exam[0] + '.LEFT_MLO.png'
            
            #Add the path the view MLO and CC to norm_cases
            # c=exam[0] + '.RIGHT_CC.png'
            # d=exam[0] + '.RIGHT_MLO.png'
            
            #append the cases 
            mal_cases.extend([a, b,])
            # norm_cases.extend([c, d])
            
        #Check a image has a malignant detection of right breast, add path to CC and MLO view
        if exam[1][2]==0 and exam[1][3]==1:
            
            #Add the path the view MLO and CC to norm_cases 
            a=exam[0] + '.LEFT_CC.png'
            b=exam[0] + '.LEFT_MLO.png'
            
            #Add the path the view MLO and CC to mal_cases
            c=exam[0] + '.RIGHT_CC.png'
            d=exam[0] + '.RIGHT_MLO.png'
            
            #append the cases 
            mal_cases.extend([c, d])
            # norm_cases.extend([a, b])
            
            
            
        #Check if a image has a malignant case on both sides    
        if exam[1][2]==1 and exam[1][3]==1:
            
            #Add the path the view MLO and CC to mal_cases 
            a=exam[0] + '.LEFT_CC.png'
            b=exam[0] + '.LEFT_MLO.png'
            
            #Add the path the view MLO and CC to mal_cases
            c=exam[0] + '.RIGHT_CC.png'
            d=exam[0] + '.RIGHT_MLO.png'
            
            mal_cases.extend([a, b, c, d])
    
            
    #Cases that not contains a malignant detection 
    else:
        #Add the path the view MLO and CC to norm_cases 
        a=exam[0] + '.LEFT_CC.png'
        b=exam[0] + '.LEFT_MLO.png'
        #Add the path the view MLO and CC to norm_cases
        c=exam[0] + '.RIGHT_CC.png'
        d=exam[0] + '.RIGHT_MLO.png'
        
        #append the cases 
        norm_cases.extend([a, b, c, d])
    

def filter_data_by_label(pkl_file_path, test_size=0.15, validation_size=0.1):
    # Load the data from the pkl file
    data = pd.read_pickle(pkl_file_path)
    
    # Convert to DataFrame if the data is a NumPy array
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=["filename", "mean_breast_tissue", "mean_background", "std_breast_tissue", "label"])
    
    # Get unique labels in the data
    labels = data['label'].unique()
    
    # Create a dictionary to store filtered data for each label
    filtered_data_dict = {}
    
    # Filter the data for each label and store it in the dictionary
    for label in labels:
        filtered_data_dict[label] = data[data['label'] == label]
    
    # Create train, validation, and test dictionaries
    train_data_dict = {}
    validation_data_dict = {}
    test_data_dict = {}
    
    # Split the data for each label into train, validation, and test sets
    for label, label_data in filtered_data_dict.items():
        train_data, test_data = train_test_split(label_data, test_size=test_size, random_state=42)
        train_data, validation_data = train_test_split(train_data, test_size=validation_size, random_state=42)
        train_data_dict[label] = train_data
        validation_data_dict[label] = validation_data
        test_data_dict[label] = test_data
    
    return train_data_dict, validation_data_dict, test_data_dict


# Example usage
train_data_dict, validation_data_dict, test_data_dict = filter_data_by_label('/home/c4ndypuff/Documentos/DDSM_Features_Otsu/ClusterDataThOtsu.pkl', test_size=0.15, validation_size=0.1)



def extract_filenames(class_num, data, norm_cases, mal_cases, img_number):
    # Get the data for the specified class number
    class_data = data[class_num]

    # Shuffle the data
    shuffled_data = class_data.sample(frac=1, random_state=42)

    # Split the data into filenames from norm_cases and mal_cases
    norm_data = shuffled_data[shuffled_data['filename'].isin(norm_cases)]
    mal_data = shuffled_data[shuffled_data['filename'].isin(mal_cases)]

    # Calculate the number of samples to select from norm_cases and mal_cases
    norm_count = int(0.75 * img_number)
    mal_count = img_number - norm_count

    # Select filenames from norm_data and mal_data
    norm_filenames = norm_data['filename'].tolist()[:norm_count]
    mal_filenames = mal_data['filename'].tolist()[:mal_count]

    return mal_filenames, norm_filenames



def images_to_dict(img_path, out_img_path, data, img_number):
    #Ext for filename in random_filenames:
    mal_filenames, norm_filenames=extract_filenames(class_num, data, norm_cases, mal_cases, img_number)
    with tqdm(total=img_number, position=0, leave=True, desc='Images Copy') as pbar:
        for filename in mal_filenames:
            src_path=os.path.join(img_path,filename)
            short_path=os.path.join(out_img_path,'pos')
            out_path=os.path.join(short_path,filename)
            shutil.copy(src_path, out_path)
            pbar.update(1)
        for filename in norm_filenames:
            src_path=os.path.join(img_path,filename)
            short_path=os.path.join(out_img_path,'neg')
            out_path=os.path.join(short_path,filename)
            shutil.copy(src_path, out_path)
            pbar.update(1)
    pbar.close()
    

def test_to_shen(class_num, data, img_path, out_img_path, mal_cases, norm_cases):
    test_split=data[class_num]
    test_file = test_split['filename'].tolist()
    with tqdm(total=len(test_file), position=0, leave=True, desc='Test Images Copy') as pbar:
        for filename in test_file:
            if filename in mal_cases:
                src_path=os.path.join(img_path,filename)
                short_path=os.path.join(out_img_path,'pos')
                out_path=os.path.join(short_path,filename)
                shutil.copy(src_path, out_path)
            elif filename in norm_cases:
                src_path=os.path.join(img_path,filename)
                short_path=os.path.join(out_img_path,'neg')
                out_path=os.path.join(short_path,filename)
                shutil.copy(src_path, out_path)
            pbar.update(1)
    pbar.close()



#Erease The Paths 
dicts=['/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/ValP/neg', '/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/ValP/pos', '/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/TrainP/neg','/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/TrainP/pos','/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/TestP/neg','/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/TestP/pos']

clear_folders(dicts)


# Loop over clusters
for class_num in range(4):  # This will loop over numbers 0, 1, 2, 3
    # General Arguments to the Cluster
    img_path = '/home/c4ndypuff/Documentos/DDSM_PNG'

    ### For Training ###
    out_img_path = '/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/TrainP'
    img_number = 4000
    data = train_data_dict
    print(f"Training Images Process of Cluster: {class_num}")
    images_to_dict(img_path, out_img_path, data, img_number)

    # For validation
    out_img_path = '/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/ValP'
    img_number = 400
    print(f"Validation Images Process of Cluster: {class_num}")
    data = validation_data_dict
    images_to_dict(img_path, out_img_path, data, img_number)

    # For Test
    out_img_path = '/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/TestP'
    data = test_data_dict
    test_to_shen(class_num, data, img_path, out_img_path, mal_cases, norm_cases)



# #General Arguments to the Cluster 

# class_num=3
# img_path='/home/c4ndypuff/Documentos/DDSM_PNG' 


# ### For Training ### 
# out_img_path='/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/TrainP'
# img_number=4000
# data=train_data_dict
# print(f"Training Images Process of Cluster: {class_num}")
# images_to_dict(img_path, out_img_path, data, img_number)

# #For validation
# out_img_path='/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/ValP'
# img_number=440
# print(f"Validation Images Process of Cluster: {class_num}")
# data=validation_data_dict
# images_to_dict(img_path, out_img_path, data, img_number)
           

# #For Test 
# out_img_path='/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/TestP'
# data=test_data_dict
# test_to_shen(class_num, data, img_path, out_img_path, mal_cases, norm_cases)









# def extract_filenames(class_num, data, img_number):
#     train_split = data[class_num]
#     filenames = train_split['filename'].tolist()
#     random_filenames = random.sample(filenames, img_number)
#     return random_filenames



# def images_to_dict(img_path, out_img_path, mal_cases, norm_cases, data, class_num, img_number):
#     #Ext for filename in random_filenames:
#     random_filenames=extract_filenames(class_num, data, img_number)
#     with tqdm(total=img_number, position=0, leave=True, desc='Images Copy') as pbar:
#         for filename in random_filenames:
#             if filename in mal_cases:
#                 src_path=os.path.join(img_path,filename)
#                 short_path=os.path.join(out_img_path,'pos')
#                 out_path=os.path.join(short_path,filename)
#                 shutil.copy(src_path, out_path)
#             else:
#                 src_path=os.path.join(img_path,filename)
#                 short_path=os.path.join(out_img_path,'neg')
#                 out_path=os.path.join(short_path,filename)
#                 shutil.copy(src_path, out_path)
#             pbar.update(1)
#     pbar.close()


# def test_to_shen(class_num, data, img_path, out_img_path, mal_cases, norm_cases):
#     test_split=data[class_num]
#     test_file = test_split['filename'].tolist()
#     with tqdm(total=len(test_file), position=0, leave=True, desc='Test Images Copy') as pbar:
#         for filename in test_file:
#             if filename in mal_cases:
#                 src_path=os.path.join(img_path,filename)
#                 short_path=os.path.join(out_img_path,'pos')
#                 out_path=os.path.join(short_path,filename)
#                 shutil.copy(src_path, out_path)
#             else:
#                 src_path=os.path.join(img_path,filename)
#                 short_path=os.path.join(out_img_path,'neg')
#                 out_path=os.path.join(short_path,filename)
#                 shutil.copy(src_path, out_path)
#             pbar.update(1)
#     pbar.close()
    
    

# #General Argumets to the Cluster 
# class_num=0
# img_path='/home/c4ndypuff/Documentos/DDSM_PNG' 
    
# #For Training 

# out_img_path='/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/TrainP'
# img_number=570
# data=train_data_dict
# images_to_dict(img_path, out_img_path, mal_cases, norm_cases, data, class_num, img_number)


# #For validation
# out_img_path='/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/ValP'
# img_number=57
# data=validation_data_dict
# images_to_dict(img_path, out_img_path, mal_cases, norm_cases, data, class_num, img_number)           


# #For Test 
# out_img_path='/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/TestP'
# data=test_data_dict
# test_to_shen(class_num, data, img_path, out_img_path, mal_cases, norm_cases)


