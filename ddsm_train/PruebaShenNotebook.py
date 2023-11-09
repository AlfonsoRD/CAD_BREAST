#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:48:16 2023

@author: c4ndypuff
"""

import pandas as pd
import numpy as np
import keras
import os 

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from keras.models import load_model
res_mod = load_model('/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/DDSM_Th128_cluser3_V3.h5', compile=False)
vgg_mod = load_model('/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/ddsm_vgg16_s10_512x1.h5', compile=False)
hybrid_mod = load_model('/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/ddsm_vgg16_s10_[512-512-1024]x2_hybrid.h5', compile=False)


from dm_image import DMImageDataGenerator
test_imgen = DMImageDataGenerator(featurewise_center=True)
test_imgen.mean = 44.4
test_generator = test_imgen.flow_from_directory(
    '/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/TestP', target_size=(1152, 896), target_scale=None,
    rescale_factor=1,
    equalize_hist=False, dup_3_channels=True, 
    classes=['neg', 'pos'], class_mode='categorical', batch_size=4, 
    shuffle=False)

from dm_keras_ext import DMAucModelCheckpoint
res_auc, res_y_true, res_y_pred = DMAucModelCheckpoint.calc_test_auc(
    test_generator, res_mod, test_samples=test_generator.nb_sample, return_y_res=True)
print (res_auc)



# from dm_keras_ext import DMAucModelCheckpoint
# res_auc_aug, res_y_true_aug, res_y_pred_aug = DMAucModelCheckpoint.calc_test_auc(
#     test_generator, res_mod, test_samples=test_generator.nb_sample, return_y_res=True, test_augment=True)
# print (res_auc_aug)







# from dm_keras_ext import DMAucModelCheckpoint
# vgg_auc, vgg_y_true, vgg_y_pred = DMAucModelCheckpoint.calc_test_auc(
#     test_generator, vgg_mod, test_samples=test_generator.nb_sample, return_y_res=True)
# print (vgg_auc)


# from dm_keras_ext import DMAucModelCheckpoint
# hybrid_auc, hybrid_y_true, hybrid_y_pred = DMAucModelCheckpoint.calc_test_auc(
#     test_generator, hybrid_mod, test_samples=test_generator.nb_sample, return_y_res=True)
# print (hybrid_auc)
