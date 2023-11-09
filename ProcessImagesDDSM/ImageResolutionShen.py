#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:27:56 2023

@author: ivan
"""

import cv2 
import os 
#import progressbar
from tqdm import tqdm


folder_dir="/home/c4ndypuff/Documentos/DDSM"


# Directorio de entrada
input_dir = "/home/c4ndypuff/Documentos/DDSM"

# Directorio de salida
output_dir = '/home/c4ndypuff/Documentos/DDSM_PNG'

# Crear el directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Lista de archivos de imagen en el directorio de entrada
img_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

# Barra de progreso
#bar = progressbar.ProgressBar(max_value=len(img_files))


with tqdm(total=len(img_files), position=0, leave=True) as pbar:
    # Tamaño de la imagen de salida
    
    width = 896
    
    # Procesamiento de imágenes
    for i, img_file in enumerate(img_files):
        # Leer la imagen
        img = cv2.imread(os.path.join(input_dir, img_file))
        
        size_image=img.shape
        
        height= int((size_image[0]*width)/size_image[1])
        
        
    
        # Cambiar la escala de la imagen
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        
        #Cambiar los canales de 3 a 1 
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Guardar la imagen en el formato PNG
        cv2.imwrite(os.path.join(output_dir, os.path.splitext(img_file)[0] + '.png'), img)
    
        # Actualizar la barra de progreso
        pbar.update(1)
        #bar.update(i + 1)
pbar.close()