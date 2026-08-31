# image preprocessing  and data augmentation :

import tensorflow as tf
"""
# 1. DATA AUGMENTATION

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)


# 2. VALIDATION GENERATOR

val_gen = ImageDataGenerator(
    rescale=1./255
)

"""

