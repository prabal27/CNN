#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:03:07 2018

@author: wizard
"""
# Part1: Building the CNN

# Importing the Libraries

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import os


# Initializing the CNN
classifier=Sequential()

# Convolution Layer
classifier.add(Conv2D(32,(3,3), input_shape=(64,64,3), activation='relu'))

# Pooling Layer
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(64,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Flattening 
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(output_dim=64, activation='relu'))
classifier.add(Dropout(p=0.5))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

#Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part2- Fitting the CNN
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/32,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000,
        use_multiprocessing=True)

# Save model
#script_dir = os.path.dirname(__file__)
#model_backup_path = os.path.join(script_dir, '../dataset/cat_or_dogs_model.h5')
#classifier.save(model_backup_path)
#print("Model saved to", model_backup_path)