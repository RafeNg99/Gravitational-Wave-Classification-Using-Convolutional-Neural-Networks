#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install torch torchvision torchaudio
#pip install -q nnAudio -qq
#pip install librosa 
#pip install efficientnet_pytorch -qq
#pip install tensorflow
#pip install -U efficientnet
#pip install opencv-python
#pip install -U segmentation-models
#pip install --upgrade tensorflow
#!pip3 install kaggle


# # Import Libraries

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import os
import json
import random
import collections
from glob import glob
from random import shuffle

import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras 
#import tensorflow.keras as keras
import tensorflow as tf
#import efficientnet.keras as efn
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, BatchNormalization, GlobalMaxPool2D, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from torch.utils import data as torch_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam, SGD
import segmentation_models as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# In[3]:

#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession

#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def convert_image_id_2_path(image_id: str) -> str:
    folder = "train"
    return "/state/partition1/ext_storage_c24/rafe/{}/{}/{}/{}/{}.npy".format(
        folder, image_id[0], image_id[1], image_id[2], image_id 
    )


df = pd.read_csv("/state/partition1/ext_storage_c24/rafe/train/training_labels.csv")
df

train_path = glob('/state/partition1/ext_storage_c24/rafe/train/*/*/*/*.npy')
len(train_path)

np.load(convert_image_id_2_path(df.iloc[0]["id"]))


# A data sample consists of 3 rows and 4096 columns. Each row represents the time series of different detectors (LIGO Hanford, LIGO Livingston & Virgo). The quantity is strain which is in the order ~${10^{-20}}$ recorded for 2 sec period sampled at 2048 Hz - 4096 data points.

np.load(convert_image_id_2_path(df.iloc[0]["id"])).shape


# # Data Visualization

sns.countplot(data = df, x = "target")

def visualize_sample(
    _id, 
    target, 
    colors=("red", "green", "blue"), 
    signal_names=("LIGO Hanford", "LIGO Livingston", "Virgo")
):
    path = convert_image_id_2_path(_id)
    x = np.load(path)
    plt.figure(figsize=(16, 7))
    for i in range(3):
        plt.subplot(4, 1, i + 1)
        plt.plot(x[i], color=colors[i])
        plt.legend([signal_names[i]], fontsize=12, loc="lower right")
        
        plt.subplot(4, 1, 4)
        plt.plot(x[i], color=colors[i])
    
    plt.subplot(4, 1, 4)
    plt.legend(signal_names, fontsize=12, loc="lower right")

    plt.suptitle(f"id: {_id} target: {target}", fontsize=16)
    plt.show()


# y-axis is the strain which is in the order ~${10^{-20}}$ recorded for 2 sec period and x-axis is the data point (4096 data points) in ths span of 2 seconds.

# In[10]:


for i in random.sample(df.index.tolist(), 3):
    _id = df.iloc[i]["id"]
    target = df.iloc[i]["target"]

    visualize_sample(_id, target)


import librosa
import librosa.display


# ## Signal Transformations - Constant Q-transform

# In mathematics and signal processing, the constant-Q transform, simply known as CQT transforms a data series to the frequency domain. It is related to the Fourier transform and very closely related to the complex Morlet wavelet transform. Its design is suited for musical representation.

import torch
from nnAudio.Spectrogram import CQT1992v2

#https://kinwaicheuk.github.io/nnAudio/v0.1.5/_autosummary/nnAudio.Spectrogram.CQT1992v2.html
Q_TRANSFORM = CQT1992v2(sr=2048,      #Sample rate
                        fmin=20,      #Min frequency
                        fmax=1024,    #Max frequency
                        hop_length=32) #The hop (or stride) size)

def visualize_sample_qtransform(
    _id, 
    target,
    signal_names=("LIGO Hanford", "LIGO Livingston", "Virgo"),
    sr=2048,
):
    x = np.load(convert_image_id_2_path(_id))
    plt.figure(figsize=(16, 5))
    for i in range(3):
        waves = x[i] / np.max(x[i])
        #Creates a Tensor from a numpy.ndarray                
        waves = torch.from_numpy(waves).float()
        image = Q_TRANSFORM(waves)
        
        plt.subplot(1, 3, i + 1)
        #Remove single-dimensional entries from the shape of an array
        #Then display data as an image
        plt.imshow(image.squeeze())
        plt.title(signal_names[i], fontsize = 14)

    plt.suptitle(f"id: {_id} target: {target}", fontsize = 16)
    plt.show()


for i in random.sample(df.index.tolist(), 3):
    _id = df.iloc[i]["id"]
    target = df.iloc[i]["target"]
    
    visualize_sample_qtransform(_id, target)


# # Set Seed

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed(42)


# # Split Dataset

x = df['id'][:156250]
y = df['target'][:156250].astype('int8').values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)


def get_qtransform(x):
    images = []
    for i in range(3):
        waves = x[i]/np.max(x[i])
        waves = torch.from_numpy(waves).float()
        image = Q_TRANSFORM(waves).squeeze().numpy()
        image = np.array(image)
        #image = image / np.max(image)
        images.append(image)

    return images


def dataset(Dataset):
    images = []
    for index in range(len(Dataset)):
        file_path = convert_image_id_2_path(Dataset[index])
        x = np.load(file_path)
        image = get_qtransform(x)
        images.append(image)
    return np.transpose(images, (0, 2, 3, 1))

x_train = dataset(x_train.values)
x_train = np.array(x_train)

x_test = dataset(x_test.values)
x_test = np.array(x_test)

lrr = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.01, patience = 2, min_lr = 0.00001)
estop = EarlyStopping(monitor = 'val_acc', patience = 3)

# # Modelling

def Model():
    inputs = layers.Input(shape = (69, 129, 3))
    densenet_layers = DenseNet121(include_top = False, input_shape = (224, 224, 3), weights = 'imagenet', pooling = 'max')
    sgd = SGD(lr = 0.001, momentum = 0.9, nesterov = False)
    model = Sequential()
    
    model.add(inputs)
    model.add(tf.keras.layers.experimental.preprocessing.Resizing(224, 224))
    model.add(keras.layers.Conv2D(3, 3, activation = 'relu', padding = 'same', kernel_regularizer = keras.regularizers.l2(0.001), use_bias = True))
    model.add(BatchNormalization())
    model.add(densenet_layers)
    model.add(Dense(1024, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.001), use_bias = True))
    model.add(Dense(512, activation = 'relu', use_bias = True))
    model.add(Dense(256, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.001), use_bias = True))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation = 'relu', use_bias = True))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = "sigmoid", use_bias = True))
    
    model.compile(optimizer = sgd,
                loss = "binary_crossentropy",
                metrics = ["acc"])

    return model


def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()
    
    return plot_history

sm.set_framework('tf.keras')
sm.framework()
batch_size = 4

model = Model()
model.build(input_shape = (4, 69, 129, 3))
model.summary()
history = model.fit(x_train, y_train, epochs = 30, batch_size = batch_size, validation_split = 0.2, callbacks = [lrr, estop], shuffle = True)

model.save("DenseNet201.h5")

y_pred = model.predict(x_test)
y_pred = [0 if val < 0.5 else 1 for val in y_pred]
print('The accuracy score is ' + str(accuracy_score(y_test, y_pred)) + '\n')
print('Confusion Matrix: \n' + str(confusion_matrix(y_test, y_pred)) + '\n') 

plot_history(history)
