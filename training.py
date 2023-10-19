 
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
import io
import imageio

import models 
import generators 
import utils 
from setup import *

# datasets and weights are available upon reasonable request

addon = np.load("addons/Wf_geopot_landSeaMask.npy")
tmp = np.zeros((batch_size, 2 , 96 ,96))
for i in range(batch_size):
    tmp[i] = np.copy(addon)
addon = tmp 
addon = addon.transpose((0,2,3,1))

train_dataset = np.load('data/training_set2016-2020_105x173.npz')['arr_0']
test_dataset = np.load('data/test_set2021_105x173.npz')['arr_0']

train_wind_dataset = np.load('data/squared_wind2016-2020_105x173.npz')['arr_0']
train_wind_dataset = train_wind_dataset[:,:96,:96]
test_wind_dataset = np.load('data/squared_wind2021_105x173.npz')['arr_0']
test_wind_dataset = test_wind_dataset[:,:96,:96]

train_timestamps_dataset = np.load('data/timestamps2016-2020.npy',allow_pickle = True)
test_timestamps_dataset = np.load('data/timestamps2021.npy',allow_pickle = True)

# normalization 

maxRtrain = train_dataset.max()
maxRtesr = test_dataset.max()
train_dataset = train_dataset / maxRtrain
test_dataset = test_dataset / maxRtesr

maxWtrain = train_wind_dataset.max()
maxWtest = test_wind_dataset.max()
train_wind_dataset = train_wind_dataset / maxWtrain
test_wind_dataset = test_wind_dataset / maxWtest

# generator definition 

train_generator50 = generators.DataGenerator(train_dataset,batch_size,0.5,train_timestamps_dataset,train_wind_dataset, addon)
test_generator50 = generators.DataGenerator(test_dataset,batch_size,0.5,test_timestamps_dataset,test_wind_dataset, addon)
test_generator20 = generators.DataGenerator(test_dataset,batch_size,0.2,test_timestamps_dataset,test_wind_dataset,addon)
full_test_generator50 = generators.FullDataGenerator(test_dataset,batch_size,0.5,test_timestamps_dataset,test_wind_dataset,addon)


# diffusion model 

model = models.DiffusionModel(image_size, 13, 3, widths, block_depth)

optimizer=keras.optimizers.experimental.AdamW
model.compile(
    optimizer=optimizer(
        learning_rate=1e-5, weight_decay=1e-6
    ),
    loss=keras.losses.mean_absolute_error,
)
# pixelwise mean absolute error is used as loss


# normer is obtained by sampling the training generator (large batch recommended)

#normer = np.load("normer4.npy")

# pre-calculated normalizer on the whole training set 
model.normalizer.adapt(train_generator50.__getitem__(1))
mean = np.load("addons/mean_normalizer.npz")['arr_0']
variance = np.load("addons/variance_normalizer.npz")['arr_0']
model.normalizer.mean = mean 
model.normalizer.variance = variance


def saver(epoch, logs):
    model.network.save_weights("weights/"+str(epoch)+"diffusion_addons")
    model.ema_network.save_weights("weights/"+str(epoch)+"diffusion_addons_ema")

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='i_loss', factor=0.5,
                              patience=2, min_lr=0)


# run training and plot generated images periodically
history = model.fit(
    train_generator50,
    epochs=30,
    steps_per_epoch=14000,
    #validation_data = val_generator,
    batch_size=batch_size,
    callbacks=[
        reduce_lr,
        #keras.callbacks.LambdaCallback(on_epoch_end=model.plotter),
        keras.callbacks.LambdaCallback(on_epoch_end=saver)
    ],
)
















