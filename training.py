 
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

#num_epochs = 200  # train for at least 50 epochs for good results
image_size = 96
num_frames = 4

# sampling

min_signal_rate = 0.015
max_signal_rate = 0.95

# architecture

embedding_dims = 64 # 32
embedding_max_frequency = 1000.0
#widths = [32, 64, 96, 128]
widths = [64, 128, 256, 384]
block_depth = 2

# optimization

batch_size =  2
ema = 0.999

addon = np.load("/home/faster/Documents/WF-Unet_comparison/Wf_geopot_landSeaMask.npy")
tmp = np.zeros((batch_size, 2 , 96 ,96))
for i in range(batch_size):
    tmp[i] = np.copy(addon)
addon = tmp 
addon = addon.transpose((0,2,3,1))

train_dataset = np.load('/home/faster/Documents/Diffusion-weather-prediction/training_set2016-2020_105x173.npz')['arr_0']
test_dataset = np.load('/home/faster/Documents/Diffusion-weather-prediction/test_set2021_105x173.npz')['arr_0']

train_wind_dataset = np.load('/home/faster/Documents/Diffusion-weather-prediction/squared_wind2016-2020_105x173.npz')['arr_0']
train_wind_dataset = train_wind_dataset[:,:96,:96]
test_wind_dataset = np.load('/home/faster/Documents/Diffusion-weather-prediction/squared_wind2021_105x173.npz')['arr_0']
test_wind_dataset = test_wind_dataset[:,:96,:96]

train_timestamps_dataset = np.load('/home/faster/Documents/Diffusion-weather-prediction/timestamps2016-2020.npy',allow_pickle = True)
test_timestamps_dataset = np.load('/home/faster/Documents/Diffusion-weather-prediction/ timestamps2016-2020.npy',allow_pickle = True)

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

train_generator50 = DataGenerator(train_dataset,batch_size,0.5,train_timestamps_dataset,train_wind_dataset)
test_generator50 = DataGenerator(test_dataset,batch_size,0.5,test_timestamps_dataset,test_wind_dataset)
test_generator20 = DataGenerator(test_dataset,batch_size,0.2,test_timestamps_dataset,test_wind_dataset)
full_test_generator50 = FullDataGenerator(test_dataset,batch_size,0.5,test_timestamps_dataset,test_wind_dataset)


# diffusion model 

model = DiffusionModel(image_size, 13, 3, widths, block_depth)

optimizer=keras.optimizers.experimental.AdamW
model.compile(
    optimizer=optimizer(
        learning_rate=1e-5, weight_decay=1e-6
    ),
    loss=keras.losses.mean_absolute_error,
)
# pixelwise mean absolute error is used as loss


# sad hack I need to do :(
normer = np.load("normer4.npy")
model.normalizer.adapt(normer)
del normer


 def saver(epoch, logs):
            model.network.save_weights("weights/"+str(epoch)+"diffusion_addons")
            model.ema_network.save_weights("weights/"+str(epoch)+"diffusion_addons_ema")

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='i_loss', factor=0.5,
                              patience=2, min_lr=0)


# run training and plot generated images periodically
history = model.fit(
    train_generator50,
    epochs=10,
    steps_per_epoch=14000,
    #validation_data = val_generator,
    batch_size=batch_size,
    callbacks=[
        reduce_lr,
        keras.callbacks.LambdaCallback(on_epoch_end=model.plotter),
        keras.callbacks.LambdaCallback(on_epoch_end=saver)
    ],
)
















