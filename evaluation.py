 
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
test_timestamps_dataset = np.load('/home/faster/Documents/Diffusion-weather-prediction/timestamps2021.npy',allow_pickle = True)

# normalization 

maxRtrain = train_dataset.max()
train_dataset = train_dataset / maxRtrain
test_dataset = test_dataset / maxRtrain

maxWtrain = train_wind_dataset.max()
maxWtest = test_wind_dataset.max()
train_wind_dataset = train_wind_dataset / maxWtrain
test_wind_dataset = test_wind_dataset / maxWtest

# generator definition 

train_generator50 = generators.DataGenerator(train_dataset,batch_size,0.5,train_timestamps_dataset,train_wind_dataset)
test_generator50 = generators.DataGenerator(test_dataset,batch_size,0.5,test_timestamps_dataset,test_wind_dataset)
test_generator20 = generators.DataGenerator(test_dataset,batch_size,0.2,test_timestamps_dataset,test_wind_dataset)
full_test_generator50 = generators.FullDataGenerator(test_dataset,batch_size,0.5,test_timestamps_dataset,test_wind_dataset)


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


# pre-calculated normalizer on the whole training set 
#normer = np.load("normer.npy")
model.normalizer.adapt(train_generator50.__getitem__(1))
#del normer


# Load weights
model.network.load_weights("weights/3diffusion_addons_ema")
model.ema_network.load_weights("weights/3diffusion_addons_ema")


# Single Diffusion 

def experiment(generator = test_generator50, n_iter=29):
    history = np.zeros((n_iter,3))
    raw_data = np.zeros((n_iter,batch_size,96,96,6))
    
    #define final mse array 
    mses = np.zeros((3))
    for i in range(n_iter):
        
        #select a random batch in the test set  
        sample = generator.__getitem__(i)
        # save a copy as ground truth 
        hist = np.copy(sample)
        #normalize sample before generation 
        sample = model.normalizer(sample)
        # compute generation with 15 diffusion steps 
        tmp = model.generate2(np.copy(sample),15)
        
        # Denormalize prediction and g.t.
        hist = hist * maxRtrain
        tmp = tmp * maxRtrain
        
        raw_data[i,:,:,:,:3] = hist[:,:,:,-3:]
        raw_data[i,:,:,:,3:] = tmp[:,:,:,-3:]
        
        
        #print(metrics[i,2])
        
        # compute the metric, sum on last two axis, mean on first. 
        mse = np.mean(np.sum((hist[:,:,:,-3:]-tmp[:,:,:,-3:])**2,axis=(1,2)),axis=0)
        history[i] = mse 
        print(mse)
        
        # add 3 relevant meteric values to array 
        mses += mse[-3:]
    # return average of all mses
    return mses / n_iter, history, raw_data

exp,hist,metrics = experiment(full_test_generator50,29)

plt.plot(hist)
print(exp)


utils.metrics_aggregator(metrics,thresh).mean(axis=0)

# Ensamble diffusion
def experiment2(generator = test_generator50, n_iter=10, ensamble_iter = 15):
    #define final mse array 
    mses = np.zeros((n_iter,3))
    raw_data = np.zeros((n_iter,batch_size,96,96,6))
    
    for i in range(n_iter):
        #select a random batch in the test set 
        test = generator.__getitem__(i)
        
        print(i)
        
        # define an accumulator variable for ensamble 
        res = np.zeros([batch_size,ensamble_iter, 96,96, 16])
        # run ensamble iterations
        for j in range(ensamble_iter):

            #make a copy of the random batch 
            sample = np.copy(test)
            
            #normalize sample before generation
            sample = model.normalizer(sample)
            # compute generation with 15 steps 
            tmp = model.generate2(np.copy(sample),15)
            
            # denormalize prediction
            tmp = tmp * maxRtrain
            
            #save prediction in accumulator 
            res[:,j] = tmp 
        
        # average all predictions in the ensamble
        average = np.mean(res,axis=1)
        # denormalize ground truth
        hist = test * maxRtrain
        
        raw_data[i,:,:,:,:3] = np.copy(hist[:,:,:,-3:])
        raw_data[i,:,:,:,3:] = np.copy(average[:,:,:,-3:])
        
        #compute mse between g.t. and average ensamble, sum on 2nd 3rd axis mean on 1st. 
        mse = np.mean(np.sum((hist[:,:,:,-3:]-average[:,:,:,-3:])**2,axis=(1,2)),axis=0)

        mses[i] = mse 
        #sum all mses 
        print(mses)
        #print(mses / i)
            
    #average mses by number of iterations
    return mses, raw_data

res_ens,raw = experiment2(full_test_generator50,29,15)

plt.plot(hist)
print(exp)

utils.metrics_aggregator(raw,thresh).mean(axis=0)


