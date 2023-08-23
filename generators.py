import numpy as np 
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
import utils 


addon = np.load("/home/faster/Documents/WF-Unet_comparison/Wf_geopot_landSeaMask.npy")
tmp = np.zeros((2, 2 , 96 ,96))
for i in range(2):
    tmp[i] = np.copy(addon)
addon = tmp 
addon = addon.transpose((0,2,3,1))

 
# generator that randomly samples data  
class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, batch_size=24, min_rainfall = 0.0, time =None, wind = None) :
        self.data = data
        self.time = time
        self.wind = wind
        self.sequence = 11
        self.batch_size = batch_size
        self.num_samples = data.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.min_rainfall = min_rainfall # Percent of minimum rainfall per image
        
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
                        
        result = np.zeros((self.batch_size,96,96,self.sequence + 5))
        result[:,:,:,:2] = addon
        
        for i in range(self.batch_size):
            
            while True:
                
                random = np.random.randint(0,(self.num_samples-self.sequence)) 
                    
                items = self.data[random:random+self.sequence]

                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)
                
                if ((np.sum(items[:,:,:,-3] != 0) / (173*105)) < self.min_rainfall):
                    pass
                else:
                    result[i,:,:,2] = (utils.date_to_sinusoidal_embedding(self.time[random]) + 1) / 2
                    result[i,:,:,3:5] = np.transpose(self.wind[random+6:random+8],(1, 2, 0))
                    result[i,:,:,5:] = items[:,:96,:96,:]
                    break
        
        return result
    
#generator that returns all the sequences of data, from start to finish 
class FullDataGenerator(keras.utils.Sequence):
    def __init__(self, data, batch_size=24, min_rainfall = 0.0,time = None, wind = None):
        self.data = data
        self.wind = wind
        self.time = time
        self.counter = 0 
        self.sequence = 11
        self.batch_size = batch_size
        self.num_samples = data.shape[0]
        self.num_batches = int(np.ceil(3740 / self.batch_size)) #14353 train set 
        self.min_rainfall = min_rainfall # Percent of minimum rainfall per image
        
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
                        
        result = np.zeros((self.batch_size,96,96,self.sequence + 5))
        result[:,:,:,:2] = addon
        
        for i in range(self.batch_size):
            
            while True:
                
                items = self.data[self.counter:self.counter+self.sequence]

                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)
                
                if ((np.sum(items[:,:,:,-3] != 0) / (173*105)) < self.min_rainfall):
                    self.counter = self.counter + 1
                else:
                    result[i,:,:,2] = (utils.date_to_sinusoidal_embedding(self.time[self.counter]) + 1)/2
                    result[i,:,:,3:5] = np.transpose(self.wind[self.counter+6:self.counter+8],(1, 2, 0))
                    result[i,:,:,5:] = items[:,:96,:96,:]
                    self.counter = self.counter + 1
                    break
        
        return result    
