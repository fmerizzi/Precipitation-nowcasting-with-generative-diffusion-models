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
