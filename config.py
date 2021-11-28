#################### env ####################
env_name = 'DefeatRoaches'
screen = 64
minimap = 64
num_actions = 7 #select 1\2 attack  up down left right
#################### worker.py ####################
lr = 1e-4
eps = 1e-3
grad_norm=40
batch_size = 512
learning_starts = 5120
save_interval = 5000
target_network_update_freq = 2500
gamma = 0.99
prioritized_replay_alpha = 0.6
prioritized_replay_beta0 = 0.4
forward_steps = 3  # n-step forward
training_steps = 1000000
buffer_capacity =  131072   #128*1024
max_episode_length = 15000
sequence_len = 1024  # cut one episode to sequences to improve the buffer space utilization

ckpt = './models/DefeatRoaches-185000.pth'
#################### train.py ####################
num_actors = 4
base_eps = 0.9
alpha = 0.7
log_interval = 10

#################### test.py ####################
save_plot = True
test_epsilon = 0.95