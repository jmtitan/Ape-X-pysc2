'''Replay buffer, learner and actor'''
import time
import os
from copy import deepcopy
import threading
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from agent import SmartAgent
from torch.cuda.amp import GradScaler
import numpy as np
from model import Network
import environment
import config
from pysc2.lib import actions as sc2_actions


############################## Replay Buffer ##############################
class Sequence:
    def __init__(self, sequence_len):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.td_error = []
        self.gamma = []
        self.length = sequence_len

    def __len__(self):
        return len(self.obs)

    def pull(self, data):
        if len(self.obs) + len(data[0]) < self.length:
            self.obs.extend(data[0])
            self.actions.extend(data[1])
            self.rewards.extend(data[2])
            self.td_error.extend(data[3])
            self.gamma.extend(data[4])
        else:
            i = 0
            while (len(self.obs) < self.length):
                self.obs.append(data[0][i])
                self.actions.append(data[1][i])
                self.rewards.append(data[2][i])
                self.td_error.append(data[3][i])
                self.gamma.append(data[4][i])
                i += 1
            self.obs = np.array(self.obs)
            self.actions = np.array(self.actions)
            self.rewards = np.array(self.rewards)
            self.td_error = np.array(self.td_error)
            self.gamma = np.array(self.gamma)


class SumTree:
    '''store priority for prioritized experience replay'''

    def __init__(self, capacity: int, tree_dtype=np.float64):
        self.capacity = capacity  # buffer_capacity

        self.layer = 1
        while capacity > 1:
            self.layer += 1
            capacity //= 2
        assert capacity == 1, 'capacity only allow n to the power of 2 size'

        self.tree_dtype = tree_dtype  # float32 is not enough
        self.tree = np.zeros(2 ** self.layer - 1, dtype=self.tree_dtype)

    def sum(self):
        assert np.sum(self.tree[-self.capacity:]) - self.tree[0] < 0.1, 'sum is {} but root is {}'.format(
            np.sum(self.tree[-self.capacity:]), self.tree[0])
        return self.tree[0]

    def __getitem__(self, idx: int):
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity - 1 + idx]

    def batch_sample(self, batch_size: int):
        p_sum = self.tree[0]
        interval = p_sum / batch_size

        prefixsums = np.arange(0, p_sum, interval, dtype=self.tree_dtype) + np.random.uniform(0, interval, batch_size)

        idxes = np.zeros(batch_size, dtype=np.int)
        for _ in range(self.layer - 1):
            nodes = self.tree[idxes * 2 + 1]
            idxes = np.where(prefixsums < nodes, idxes * 2 + 1, idxes * 2 + 2)
            prefixsums = np.where(idxes % 2 == 0, prefixsums - self.tree[idxes - 1], prefixsums)

        priorities = self.tree[idxes]
        idxes -= self.capacity - 1

        assert np.all(priorities > 0), 'idx: {}, priority: {}'.format(idxes, priorities)
        assert np.all(idxes >= 0) and np.all(idxes < self.capacity)

        return idxes, priorities

    def batch_update(self, idxes: np.ndarray, priorities: np.ndarray):
        idxes += self.capacity - 1
        self.tree[idxes] = priorities

        for _ in range(self.layer - 1):
            idxes = (idxes - 1) // 2
            idxes = np.unique(idxes)
            self.tree[idxes] = self.tree[2 * idxes + 1] + self.tree[2 * idxes + 2]

        # check
        assert np.sum(self.tree[-self.capacity:]) - self.tree[0] < 0.1, 'sum is {} but root is {}'.format(
            np.sum(self.tree[-self.capacity:]), self.tree[0])


class ReplayBuffer:
    def __init__(self, buffer_capacity=config.buffer_capacity, sequence_len=config.sequence_len,
                 alpha=config.prioritized_replay_alpha, beta=config.prioritized_replay_beta0,
                 batch_size=config.batch_size):

        self.buffer_capacity = buffer_capacity
        self.sequence_len = sequence_len
        self.num_sequences = buffer_capacity // sequence_len
        self.seq_ptr = 0

        # prioritized experience replay
        self.priority_tree = SumTree(buffer_capacity)
        self.alpha = alpha
        self.beta = beta

        self.batched_data = []  # storage data for thread finding data from locked buffer
        self.batch_size = batch_size

        self.lock = threading.Lock()

        self.obs_buf = [None for _ in range(self.num_sequences)]
        self.act_buf = [None for _ in range(self.num_sequences)]
        self.rew_buf = [None for _ in range(self.num_sequences)]
        self.gamma_buf = [None for _ in range(self.num_sequences)]
        self.size_buf = np.zeros(self.num_sequences, dtype=np.uint16)

    def __len__(self):
        return np.sum(self.size_buf).item()

    def run(self):
        self.background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        self.background_thread.start()

    def prepare_data(self):
        '''background thread'''
        while True:
            if len(self.batched_data) < 16:
                data = self.sample_batch(self.batch_size)
                self.batched_data.append(data)
            else:
                time.sleep(0.2)

    def get_data(self):
        '''call by learner to get batched data'''

        if len(self.batched_data) == 0:
            data = self.sample_batch(self.batch_size)
            return data
        else:
            return self.batched_data.pop(0)

    def add(self, sequence: Sequence):
        '''Call by actors to add data to replaybuffer

        Args:
            sequences: tuples of data, each tuple represents a slot
                in each tuple: 0 obs, 1 action, 2 reward, 3 gamma, 4 td_errors
        '''

        with self.lock:
            idxes = np.arange(self.seq_ptr * self.sequence_len, (self.seq_ptr + 1) * self.sequence_len)

            slot_size = np.size(sequence.obs, 0)

            self.size_buf[self.seq_ptr] = slot_size

            self.priority_tree.batch_update(idxes, np.power(sequence.td_error, self.alpha))

            self.obs_buf[self.seq_ptr] = torch.from_numpy(sequence.obs)
            self.act_buf[self.seq_ptr] = torch.from_numpy(sequence.actions)
            self.rew_buf[self.seq_ptr] = torch.from_numpy(sequence.rewards)
            self.gamma_buf[self.seq_ptr] = torch.from_numpy(sequence.gamma)

            self.seq_ptr = (self.seq_ptr + 1) % self.num_sequences

    def sample_batch(self, batch_size):
        '''sample one batch of training data'''
        batch_obs, batch_action, batch_reward, batch_next_obs, batch_gamma = [], [], [], [], []
        idxes, priorities = [], []

        with self.lock:

            idxes, priorities = self.priority_tree.batch_sample(batch_size)
            global_idxes = idxes // self.sequence_len
            local_idxes = idxes % self.sequence_len

            for global_idx, local_idx in zip(global_idxes.tolist(), local_idxes.tolist()):

                assert local_idx < len(self.act_buf[global_idx]), 'index is {} but size is {}'.format(local_idx, len(self.act_buf[global_idx]))

                # forward_steps = min(config.forward_steps, self.size_buf[global_idx].item()-local_idx)
                # it includes obs and next_obs
                obs = self.obs_buf[global_idx][local_idx]
                action = self.act_buf[global_idx][local_idx]
                reward = self.rew_buf[global_idx][local_idx]
                gamma = self.gamma_buf[global_idx][local_idx]
                # without done information->judge more times
                if len(self.obs_buf[global_idx]) <= local_idx + 1:
                    next_obs = obs
                else:
                    next_obs = self.obs_buf[global_idx][(local_idx + 1) % self.sequence_len]

                batch_obs.append(obs)
                batch_action.append(action)
                batch_reward.append(reward)
                batch_next_obs.append(next_obs)
                batch_gamma.append(gamma)

            # importance sampling weight
            min_p = np.min(priorities)
            weights = np.power(priorities / min_p, -self.beta)

            data = (
                torch.stack(batch_obs),
                torch.stack(batch_action).unsqueeze(1),
                torch.stack(batch_reward).unsqueeze(1),
                torch.stack(batch_next_obs),
                torch.stack(batch_gamma).unsqueeze(1),
                idxes,
                torch.from_numpy(weights.astype(np.float16)).unsqueeze(1),
                self.seq_ptr
            )
            return data

    def update_priorities(self, idxes: np.ndarray, priorities: np.ndarray, old_ptr: int):
        """Update priorities of sampled transitions"""
        with self.lock:

            # discard the indices that already been replaced by new data in replay buffer during training
            if self.seq_ptr > old_ptr:
                # range from [old_ptr, self.seq_ptr)
                mask = (idxes < old_ptr * self.sequence_len) | (idxes >= self.seq_ptr * self.sequence_len)
                idxes = idxes[mask]
                priorities = priorities[mask]
            elif self.seq_ptr < old_ptr:
                # range from [0, self.seq_ptr) & [old_ptr, self,capacity)
                mask = (idxes < old_ptr * self.sequence_len) & (idxes >= self.seq_ptr * self.sequence_len)
                idxes = idxes[mask]
                priorities = priorities[mask]

            self.priority_tree.batch_update(np.copy(idxes), np.copy(priorities) ** self.alpha)

    def ready(self):
        if len(self) >= config.learning_starts:
            return True
        else:
            return False


############################## Learner ##############################


class Learner:
    def __init__(self, buffer, env_name=config.env_name, lr=config.lr, eps=config.eps, num_act=config.num_actions):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env_name = env_name
        self.model = Network(num_act)
        self.model.to(self.device)
        self.model.train()
        self.tar_model = deepcopy(self.model)
        self.tar_model.eval()
        self.optimizer = Adam(self.model.parameters(), lr=lr, eps=eps)
        self.buffer = buffer
        self.counter = 0
        self.last_counter = 0
        self.done = False
        self.loss = 0
        self.last_bufferlen = 0
        self.store_weights()

    def get_weights(self):
        return self.weights

    def store_weights(self):
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        self.weights = state_dict

    def run(self):
        if self.ckpt != None:
            self.mode.load_state_dict(torch.load(self.ckpt))
        self.learning_thread = threading.Thread(target=self.train, daemon=True)
        self.learning_thread.start()

    def train(self):
        scaler = GradScaler()

        while self.counter < config.training_steps:
            data = self.buffer.get_data()
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_gamma, idxes, weights, old_ptr = data
            batch_obs, batch_action, batch_reward = batch_obs.to(self.device), batch_action.to(
                self.device), batch_reward.to(self.device)
            batch_next_obs, batch_gamma, weights = batch_next_obs.to(self.device), batch_gamma.to(
                self.device), weights.to(self.device)
            # double q learning
            with torch.no_grad():
                batch_action_ = self.model(batch_next_obs.to(torch.float32)).argmax(1).unsqueeze(1)
                batch_q_ = torch.gather(self.tar_model(batch_next_obs.to(torch.float32)), 1, batch_action_)

            batch_q = torch.gather(self.model(batch_obs.to(torch.float32)), 1, batch_action)
            td_error = (batch_q - (batch_reward + batch_gamma * batch_q_))
            priorities = td_error.detach().squeeze().abs().clamp(1e-4).cpu().numpy()
            # print(f'td:{td_error}')
            loss = (weights * self.huber_loss(td_error)).mean()
            self.loss += loss.item()
            # automatic mixed precision training
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_norm)
            scaler.step(self.optimizer)
            scaler.update()

            self.counter += 1

            self.buffer.update_priorities(idxes, priorities, old_ptr)

            # store new weights in shared memory
            if self.counter % 5 == 0:
                self.store_weights()

            # update target net, save model
            if self.counter % config.target_network_update_freq == 0:
                self.tar_model.load_state_dict(self.model.state_dict())

            if self.counter % config.save_interval == 0:
                torch.save(self.model.state_dict(),
                           os.path.join('models', '{}-{}.pth'.format(self.env_name, self.counter)))

        self.done = True

    def huber_loss(self, td_error, kappa=1.0):
        abs_td_error = td_error.abs()
        flag = (abs_td_error < kappa).half()
        return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)

    def stats(self, interval: int):
        print('number of updates (batch): {}'.format(self.counter))
        print('update speed/throughput rate(batch/s): {} '.format((self.counter - self.last_counter) / interval))
        if self.counter != self.last_counter:
            print('loss: {:.4f}'.format(self.loss / (self.counter - self.last_counter)))
        print('buffer size (frame): {}'.format(self.buffer.__len__()))
        print("Sampling rate (frame/s): {}".format((self.buffer.__len__() - self.last_bufferlen) / interval))
        self.last_counter = self.counter
        self.loss = 0
        self.last_bufferlen = self.buffer.__len__()
        return self.done


############################## Actor ##############################

class LocalBuffer:
    '''store transition of one episode'''

    def __init__(self, forward_steps=config.forward_steps,
                 sequence_len=config.sequence_len, gamma=config.gamma):
        self.forward_steps = forward_steps
        self.sequence_len = sequence_len
        self.gamma = gamma

    def __len__(self):
        return self.size

    def reset(self):
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.qval_buffer = []
        self.size = 0

    def add(self, action, reward, next_obs, q_value):
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.obs_buffer.append(next_obs)
        self.qval_buffer.append(q_value)
        self.size += 1
        # print(self.size)
        # print(" action:{}\n reward:{}\n next_obs:{}\n qval:{}\n".format(action,reward,next_obs,q_value))

    def finish(self, last_q_val: np.ndarray = None):
        cumulated_gamma = [self.gamma ** self.forward_steps for _ in range(self.size - self.forward_steps)]

        # if last_q_val is None, means done
        if last_q_val:
            self.qval_buffer.append(last_q_val)
            cumulated_gamma.extend([self.gamma ** i for i in reversed(range(1, self.forward_steps + 1))])
        else:
            self.qval_buffer.append(np.zeros_like(self.qval_buffer[-1]))
            cumulated_gamma.extend([0 for _ in range(self.forward_steps)])  # set gamma to 0 so don't need 'done'
        cumulated_gamma = np.array(cumulated_gamma, dtype=np.float16)
        self.obs_buffer = np.array(self.obs_buffer)
        self.action_buffer = np.array(self.action_buffer, dtype=np.int64)
        self.qval_buffer = np.concatenate(self.qval_buffer)
        self.reward_buffer = self.reward_buffer + [0 for _ in range(self.forward_steps - 1)]
        cumulated_reward = np.convolve(self.reward_buffer,
                                       [self.gamma ** (self.forward_steps - 1 - i) for i in range(self.forward_steps)],
                                       'valid').astype(np.float16)

        # num_sequences = self.size // self.sequence_len + 1

        # td_errors
        max_qval = np.max(self.qval_buffer[self.forward_steps:self.size + 1], axis=1)
        max_qval = np.concatenate((max_qval, np.array([max_qval[-1] for _ in range(self.forward_steps - 1)])))
        target_qval = self.qval_buffer[np.arange(self.size), self.action_buffer]
        td_errors = np.abs(cumulated_reward + max_qval - target_qval).clip(1e-4)

        # cut one episode to sequences to improve the buffer space utilization
        # sequences = []
        # for i in range(0, num_sequences*self.sequence_len, self.sequence_len):

        #     obs = self.obs_buffer[i:i+self.sequence_len]
        #     actions = self.action_buffer[i:i+self.sequence_len]
        #     rewards = cumulated_reward[i:i+self.sequence_len]
        #     td_error = td_errors[i:i+self.sequence_len]
        #     gamma = cumulated_gamma[i:i+self.sequence_len]
        # sequences.append((obs, actions, rewards, gamma, td_error))
        # print('td_errors{}'.format(np.shape(td_errors)))
        # print('obs_buffer{}'.format(np.shape(self.obs_buffer)))
        # print('cu_gamma{}'.format(np.shape(cumulated_gamma)))
        # print('action_buffer{}'.format(np.shape(self.action_buffer)))
        # print('cu_reward{}'.format(np.shape(cumulated_reward)))

        return (self.obs_buffer, self.action_buffer, cumulated_reward, td_errors, cumulated_gamma)


class Actor:
    def __init__(self, epsilon, learner, buffer):
        self.device = torch.device('cpu')
        self.env = environment.create_env()
        self.num_actions = config.num_actions
        self.model = Network(self.num_actions)
        self.model.eval()
        self.local_buffer = LocalBuffer()
        self.epsilon = epsilon
        self.learner = learner
        self.replay_buffer = buffer
        self.max_episode_length = config.max_episode_length
        self.counter = 0
        self.agent = SmartAgent(self.model, self.epsilon)


    def run(self):
        obs = self.reset()  # reset buffer and env each episode
        done = False
        sequence = Sequence(config.sequence_len)
        while True:
            action_no, action, q_val, rew = self.agent.step(obs[0])
            obs = self.env.step([action])
            if action_no == -1:
                continue
            q_val = q_val.numpy()
            done = obs[0].last()
            state = self.agent.previous_state
            self.local_buffer.add(action_no, rew, state, [q_val])

            if done or len(self.local_buffer) == self.max_episode_length:
                # finish and send buffer
                if done:
                    data = self.local_buffer.finish()
                else:
                    _, q_val = self.model.step(Variable(torch.unsqueeze(torch.FloatTensor(state), 0  )))
                    q_val = q_val.numpy()
                    data = self.local_buffer.finish(q_val)
                sequence.pull(data)
                # print("\nlen sequence:{}".format(len(sequence)))
                if (len(sequence) == config.sequence_len):
                    self.replay_buffer.add(sequence)
                    sequence = Sequence(config.sequence_len)  # 再次初始化
                done = False
                self.update_weights()
                self.reset()

            self.counter += 1

    def update_weights(self):
        '''load latest weights from learner'''
        weights = self.learner.get_weights()
        self.model.load_state_dict(weights)

    def reset(self):
        obs = self.env.reset()
        self.local_buffer.reset()
        self.agent.reset()
        return obs

