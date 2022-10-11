import random
import numpy as np
import os


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if state.ndim == 1:
            num_sample = 1
            list_sample = [(state, action, reward, next_state, done)]
        else:
            num_sample = state.shape[0]
            list_sample = list(zip(state, action, reward, next_state, done))

        augmented_size = min(self.capacity - len(self.buffer), num_sample)
        self.buffer.extend([None] * augmented_size)

        if self.position + num_sample <= self.capacity:
            self.buffer[self.position: (self.position + num_sample)] = list_sample
        else:
            tail_size = self.capacity - self.position
            head_size = num_sample - tail_size
            self.buffer[self.position:] = list_sample[:tail_size]
            self.buffer[:head_size] = list_sample[tail_size:]
        self.position = (self.position + num_sample) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
