import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os

class Model(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return out

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Agent:
    games_played = 0

    def __init__(self, model, hyperparam):
        self.model = model
        self.lr = hyperparam['LEARNING_RATE']
        self.gamma = hyperparam['GAMMA']
        self.optimizer = hyperparam['OPTIMIZER'](self.model.parameters(), lr=self.lr)
        self.loss_function = hyperparam['LOSS_FUNCTION']
        self.memory = deque(maxlen=hyperparam['MAX_MEMORY'])
        self.ep = hyperparam['NUM_EPISODES']
        self.batch = hyperparam['BATCH_SIZE']

    def get_state(self, board):
        board = np.array(board).flatten()
        return np.append(np.append(board == 1, board == -2), board == 0) * np.ones(3 * 9)

    def select_action(self, state):
        if random.randint(0, self.ep) < max(self.ep - self.games_played, self.ep * 0.01):
            return random.randint(0, 8)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            return torch.argmax(self.model(state))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def optimize_model(self):
        if len(self.memory) > self.batch:
            mini_sample = random.sample(self.memory, self.batch)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)

        prediction = self.model(states)
        target = prediction.clone()

        for i in range(len(dones)):  # len(dones) is number of samples
            new_Q = rewards[i]
            if not dones[i]:
                new_Q = rewards[i] + self.gamma * torch.max(self.model(next_states[i]))

            target[i][torch.argmax(actions[i]).item()] = new_Q

        self.optimizer.zero_grad()
        loss = self.loss_function(target, prediction)
        loss.backward()

        self.optimizer.step()
