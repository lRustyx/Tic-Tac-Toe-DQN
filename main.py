
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt


from tictactoe import Tictactoe
from DQNAgent import Model, Agent


INPUT_SIZE = 9*3
HIDDEN_SIZE = 128
OUTPUT_SIZE = 9


MODEL = Model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

hyperparam = {
    'NUM_EPISODES': 100,
    'LEARNING_RATE': 0.01,
    'OPTIMIZER': optim.Adam,
    'GAMMA': 0.95,
    'LOSS_FUNCTION':  nn.MSELoss(),
    'MAX_MEMORY': 50_000,
    'BATCH_SIZE': 1000,
}


game = Tictactoe()
agent = Agent(MODEL, hyperparam)

game_history = []
for episode in tqdm(range(agent.ep)):
    state = game.reset()
    game_done = False
    while not game_done:
        state = agent.get_state(game.get_board())
        action = agent.select_action(state)
        reward, game_done = game.play(action)
        next_state = agent.get_state(game.get_board())
        agent.remember(state, action, reward, next_state, game_done)

        if game_done:
            agent.games_played += 1
            game_history.append(reward)
            agent.optimize_model()


plt.plot(game_history)
plt.show()

