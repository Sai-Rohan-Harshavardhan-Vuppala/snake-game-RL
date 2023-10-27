import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from helper import plot
from model import QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
BLOCK_SIZE = 20

class Agent:
    def __init__(self):
        self.epsilon = 0
        self.gamma = 0.9
        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        l_pt = Point(head.x - BLOCK_SIZE, head.y)
        r_pt = Point(head.x + BLOCK_SIZE, head.y)
        u_pt = Point(head.x, head.y - BLOCK_SIZE)
        d_pt = Point(head.x, head.y + BLOCK_SIZE)
        
        direction = game.direction
        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN

        food = game.food

        state = [
            # danger straight
            (dir_l and game.is_collision(l_pt)) or
            (dir_r and game.is_collision(r_pt)) or
            (dir_u and game.is_collision(u_pt)) or
            (dir_d and game.is_collision(d_pt)),

            # danger right
            (dir_l and game.is_collision(u_pt)) or
            (dir_r and game.is_collision(d_pt)) or
            (dir_u and game.is_collision(r_pt)) or
            (dir_d and game.is_collision(l_pt)),

            # danger left
            (dir_l and game.is_collision(d_pt)) or
            (dir_r and game.is_collision(u_pt)) or
            (dir_u and game.is_collision(l_pt)) or
            (dir_d and game.is_collision(r_pt)),

            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food left
            food.x < game.head.x,
            # food right
            food.x > game.head.x,
            # food up
            food.y < game.head.y,
            # food down
            food.y > game.head.y

        ]
        return np.array(state, dtype=int)

    def get_action(self, state):
        # tradeoff for exploration vs exploitation
        self.epsilon = 80 - self.n_games      
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        # print(final_move)
        return final_move 

    
    def remember(self, old_state, final_move, reward, new_state, done):
        # popleft if the memory exceeds MAX_MEMORY
        self.memory.append((old_state, final_move, reward, new_state, done))

    def train_short_memory(self, old_state, final_move, reward, new_state, done):
        # print(reward)
        self.trainer.train_step(old_state, final_move, reward, new_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        old_states, final_moves, rewards, new_states, dones = zip(*mini_sample)
        self.trainer.train_step(old_states, final_moves, rewards, new_states, dones)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state = agent.get_state(game)
        move = agent.get_action(state)
        reward, done, score = game.play_step(move)
        new_state = agent.get_state(game)
        # print(reward)
        agent.train_short_memory(state, move, reward, new_state, done)
        agent.remember(state, move, reward, new_state, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()

            print('Game: ', agent.n_games, ' Score: ', score, 'Record: ', record)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
    
