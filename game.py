import pygame
import random
from collections import namedtuple
from enum import Enum;
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colours
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        # init display
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
                        self.head,
                        Point(self.head.x - BLOCK_SIZE, self.head.y),
                        Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
                    ]
        self.food = None
        self.direction = Direction.RIGHT
        self.score = 0
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, ((self.w - BLOCK_SIZE) // BLOCK_SIZE)) * BLOCK_SIZE
        y = random.randint(0, ((self.h - BLOCK_SIZE) // BLOCK_SIZE)) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _move(self, action):
        # action [straight, right, left]
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)
        next_idx = None
        if np.array_equal(action, np.array([1, 0, 0])):
            next_idx = idx
        elif np.array_equal(action, np.array([0, 1, 0])):
            next_idx = (idx + 1) % 4
        else:
            next_idx = (idx - 1) % 4
        self.direction = clockwise[next_idx]

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        else:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def play_step(self, action):
        print("You are at play setp now")
        self.frame_iteration += 1
        # 1. collect the user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. check if game is over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            print("Collision")
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.food == self.head:
            self.score += 1
            reward = 10
            self._place_food()

        else:
            self.snake.pop()

        # 5. update the ui
        self._update_ui()
        self.clock.tick(SPEED)
        print("Reward:", reward)
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    