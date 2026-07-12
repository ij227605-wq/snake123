import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import json
import os

# 💡 설정 파일 불러오기
config = {}
if os.path.exists('settings.json'):
    with open('settings.json', 'r') as f:
        config = json.load(f)

# Pygame 초기화
pygame.init()
font = pygame.font.Font(pygame.font.get_default_font(), 25)
btn_font = pygame.font.Font(pygame.font.get_default_font(), 18)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 180, 0)
DARK_RED = (180, 0, 0)

BLOCK_SIZE = 20
SPEED = config.get("SPEED", 40)

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake 강화학습 (보상 셰이핑 + 제어 + 저장)')
        self.clock = pygame.time.Clock()
        
        # 시작/중지 버튼
        self.start_btn = pygame.Rect(450, 10, 80, 35)
        self.stop_btn = pygame.Rect(540, 10, 80, 35)
        self.is_paused = False 

        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, Point(self.head.x-BLOCK_SIZE, self.head.y), Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action, n_games=0, record=0):
        # 1. 정상 속도일 때 이벤트 감지
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # 💡 강제 종료 신호(True) 반환
                return 0, True, self.score, True 
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.stop_btn.collidepoint(event.pos):  
                    self.is_paused = True
                elif self.start_btn.collidepoint(event.pos): 
                    self.is_paused = False
        
        # 2. 일시 정지 루프
        while self.is_paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0, True, self.score, True 
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.start_btn.collidepoint(event.pos):
                        self.is_paused = False 
            self._update_ui(n_games, record)
            self.clock.tick(SPEED) 

        self.frame_iteration += 1
        old_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        
        # 💡 [바로 이 부분!] is_collision이 무사히 살아있습니다.
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score, False

        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        if self.head == self.food:
            self.score += 1
            reward = 10 
            self._place_food()
        else:
            self.snake.pop()
            if new_distance < old_distance:
                reward = 0.5
            else:
                reward = -0.5 
        
        self._update_ui(n_games, record)
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score, False

    # 💡 [핵심 부활] 아까 지워졌던 충돌 감지 함수입니다.
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self, n_games=0, record=0):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text_score = font.render(f"Score: {self.score}", True, WHITE)
        text_games = font.render(f"Game: {n_games}", True, WHITE)
        text_record = font.render(f"Record: {record}", True, WHITE)
        
        self.display.blit(text_score, [0, 0])
        self.display.blit(text_games, [0, 30])
        self.display.blit(text_record, [0, 60])

        # 버튼 그리기
        pygame.draw.rect(self.display, GREEN, self.start_btn)
        start_text = btn_font.render("START", True, WHITE)
        self.display.blit(start_text, (460, 16))

        pygame.draw.rect(self.display, DARK_RED, self.stop_btn)
        stop_text = btn_font.render("STOP", True, WHITE)
        self.display.blit(stop_text, (555, 16))
        
        if self.is_paused:
            pause_text = font.render("PAUSED", True, WHITE)
            self.display.blit(pause_text, (self.w/2 - 50, self.h/2 - 20))
        
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] 
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else: 
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE

        self.head = Point(x, y)