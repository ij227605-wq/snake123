import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer
import json
import os
import pygame
import tkinter as tk
from tkinter import messagebox

config = {}
if os.path.exists('settings.json'):
    with open('settings.json', 'r') as f:
        config = json.load(f)

MAX_MEMORY = 100_000  
BATCH_SIZE = config.get("BATCH_SIZE", 1000)
LR = config.get("LR", 0.001)
TARGET_SCORE = config.get("TARGET_SCORE", 10)

class Agent:
    def __init__(self):
        # 💡 [수정] 저장된 판수 불러오기
        self.n_games = config.get("N_GAMES", 0) 
        self.epsilon = 0 
        self.gamma = config.get("GAMMA", 0.9) 
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # 💡 [추가] 시작 시 '불러오기'가 체크되어 있고 파일이 존재한다면 뇌 구조 복구
        if config.get("LOAD_MODEL", False) and os.path.exists('model.pth'):
            self.model.load_state_dict(torch.load('model.pth'))
            print("🧠 기존 AI 모델(model.pth)을 성공적으로 불러왔습니다!")

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or (dir_l and game.is_collision(point_l)) or (dir_u and game.is_collision(point_u)) or (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or (dir_d and game.is_collision(point_l)) or (dir_l and game.is_collision(point_u)) or (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r)) or (dir_u and game.is_collision(point_l)) or (dir_r and game.is_collision(point_u)) or (dir_l and game.is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            game.food.x < game.head.x, game.food.x > game.head.x, game.food.y < game.head.y, game.food.y > game.head.y  
        ]
        return np.array(state, dtype=int) 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
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
        return final_move

def train():
    record = config.get("RECORD", 0) # 💡 저장된 최고 기록 불러오기
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        
        # 💡 [수정] user_quit(강제 종료 여부) 반환받기
        reward, done, score, user_quit = game.play_step(final_move, agent.n_games, record)
        state_new = agent.get_state(game) 
        
        # 강제 종료가 아닐 때만 학습 진행
        if not user_quit:
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)
        
        # 💡 [추가] X버튼을 눌렀거나 목표 점수에 도달했을 때
        if user_quit or score >= TARGET_SCORE:
            if score >= TARGET_SCORE:
                print(f"\n🎉 목표 점수 {TARGET_SCORE}점 달성!")
            
            # 안내 팝업창 띄우기
            root = tk.Tk()
            root.withdraw() # 메인 창 숨기기
            root.attributes('-topmost', True) # 팝업을 맨 앞으로
            answer = messagebox.askyesno("학습 종료", "지금까지의 인공지능 결과(모델)와 현재 기록을 저장하시겠습니까?")
            root.destroy()
            
            if answer:
                agent.model.save('model.pth')
                config["N_GAMES"] = agent.n_games
                config["RECORD"] = record
                with open('settings.json', 'w') as f:
                    json.dump(config, f, indent=4)
                print("💾 성공적으로 저장되었습니다.")
            else:
                print("🗑️ 저장하지 않고 종료합니다.")
            
            pygame.quit()
            break # 프로그램 완전 종료
        
        if done:
            game.reset()
            agent.n_games += 1 
            agent.train_long_memory()
            if score > record:
                record = score
            print(f'Game {agent.n_games} | Score {score} | Record: {record}')

if __name__ == '__main__':
    train()