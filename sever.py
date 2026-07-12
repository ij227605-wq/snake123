import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import json
import asyncio
import threading
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import torch

import agent 
import game 

app = FastAPI()

game_state = {
    "n_games": 0, "score": 0, "record": 0,
    "food": {"x": 0, "y": 0}, "snake": [],
    "config": {}
}

current_game = None
reset_flag = False 

def run_ai_in_background():
    global game_state, current_game, reset_flag
    
    while True: 
        if os.path.exists('settings.json'):
            with open('settings.json', 'r') as f:
                config = json.load(f)
        else:
            config = {"SPEED": 40, "LR": 0.001, "GAMMA": 0.9, "EPSILON": 80, "BATCH_SIZE": 1000, "TARGET_SCORE": 10}
        
        game_state["config"] = config
        
        game.SPEED = int(config.get("SPEED", 40))
        agent.BATCH_SIZE = int(config.get("BATCH_SIZE", 1000))
        agent.LR = float(config.get("LR", 0.001))
        target_score = int(config.get("TARGET_SCORE", 10))
        
        my_agent = agent.Agent()
        
        my_agent.gamma = float(config.get("GAMMA", 0.9))
        if hasattr(my_agent, 'trainer'):
            my_agent.trainer.gamma = my_agent.gamma
            
        if hasattr(my_agent, 'epsilon'):
            my_agent.epsilon = int(config.get("EPSILON", 80))
        
        if config.get("LOAD_MODEL", False) and os.path.exists('model.pth'):
            my_agent.model.load_state_dict(torch.load('model.pth'))
            my_agent.n_games = config.get("N_GAMES", 0)
            record = config.get("RECORD", 0)
            print("🧠 기존 AI 모델 불러오기 완료!")
        else:
            record = 0

        current_game = game.SnakeGameAI()
        current_game.is_paused = True 
        reset_flag = False

        while not reset_flag:
            state_old = my_agent.get_state(current_game)
            final_move = my_agent.get_action(state_old)
            
            reward, done, score, user_quit = current_game.play_step(final_move, my_agent.n_games, record)
            state_new = my_agent.get_state(current_game) 
            
            if not current_game.is_paused:
                my_agent.train_short_memory(state_old, final_move, reward, state_new, done)
                my_agent.remember(state_old, final_move, reward, state_new, done)
            
            game_state["n_games"] = my_agent.n_games
            game_state["score"] = score
            game_state["record"] = record
            game_state["food"] = {"x": current_game.food.x, "y": current_game.food.y}
            game_state["snake"] = [{"x": pt.x, "y": pt.y} for pt in current_game.snake]
            
            if score >= target_score and not current_game.is_paused:
                my_agent.model.save('model.pth')
                config["N_GAMES"] = my_agent.n_games
                config["RECORD"] = record
                with open('settings.json', 'w') as f:
                    json.dump(config, f, indent=4)
                current_game.is_paused = True 
                print(f"🎉 목표 점수 {target_score} 달성! 모델 저장 완료.")

            if done:
                current_game.reset()
                my_agent.n_games += 1 
                my_agent.train_long_memory()
                if score > record:
                    record = score

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=run_ai_in_background, daemon=True).start()

@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async def listen_to_web_buttons():
        global current_game, reset_flag
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                
                if "command" in msg:
                    if msg["command"] == "start" and current_game:
                        current_game.is_paused = False
                    elif msg["command"] == "stop" and current_game:
                        current_game.is_paused = True
                    elif msg["command"] == "update_settings":
                        with open('settings.json', 'w') as f:
                            json.dump(msg["settings"], f, indent=4)
                        reset_flag = True 
                        if current_game:
                            current_game.is_paused = False 
        except Exception:
            pass

    asyncio.create_task(listen_to_web_buttons())

    try:
        while True:
            await websocket.send_text(json.dumps(game_state))
            await asyncio.sleep(0.03) 
    except Exception:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)