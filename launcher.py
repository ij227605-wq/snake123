import tkinter as tk
import json
import sys
import subprocess
import os

DEFAULT_SETTINGS = {
    "SPEED": 40,
    "LR": 0.001,
    "GAMMA": 0.9,
    "BATCH_SIZE": 1000,
    "TARGET_SCORE": 10,
    "LOAD_MODEL": False,
    "N_GAMES": 0,
    "RECORD": 0
}

if os.path.exists('settings.json'):
    with open('settings.json', 'r') as f:
        current_settings = json.load(f)
else:
    current_settings = DEFAULT_SETTINGS

def save_and_start():
    settings = {
        "SPEED": int(speed_var.get()),
        "LR": float(lr_var.get()),
        "GAMMA": float(gamma_var.get()),
        "BATCH_SIZE": int(batch_var.get()),
        "TARGET_SCORE": int(target_var.get()),
        "LOAD_MODEL": load_model_var.get(),
        # 불러오기를 체크했다면 기존 판수와 기록을 유지, 아니면 0으로 초기화
        "N_GAMES": current_settings.get("N_GAMES", 0) if load_model_var.get() else 0,
        "RECORD": current_settings.get("RECORD", 0) if load_model_var.get() else 0
    }
    
    with open('settings.json', 'w') as f:
        json.dump(settings, f, indent=4)
    
    # 💡 웹 서버 파일(sever.py) 실행으로 변경!
    subprocess.Popen([sys.executable, "sever.py"])
    root.destroy()

root = tk.Tk()
root.title("AI 강화학습 제어 패널")
root.geometry("320x400")
root.configure(padx=20, pady=20)

speed_var = tk.StringVar(value=str(current_settings.get("SPEED", 40)))
lr_var = tk.StringVar(value=str(current_settings.get("LR", 0.001)))
gamma_var = tk.StringVar(value=str(current_settings.get("GAMMA", 0.9)))
batch_var = tk.StringVar(value=str(current_settings.get("BATCH_SIZE", 1000)))
target_var = tk.StringVar(value=str(current_settings.get("TARGET_SCORE", 10)))
load_model_var = tk.BooleanVar(value=current_settings.get("LOAD_MODEL", False))

def add_field(row, label_text, var):
    tk.Label(root, text=label_text, font=("Arial", 10, "bold")).grid(row=row, column=0, pady=10, sticky="e")
    tk.Entry(root, textvariable=var, width=12, font=("Arial", 10)).grid(row=row, column=1, padx=10, pady=10)

add_field(0, "게임 배속 (SPEED):", speed_var)
add_field(1, "학습률 (LR):", lr_var)
add_field(2, "미래 할인율 (GAMMA):", gamma_var)
add_field(3, "복습량 (BATCH):", batch_var)
add_field(4, "목표 점수 (TARGET):", target_var)

# 💡 [추가] 불러오기 체크박스
tk.Checkbutton(root, text="시작 시 기존 AI 모델 불러오기 (이어서 학습)", 
               variable=load_model_var, font=("Arial", 9, "bold")).grid(row=5, column=0, columnspan=2, pady=15)

start_btn = tk.Button(root, text="저장 및 AI 훈련 시작 ▶", bg="#008000", fg="white", 
                      font=("Arial", 12, "bold"), command=save_and_start)
start_btn.grid(row=6, column=0, columnspan=2, pady=10, ipadx=20, ipady=5)

root.mainloop()