import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# 1. 피드포워드 인공신경망 (Feed-Forward Neural Network)
class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 입력층(11개 센서값) -> 은닉층(256개 뉴런) 연결
        self.linear1 = nn.Linear(input_size, hidden_size)
        # 은닉층(256개 뉴런) -> 출력층(3개 행동의 기대 점수) 연결
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # ReLU 활성화 함수 적용: 0보다 작은 값은 0으로, 큰 값은 그대로 통과시켜 비선형성을 부여함
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x # 3가지 행동(직진/우/좌) 각각에 대한 Q-Value(예상 점수) 반환

    def save(self, file_name='model.pth'):
        # 가장 높은 점수를 기록했을 때, 신경망의 가중치(기억)를 파일로 보존
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# 2. 신경망 훈련기 (벨만 방정식 적용)
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr       # Learning Rate: 한 번에 얼마나 크게 가중치를 변경할지 결정
        self.gamma = gamma # Discount Factor: 미래의 보상을 현재 가치로 얼마나 할인할지 결정
        self.model = model
        # Adam 옵티마이저: 인공신경망의 가중치를 빠르고 효율적으로 최적화하는 알고리즘
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # 손실 함수(MSE): (인공지능의 예측값 - 실제 정답)의 제곱 평균 오차를 구함
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # 파이썬 배열을 파이토치 신경망이 계산할 수 있는 '텐서(Tensor)'로 변환
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1: # 데이터가 1차원(1개)일 경우 2차원(배치)으로 형태 변환
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. 예측(Prediction): 현재 상태에서 AI가 생각한 각 행동의 가치
        pred = self.model(state)

        # 2. 정답(Target): 실제로 행동을 취하고 환경에서 얻은 결과를 바탕으로 계산한 가치
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # 핵심 이론 [벨만 방정식]: (현재 얻은 보상) + (미래에 얻을 수 있는 최대 보상 * 할인율)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            # AI가 실제로 선택했던 행동의 Q-Value만 정답(Q_new)으로 덮어씌움
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 3. 역전파(Backpropagation): 예측값과 정답 간의 오차(Loss)를 계산하고, 오차를 줄이도록 뇌세포(가중치)를 갱신
        self.optimizer.zero_grad() # 기울기 초기화
        loss = self.criterion(target, pred) # 오차 계산
        loss.backward() # 역전파 (오차가 발생한 원인을 거슬러 올라감)
        self.optimizer.step() # 가중치 수정