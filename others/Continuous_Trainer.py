import argparse
import torch
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from Objective import SBEED

# [개선 1] Tanh Squashing을 적용한 Policy (SAC 스타일)
# SBEED/SAC 같은 엔트로피 기반 알고리즘은 Clamp 대신 Tanh 변환이 필수적입니다.
class ContinuousPolicy(torch.nn.Module): 
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256), # 레이어 추가로 표현력 증대
            torch.nn.ReLU()
        )
        self.mu = torch.nn.Linear(256, act_dim)
        self.log_std = torch.nn.Linear(256, act_dim) 

    def forward(self, x):
        x = self.net(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        # 표준편차 범위를 제한 (학습 안정성)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, x):
        mu, std = self(x)
        dist = torch.distributions.Normal(mu, std)
        # Reparameterization Trick (미분 가능하게 샘플링)
        u = dist.rsample() 
        action = torch.tanh(u) # (-1, 1) 범위로 압축
        
        # Tanh 변환에 따른 log_prob 보정 (Change of Variable formula)
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Pendulum의 범위인 -2 ~ 2 로 확장
        return action * 2.0, log_prob

class ContinuousValue(torch.nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.env = gym.make(args.env)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.gamma = args.gamma
        self.max_steps = args.num_steps
        
        # 모델 초기화
        self.policy = ContinuousPolicy(self.obs_dim, self.act_dim)
        self.value = ContinuousValue(self.obs_dim)
        
        # Target Network
        self.target_value = ContinuousValue(self.obs_dim)
        self.target_value.load_state_dict(self.value.state_dict())
        self.polyak = 0.995

        # SBEED Optimizer
        self.sbeed = SBEED(self.obs_dim, self.act_dim, args.sbeed_learning_rate, args.tau, args.gamma)

        # Policy/Value Optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=args.learning_rate
        )

        self.replay_buffer = [] # list 기반 버퍼 (간단 구현)
        self.max_buffer_size = 100000 # 버퍼 용량 제한
        
        self.reward_history = []
        self.loss_history = []

    def soft_update(self):
        with torch.no_grad():
            for param, target_param in zip(self.value.parameters(), self.target_value.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1 - self.polyak) * param.data)

    def run(self):
        state, _ = self.env.reset()
        total_reward = 0
        warm_up_steps = 5000

        for step in range(1, self.max_steps + 1):
            # 1. Action Sampling
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            if step < warm_up_steps:
                action_np = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    # Tanh Policy 사용
                    action, _ = self.policy.sample(state_tensor)
                    action_np = action.cpu().numpy().flatten()

            # 2. Env Step
            next_state, reward, done, truncated, _ = self.env.step(action_np)
            scaled_reward = reward * 0.1 # 학습용 스케일링
            
            # 버퍼 저장 (항상 저장)
            # 버퍼 크기 관리 (오래된 데이터 삭제)
            if len(self.replay_buffer) >= self.max_buffer_size:
                self.replay_buffer.pop(0)
            self.replay_buffer.append((state, action_np, scaled_reward, next_state))
            
            state = next_state
            total_reward += reward
            
            if done or truncated:
                state, _ = self.env.reset()
                self.reward_history.append(total_reward)
                # 로그 출력 (자주 확인)
                if len(self.reward_history) % 10 == 0:
                    avg_reward = np.mean(self.reward_history[-10:])
                    print(f"Step: {step}, Avg Reward (Last 10): {avg_reward:.1f}")
                total_reward = 0

            # 3. Training Step
            # [수정] 버퍼 초기화 로직 제거됨. 이제 매 스텝 학습합니다.
            if len(self.replay_buffer) >= self.args.batch_size and step > warm_up_steps:
                indices = np.random.choice(len(self.replay_buffer), self.args.batch_size)
                batch = [self.replay_buffer[i] for i in indices]
                s, a, r, ns = zip(*batch) # log_prob은 현재 정책에서 다시 계산하므로 버퍼에서 뺌

                states = torch.FloatTensor(np.array(s))
                actions = torch.FloatTensor(np.array(a)).view(-1, self.act_dim)
                next_states = torch.FloatTensor(np.array(ns))
                rewards = torch.FloatTensor(np.array(r)).unsqueeze(1)

                # 현재 Policy로 log_prob 재계산 (Off-policy의 핵심)
                # 버퍼에 있는 행동(a)에 대한 log_prob을 구하는 것이 아니라
                # SBEED 수식에 따라 s에서의 새로운 a'를 뽑거나(Reparam), 
                # 혹은 버퍼의 a를 평가할 때 Tanh 보정을 적용해야 함.
                # 여기서는 SBEED의 일반적 구현(Bellman Error 최소화)을 위해 버퍼의 action을 평가합니다.
                
                # [주의] 버퍼의 행동을 평가할 때는 Tanh 역변환 등이 복잡하므로,
                # 간략화를 위해 버퍼 행동에 대한 Log Prob 대신,
                # 현재 상태 s에서 "새로 샘플링한 행동"을 사용하여 Loss를 계산하는 것이 더 안정적일 수 있습니다.
                # 하지만 Standard SBEED는 (s, a_buffer) 쌍을 평가합니다.
                
                # Tanh Policy에서 버퍼의 action에 대한 log_prob 구하기
                # (action은 이미 2.0이 곱해져 있으므로 나눠줌)
                mu, std = self.policy(states)
                dist = torch.distributions.Normal(mu, std)
                
                # action 복원 (역함수): arctanh(a / 2.0)
                # clipping to avoid NaN
                action_norm = torch.clamp(actions / 2.0, -0.9999, 0.9999)
                u = torch.atanh(action_norm)
                log_probs = dist.log_prob(u) - torch.log(1 - action_norm.pow(2) + 1e-6)
                log_probs = log_probs.sum(dim=-1, keepdim=True)

                current_v = self.value(states)
                next_v = self.target_value(next_states).detach()

                loss = self.sbeed.calculate_loss(states, actions, rewards, next_states, current_v, next_v, log_probs)
                self.loss_history.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.soft_update()

# Plot 함수는 기존과 동일하게 유지...
                    
def plot_result(trainer):
    plt.figure(figsize=(15, 6))

    # 1. 보상 그래프 (Original Scale + Moving Average)
    plt.subplot(1, 2, 1)
    # 학습 시 0.1을 곱했으므로 시각화를 위해 다시 10을 곱함
    rewards = np.array(trainer.reward_history) * 10 
    plt.plot(rewards, alpha=0.3, color='gray', label='Raw Episode Reward')
    
    # 이동 평균 계산 (최근 20개 에피소드)
    if len(rewards) >= 20:
        ma_rewards = np.convolve(rewards, np.ones(20)/20, mode='valid')
        plt.plot(np.arange(19, len(rewards)), ma_rewards, color='red', label='Moving Average (20)')
    
    plt.title('Total Reward per Episode (Original Scale)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Loss 그래프 (Stability 점검)
    plt.subplot(1, 2, 2)
    losses = np.array(trainer.loss_history)
    # 너무 튀는 값은 제외하고 그림 (하위 95% 값만 출력)
    if len(losses) > 0:
        plt.plot(losses, color='blue', alpha=0.6)
        # y축 범위를 제한하여 변화가 잘 보이게 함
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        plt.ylim(mean_loss - 2*std_loss, mean_loss + 2*std_loss)

    plt.title('SBEED Primal Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v1')
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate')
    parser.add_argument('--slr', type=float, default=1e-3, dest='sbeed_learning_rate')
    parser.add_argument('--tau', type=float, default=0.01, help='Entropy Regularization')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--rollout', type=int, default=10)
    parser.add_argument('--trust_region', action='store_true')
    parser.add_argument('--kl_coeff', type=float, default=0.01)
    parser.add_argument('--use_replay_buffer', action='store_true')

    args = parser.parse_args()
    
    trainer = Trainer(args)
    trainer.run()
    plot_result(trainer)