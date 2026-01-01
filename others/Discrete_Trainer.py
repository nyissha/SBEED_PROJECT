import argparse
import torch
import torch.optim as optim
import numpy as np
import gymnasium as gym
import os
import matplotlib.pyplot as plt
from Objective import PCL, TRPO

class PCLPolicy(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PCLPolicy, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_dim) # Logits output
        )
    def forward(self, x):
        return self.net(x)
    
class PCLValue(torch.nn.Module):
    def __init__(self, obs_dim): # observation dim
        super(PCLValue, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),  
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
    def forward(self, x): # 실제 입력
        return self.net(x)


class Trainer:
    def __init__(self, args):
        self.args = args

        # 1. 환경설정
        self.env = gym.make(args.env)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n

        # 2. 하이퍼파라미터
        self.tau = args.tau
        self.gamma = args.gamma
        self.rollout = args.rollout
        self.max_steps = args.num_steps

        # 3. 모델 초기화 (Actor Critic)
        #self.policy = PCLPolicy(self.obs_dim, self.act_dim)
        self.policy = PCLPolicy(self.obs_dim, self.act_dim)
        self.value = PCLValue(self.obs_dim)

        #loss fn
        self.trpo = TRPO(self.parameter_list, args.learning_rate)

        # 4. 옵티마이져 Adam에게 조절해야할 파라미터 레버를 전달
        self.parameter_list = list(self.policy.parameters()) + list(self.value.parameters())
        self.optimizer = optim.Adam(
            self.parameter_list,
            lr = args.learning_rate
        )

        # 5. 리플레이 버퍼
        self.replay_buffer = []

        # 6. 시각화 준비
        self.reward_history = []
        self.loss_history = []
    
    def get_pcl_objective(self, states, actions, rewards, next_states, old_log_probs=None): #Objective에서 PCL, TRPO 가져와서 실험중. 사용 x
        #Tensor 변환
        #딥러닝 디버깅의 80%는 모양을 맞추는 일
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)

        #Forward pass nn.Module의 객체를 함수처럼 호출하면 자동으로 forward()매서드 실행
        logits = self.policy(states) #logits은 확률 변환전 자신감의 정도
        current_v = self.value(states)
        next_v = self.value(next_states).detach() #next_v는 학습에 영향끼치면 안된다.

        #Log probabilities
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions).unsqueeze(1)

        # -- Path Consistency Equation --
        soft_reward = rewards - self.tau * log_probs # r - tau*log_pi(a|s)
        consistency_error = current_v - (self.gamma * next_v + soft_reward)
        
        pcl_loss = 0.5 * torch.mean(consistency_error ** 2)

        if self.args.trust_region:
            kl_div = torch.mean(old_log_probs - log_probs)
            pcl_loss += self.args.kl_coeff * kl_div

        return pcl_loss

    def run(self):
        state, _ = self.env.reset()
        total_reward = 0

        for step in range(1, self.max_steps + 1):
            # 1. action sampling
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits = self.policy(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor).item()
                action = action_tensor.item()

            # 2. Env step
            next_state, reward, done, _, _ = self.env.step(action)

            self.replay_buffer.append((state, action, reward, next_state, log_prob))
            state = next_state
            total_reward += reward

            if done:
                state, _ = self.env.reset()
                self.reward_history.append(total_reward)
                print(f"Step: {step}, Total Reward: {total_reward}")
                total_reward = 0
            
            # 3. Training Step
            if len(self.replay_buffer) >= self.args.batch_size:
                # 1. 랜덤 샘플링
                indices = np.random.choice(len(self.replay_buffer), self.args.batch_size)
                batch = [self.replay_buffer[i] for i in indices]
                s, a, r, ns, old_lp = zip(*batch)

                # 2. 텐서 전환
                states = torch.FloatTensor(s)
                actions = torch.LongTensor(a)
                next_states = torch.FloatTensor(ns)
                rewards = torch.FloatTensor(r).unsqueeze(1)
                old_lp = torch.FloatTensor(old_lp)

                # 3. pi와 V 구하기
                value = self.value(states)
                logits = self.policy(states) 
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions).unsqueeze(1)

                # 4. loss/Objective
                loss = self.trpo.calculate_loss(rewards, value, log_probs, old_lp)
                self.loss_history.append(loss.item())

                # 5. Gradient Descent - Optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if not self.args.use_replay_buffer:
                    self.replay_buffer = []

def plot_result(trainer):
    plt.figure(figsize=(12, 5))

    #보상 그래프
    plt.subplot(1, 2, 1)
    plt.plot(trainer.reward_history)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(trainer.loss_history)
    plt.title('Loss')
    plt.xlabel('Training step')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-6, dest='learning_rate')
    parser.add_argument('--tau', type=float, default=0.1, help='Entropy Regularization')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--rollout', type=int, default=10)
    parser.add_argument('--trust_region', action='store_true')
    parser.add_argument('--kl_coeff', type=float, default=0.1)
    parser.add_argument('--use_replay_buffer', action='store_true')

    args = parser.parse_args()
    
    trainer = Trainer(args)
    trainer.run()
    plot_result(trainer)