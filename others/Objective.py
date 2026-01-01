import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Objective:
    def __init__(self, parameters, learning_rate, clip_norm=None):
        self.optimizer = optim.Adam(parameters, lr=learning_rate, eps=2e-4)
        self.clip_norm = clip_norm
        self.parameters = list(parameters)

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        if self.clip_norm:
            nn.utils.clip_grad_norm_(self.parameters, self.clip_norm)
        
        self.optimizer.step()
        
def discounted_future_sum(values, discount, rollout):
    seq_len, batch_size = values.shape
    returns = torch.zeros_like(values)
    running_add = 0

    for t in reversed(range(seq_len)):
        running_add = running_add * discount + values[t]
        returns[t] = running_add

        if t + rollout < seq_len:
            pass
    #[batch, 1, seq] 형태
    #loop를 돌리지 않고 conv로 한 번에 찍어낸다.
    x = values.permute(1, 0).unsqueeze(1)

    kernel = (discount ** torch.arange(float(rollout))).view(1, 1, -1).to(values.device)
    pad_x = F.pad(x, (0, int(rollout)-1))

    conv_res = F.conv1d(pad_x, kernel)
    return conv_res.squeeze(1).permute(1, 0)

class ActorCritic(Objective):
    def __init__(self, parameters, learning_rate, clip_norm=5, policy_weight = 1.0, critic_weight=0.1, tau=0.1, gamma=1.0, rollout=10):
        super().__init__(parameters, learning_rate, clip_norm)
        self.policy_weight = policy_weight
        self.critic_weight = critic_weight
        self.tau = tau
        self.gamma = gamma
        self.rollout = rollout

    def calculate_loss(self, rewards, values, log_probs, entropies, next_val=0.0):
        # 1. Returns 계산
        returns = discounted_future_sum(values, self.gamma, self.rollout)
        bootstrap_values = torch.cat([values[self.rollout:], torch.zeros_like(values[:self.rollout])])
        future_values = returns + (self.gamma ** self.rollout) * bootstrap_values

        # 2. Advantage & Loss        
        baseline = values
        adv = (future_values - baseline).detach() # Critic Target 고정
        
        policy_loss = -adv * log_probs
        critic_loss = -adv * baseline # TF 코드 특이점: Critic update를 adv * V로 함 (일반적인 MSE와 다름)
        # 보통은 critic_loss = F.mse_loss(values, future_values.detach())
        
        entropy_loss = -self.tau * entropies
        
        total_loss = (self.policy_weight * policy_loss.mean() + 
                      self.critic_weight * critic_loss.mean() + 
                      entropy_loss.mean())
        
        return total_loss

class PCL(ActorCritic):
    """
    Path Consistency Learning (PCL) Implementation
    List A: PCL의 핵심인 'Soft Consistency' 수식 구현
    """
    def calculate_loss(self, rewards, values, log_probs, target_log_probs=None):
        # 1. Soft Reward 생성 (r - tau * log_pi)
        # 엔트로피가 보상의 일부로 취급됨 (Softmax Policy 유도)
        soft_rewards = rewards - self.tau * log_probs
        
        if target_log_probs is not None:
             #soft reward에 KL divergence까지
             # Trust-PCL / Unified PCL의 경우 relative entropy 항 추가
             soft_rewards -= self.eps_lambda * (log_probs - target_log_probs)

        # 2. k-step Consistency Check
        # 식: V(s_t) = sum(soft_rewards) + gamma^k * V(s_{t+k})
        sum_soft_rewards = discounted_future_sum(soft_rewards, self.gamma, self.rollout)
        
        # Bootstrap Value (k-step 후의 가치)
        # PyTorch slicing으로 TF의 shift_values 구현
        # V_{t+k} 가져오기. 길이가 안 맞는 부분은 0 처리 (Terminal 가정)
        bootstrap_vals = torch.zeros_like(values)
        T = values.shape[0] #몇 에피소드인가
        valid_len = T - self.rollout
        if valid_len > 0:
            bootstrap_vals[:valid_len] = values[self.rollout:].detach()
            
        rhs = sum_soft_rewards + (self.gamma ** self.rollout) * bootstrap_vals
        lhs = values
        
        # 3. Consistency Error (Objective)
        # PCL은 이 차이(adv)를 줄이는 것이 목표
        # LHS(현재 가치)와 RHS(미래 예측치)의 차이
        adv = (rhs - lhs).detach() 
        
        # PCL Loss: (V - V_target)^2 형태가 아니라, 
        # Policy와 Value가 일관성 에러(adv)를 최소화하도록 공통 Loss 사용
        # TF 코드: policy_loss = -adv * sum_log_probs (다소 복잡한 유도 과정 결과)
        # 직관적 구현: Consistency Error 자체를 줄임
        consistency_error = lhs - rhs # == -adv
        loss = 0.5 * (consistency_error ** 2).mean()
        
        return loss

class TRPO(ActorCritic):
    """
    TF 코드의 TRPO는 Conjugate Gradient 없는 간소화 버전 (PPO style ratio objective)
    """
    def calculate_loss(self, rewards, values, log_probs, prev_log_probs):
        # 1. Advantage 계산 (ActorCritic과 동일)
        returns = discounted_future_sum(rewards, self.gamma, self.rollout)
        # ... (Bootstrapping 로직 동일) ...
        future_values = returns # + bootstrap
        adv = (future_values - values).detach()
        
        # 2. Importance Sampling Ratio
        # exp(log_pi_new - log_pi_old)
        ratio = torch.exp(log_probs - prev_log_probs)
        
        # 3. Surrogate Loss
        policy_loss = -adv * ratio
        critic_loss = (values - future_values).pow(2) # 일반적인 MSE
        
        total_loss = (self.policy_weight * policy_loss.mean() + 
                      self.critic_weight * critic_loss.mean())
        
        return total_loss   
    


class SBEED:
    def __init__(self, obs_dim, act_dim, lr, tau, gamma):
        self.tau = tau
        self.gamma = gamma
        
        # f-net 안정화: 출력이 너무 커지지 않게 조정
        self.f_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 512), # 더 넓게
            nn.LayerNorm(512),               # 안정화
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.f_optimizer = torch.optim.Adam(self.f_net.parameters(), lr=lr)

    def calculate_loss(self, states, actions, rewards, next_states, current_v, next_v, log_probs):
        f_input = torch.cat([states, actions], dim=-1)
        
        # 1. f-net 업데이트 (Dual Step)
        for _ in range(5):
            f_val = self.f_net(f_input)
            with torch.no_grad():
                # delta = (Target - Current)
                # 오차가 양수면(Target이 더 크면) f도 양수가 되도록 학습
                delta = rewards + self.gamma * next_v - (current_v + self.tau * log_probs)
            
            # 목적: f가 delta와 같아지도록 학습
            loss_f = -(f_val * delta - 0.5 * f_val**2).mean()
            
            self.f_optimizer.zero_grad()
            loss_f.backward()
            torch.nn.utils.clip_grad_norm_(self.f_net.parameters(), 0.5)
            self.f_optimizer.step()

        # 2. Primal Loss (V와 pi 업데이트용)
        f_val_final = self.f_net(f_input).detach()
      
        delta_with_grad = (rewards + self.gamma * next_v) - (current_v + self.tau * log_probs)
     
        loss_primal = (f_val_final * delta_with_grad).mean() 
        
        return loss_primal