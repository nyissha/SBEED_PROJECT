import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# ==========================================
# 1. 설정 및 하이퍼파라미터
# ==========================================
ENV_ID = "Acrobot-v1"
Total_Timesteps = 300_000   # 충분한 학습 시간
Num_Envs = 8                # 병렬 환경 개수 (속도 향상 핵심)
Learning_Rate = 3e-4
Save_Dir = "weights"
os.makedirs(Save_Dir, exist_ok=True)

# ==========================================
# 2. 모델 네트워크 정의 (Actor 구조)
# ==========================================
class QNetwork(nn.Module):
    """
    SB3의 MlpPolicy 구조와 동일하게 맞춘 PyTorch 네트워크
    나중에 .pt 파일을 로드해서 사용할 때 필요함
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # SB3 PPO 기본 구조: 64x64 또는 설정에 따라 다름.
        # 여기서는 [128, 128]로 맞출 예정
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)

def save_ppo_actor_as_pt(sb3_model, save_path, obs_dim, act_dim):
    """Stable Baselines3 모델에서 가중치를 추출하여 순수 PyTorch 모델로 저장"""
    custom_model = QNetwork(obs_dim, act_dim)
    
    with torch.no_grad():
        # SB3 내부 변수명에 맞춰 가중치 복사 (net_arch=[128, 128] 기준)
        # Layer 1
        custom_model.net[0].weight.data = sb3_model.policy.mlp_extractor.policy_net[0].weight.data.clone()
        custom_model.net[0].bias.data = sb3_model.policy.mlp_extractor.policy_net[0].bias.data.clone()
        # Layer 2
        custom_model.net[2].weight.data = sb3_model.policy.mlp_extractor.policy_net[2].weight.data.clone()
        custom_model.net[2].bias.data = sb3_model.policy.mlp_extractor.policy_net[2].bias.data.clone()
        # Output Layer (Action Net)
        custom_model.net[4].weight.data = sb3_model.policy.action_net.weight.data.clone()
        custom_model.net[4].bias.data = sb3_model.policy.action_net.bias.data.clone()

    torch.save(custom_model.state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")

# ==========================================
# 3. 콜백 (학습 중 저장 및 로그 기록)
# ==========================================
class CheckpointCallback(BaseCallback):
    def __init__(self, env_id, obs_dim, act_dim, verbose=1):
        super().__init__(verbose)
        self.env_id = env_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.medium_saved = False
        self.expert_saved = False
        self.reward_history = [] # 그래프 그리기용 데이터

    def _on_step(self) -> bool:
        # 벡터 환경에서는 infos에 에피소드 종료 정보가 들어옴
        for info in self.locals['infos']:
            if 'episode' in info:
                ep_rew = info['episode']['r']
                self.reward_history.append(ep_rew)
                
                # 최근 50개 에피소드 평균 보상 계산
                if len(self.reward_history) >= 50:
                    mean_reward = np.mean(self.reward_history[-50:])
                    
                    # Medium 저장 (-200점 돌파 시)
                    if not self.medium_saved and mean_reward > -200:
                        print(f"\n🚀 Medium Reached! (Avg: {mean_reward:.1f})")
                        save_path = f"{Save_Dir}/{self.env_id}_medium.pt"
                        save_ppo_actor_as_pt(self.model, save_path, self.obs_dim, self.act_dim)
                        self.medium_saved = True

                    # Expert 저장 (-90점 돌파 시)
                    if not self.expert_saved and mean_reward > -90:
                        print(f"\n🏆 Expert Reached! (Avg: {mean_reward:.1f})")
                        save_path = f"{Save_Dir}/{self.env_id}_expert.pt"
                        save_ppo_actor_as_pt(self.model, save_path, self.obs_dim, self.act_dim)
                        self.expert_saved = True
                        # Expert 달성 시 조기 종료를 원하면 아래 주석 해제
                        # return False 

        return True

# ==========================================
# 4. 결과 시각화 및 관전 함수
# ==========================================
def plot_learning_curve(rewards):
    """학습 보상 그래프 그리기"""
    plt.figure(figsize=(10, 5))
    plt.title(f"{ENV_ID} Training Reward Curve")
    plt.plot(rewards, alpha=0.3, color='gray', label='Raw')
    
    # 이동 평균
    window = 50
    if len(rewards) >= window:
        avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(avg, color='red', label=f'Moving Avg ({window})')
    
    plt.axhline(-100, color='green', linestyle='--', label='Expert (-100)')
    plt.axhline(-200, color='blue', linestyle='--', label='Medium (-200)')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    print("\n📊 그래프 창을 닫으면 시뮬레이션이 시작됩니다.")
    plt.show()

def watch_agent(model_path, obs_dim, act_dim):
    """저장된 .pt 모델을 로드하여 화면에 렌더링"""
    if not os.path.exists(model_path):
        print(f"⚠️ {model_path} 파일이 없어 관전을 건너뜁니다.")
        return

    print(f"\n🎬 Watching Agent: {model_path}")
    env = gym.make(ENV_ID, render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    model = QNetwork(obs_dim, act_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3판 플레이
    for ep in range(3):
        obs, _ = env.reset()
        done = False
        total_rew = 0
        while not done:
            # PPO Actor Inference
            state_t = torch.tensor(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(state_t)
                action = torch.argmax(logits, dim=1).item()
            
            obs, rew, terminated, truncated, _ = env.step(action)
            total_rew += rew
            done = terminated or truncated
        print(f"Episode {ep+1} Score: {total_rew:.1f}")
    
    env.close()

# ==========================================
# 5. 메인 실행 블록
# ==========================================
if __name__ == "__main__":
    # A. 환경 생성 (병렬 처리)
    # n_envs=8: 8배 빠른 데이터 수집
    env = make_vec_env(ENV_ID, n_envs=Num_Envs)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print(f"System: {ENV_ID} | Obs: {obs_dim} | Act: {act_dim}")

    # B. PPO 모델 설정
    # net_arch=[128, 128] : QNetwork 클래스와 구조를 맞추기 위함
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=Learning_Rate,
        n_steps=1024,
        batch_size=64,
        ent_coef=0.0, # 병렬 환경이라 랜덤성 충분함
        tensorboard_log=f"./{ENV_ID}_tb_log/"
    )

    # C. 학습 시작
    callback = CheckpointCallback(ENV_ID, obs_dim, act_dim)
    print("🚀 Training Started...")
    model.learn(total_timesteps=Total_Timesteps, callback=callback)
    print("✅ Training Finished.")

    # 만약 Expert 저장이 안 됐으면 마지막 모델이라도 저장
    final_path = f"{Save_Dir}/{ENV_ID}_expert.pt"
    if not callback.expert_saved:
        print("⚠️ Expert 기준 미달, 마지막 모델을 저장합니다.")
        save_ppo_actor_as_pt(model, final_path, obs_dim, act_dim)

    env.close()

    # D. 시각화 (그래프)
    if callback.reward_history:
        plot_learning_curve(callback.reward_history)

    # E. 관전 (Expert 모델)
    watch_agent(final_path, obs_dim, act_dim)