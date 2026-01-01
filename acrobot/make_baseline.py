import gymnasium as gym
import numpy as np
import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# ==========================================
# 설정
# ==========================================
ENV_ID = "Acrobot-v1"
TOTAL_TIMESTEPS = 150_000   # Acrobot Expert 도달에 충분한 시간
SAVE_DIR = "baseline_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# Medium 모델을 저장할 기준 점수 (Acrobot: -500 ~ 0, 대략 -200~-300 사이)
MEDIUM_THRESHOLD = -200 

class DataCollectionCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.steps = []
        self.rewards = []
        self.medium_saved = False
        self.medium_path = os.path.join(SAVE_DIR, "medium_model")

    def _on_step(self) -> bool:
        # 1. 학습 로그 기록 (Step vs Reward)
        for info in self.locals['infos']:
            if 'episode' in info:
                self.steps.append(self.num_timesteps)
                self.rewards.append(info['episode']['r'])

                # 2. Medium 모델 저장 (최근 50 에피소드 평균 기준)
                if not self.medium_saved and len(self.rewards) >= 50:
                    avg_rew = np.mean(self.rewards[-50:])
                    if avg_rew > MEDIUM_THRESHOLD:
                        print(f"\n[Save] Medium Model Reached! Avg: {avg_rew:.1f}")
                        self.model.save(self.medium_path)
                        self.medium_saved = True
        return True

def evaluate_mixed_policy(env_id, medium_model_path, episodes=100):
    """
    Medium 70% + Random 30% 정책의 평균 점수 계산
    """
    print(f"\nEvaluating Mixed Policy (Medium 0.7 + Random 0.3)...")
    model = PPO.load(medium_model_path)
    env = gym.make(env_id)
    
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_rew = 0
        
        while not done:
            # 30% 확률로 랜덤 행동
            if np.random.rand() < 0.3:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, rew, terminated, truncated, _ = env.step(action)
            total_rew += rew
            done = terminated or truncated
        returns.append(total_rew)
    
    mean_score = np.mean(returns)
    print(f"Mixed Policy Average Score: {mean_score:.2f}")
    return mean_score

if __name__ == "__main__":
    # 1. Expert 학습 및 로그 수집
    env = make_vec_env(ENV_ID, n_envs=8)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
    callback = DataCollectionCallback()
    
    print(">>> Start Training Expert PPO...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    env.close()

    # 2. Mixed Policy (Pi_b) 성능 측정
    if callback.medium_saved:
        pi_b_score = evaluate_mixed_policy(ENV_ID, callback.medium_path)
    else:
        print("Warning: Medium threshold not reached. Using last model as Medium.")
        model.save(callback.medium_path)
        pi_b_score = evaluate_mixed_policy(ENV_ID, callback.medium_path)

    # 3. 데이터 통합 저장
    save_path = os.path.join(SAVE_DIR, "comparison_data.npz")
    np.savez(
        save_path,
        expert_steps=np.array(callback.steps),
        expert_rewards=np.array(callback.rewards),
        pi_b_score=pi_b_score
    )
    print(f"\nAll data saved to {save_path}")