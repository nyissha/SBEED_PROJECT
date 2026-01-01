import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #GPU 점유를 0번 GPU로 한정, torch보다 먼저 작성되어야 한다.
import numpy as np
import gymnasium as gym
import torch # import되며 C/C++ 백엔드 수준에서 하드웨어 토톨로지 스캔
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import matplotlib.pyplot as plt
import time
import random
from torch.distributions import Categorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #MPS로 바꿀 필요

def set_seed(seed): # 재현성을 위한 시드 설정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True #CuDNN의 무작위성을 제어
        torch.backends.cudnn.benchmark = False

# ===================== 1. One-step Offline Buffer =====================
class OfflineReplayBuffer:
    def __init__(self, obs, act, rew, obs2, terminated, truncated):
        self.obs  = torch.tensor(obs,  dtype=torch.float32, device=DEVICE)
        self.act  = torch.tensor(act,  dtype=torch.int64,   device=DEVICE)
        self.rew  = torch.tensor(rew,  dtype=torch.float32, device=DEVICE)
        self.obs2 = torch.tensor(obs2, dtype=torch.float32, device=DEVICE)
        self.done = (torch.tensor(terminated) + torch.tensor(truncated)).clamp(max=1.0).to(DEVICE)
        self.N = len(obs)
        print(f"[Buffer] Loaded 1-step transitions: {self.N}")

    def sample(self, batch_size):
        idx = np.random.choice(self.N, batch_size, replace=batch_size > self.N) #복원 추출
        return {
            "obs":  self.obs[idx],
            "act":  self.act[idx],
            "rew":  self.rew[idx],
            "obs2": self.obs2[idx],
            "done": self.done[idx],
        }

# ===================== 2. Networks =====================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers) #unpacking
    def forward(self, x):
        return self.net(x)

class DiscretePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, temperature=2.0, logit_clip=10.0):
        super().__init__()
        self.backbone = MLP(obs_dim, act_dim)
        self.temperature = temperature
        self.logit_clip = logit_clip

    def logits(self, obs):
        z = self.backbone(obs)
        z = torch.clamp(z, -self.logit_clip, self.logit_clip)
        return z / self.temperature

    # softmax distribution
    def dist(self, obs):
        return Categorical(logits=self.logits(obs))

    def log_prob(self, obs, act):
        return self.dist(obs).log_prob(act)

    def act_greedy(self, obs):
        return torch.argmax(self.logits(obs), dim=-1)

class RhoNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, 1)
    def forward(self, obs, act):
        a_onehot = F.one_hot(act, num_classes=4).float() #one-hot vector : 표현하고 싶은 클래스의 인덱스만 1, 나머지 0
        return self.net(torch.cat([obs, a_onehot], dim=-1)).squeeze(-1)

def kl_categorical(p: Categorical, q: Categorical):
    return torch.distributions.kl.kl_divergence(
        Categorical(logits=p.logits.detach()),
        Categorical(logits=q.logits.detach())
    )

# ===================== 3. SBEED Agent =====================
class OfflineSBEED:
    def __init__(self, obs_dim=8, act_dim=4, lr=1e-4, eta=0.05, lam=0.01, gamma=0.99, kl_beta=0.1, grad_clip=10.0):
        self.gamma = gamma
        self.eta = eta
        self.lam = lam
        self.kl_beta = kl_beta
        self.grad_clip = grad_clip

        self.pi  = DiscretePolicy(obs_dim, act_dim).to(DEVICE)
        self.v   = MLP(obs_dim, 1).to(DEVICE)
        self.rho = RhoNet(obs_dim, act_dim).to(DEVICE)

        self.opt_pi  = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.opt_v   = torch.optim.Adam(self.v.parameters(),  lr=lr)
        self.opt_rho = torch.optim.Adam(self.rho.parameters(),lr=lr)

    def update(self, b):
        obs, act, rew, obs2, done = b["obs"], b["act"], b["rew"], b["obs2"], b["done"]

        # old policy snapshot for KL prox
        with torch.no_grad():
            dist_old = self.pi.dist(obs)
            logp_old = self.pi.log_prob(obs, act)

        # target δ
        v_next = self.v(obs2).squeeze(-1)
        delta = rew - self.lam * logp_old + self.gamma * (1.0 - done) * v_next

        # ρ(s,a) regression
        rho_pred = self.rho(obs, act)
        loss_rho = F.mse_loss(rho_pred, delta.detach())
        self.opt_rho.zero_grad()
        loss_rho.backward()
        utils.clip_grad_norm_(self.rho.parameters(), self.grad_clip)
        self.opt_rho.step()

        # V update
        v0_now = self.v(obs).squeeze(-1)
        mse_td = ((delta.detach() - v0_now)**2).mean()
        mse_dual = ((delta.detach() - rho_pred.detach())**2).mean()

        loss_v = mse_td - self.eta * mse_dual
        self.opt_v.zero_grad()
        loss_v.backward()
        utils.clip_grad_norm_(self.v.parameters(), self.grad_clip)
        self.opt_v.step()

        # policy update
        dist_new = self.pi.dist(obs)
        logp_new = dist_new.log_prob(act)

        with torch.no_grad():
            # policy gradient
            adv = (1.0 - self.eta) * delta.detach() + self.eta * rho_pred.detach() - v0_now.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pg_loss = -2.0 * (self.lam * adv * logp_new).mean()
        
        kl = kl_categorical(dist_new, dist_old).mean()
        loss_pi = pg_loss + (1.0 / self.kl_beta) * kl

        self.opt_pi.zero_grad()
        loss_pi.backward()
        utils.clip_grad_norm_(self.pi.parameters(), self.grad_clip)
        self.opt_pi.step()

        return {"loss_pi": loss_pi.item(), "loss_v": loss_v.item(), "loss_rho": loss_rho.item(), "entropy": dist_new.entropy().mean().item()}

# ===================== 4. Evaluation =====================
@torch.no_grad()
def evaluate_policy(env, policy, episodes=20, seed=1):
    policy.eval()
    rets = []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        done = False
        r_sum = 0.0
        while not done:
            a = policy.act_greedy(torch.tensor(o, dtype=torch.float32, device=DEVICE).unsqueeze(0)).item()
            o, r, term, trunc, _ = env.step(a)
            done = term or trunc
            r_sum += r
        rets.append(r_sum)
    policy.train()
    return float(np.mean(rets)), float(np.std(rets))

# ===================== 5. Run =====================
def run(seed, updates=100_000, eval_freq=2_000):
    set_seed(seed)
    data = np.load("D_lunarlander_mixed_1m.npz")
    buffer = OfflineReplayBuffer(data["obs"], data["act"], data["rew"], data["obs2"], data["terminated"], data["truncated"])
    env = gym.make("LunarLander-v3")
    agent = OfflineSBEED()

    steps, rets = [], []
    start = time.time()

    for i in range(1, updates+1):
        batch = buffer.sample(256)
        agent.update(batch)
        if i % eval_freq == 0:
            m,s = evaluate_policy(env, agent.pi, 20, seed=seed*10_000)
            steps.append(i)
            rets.append(m)
            print(f"[Seed {seed}] Update {i} | Return {m:.2f} ± {s:.2f} | Entropy {agent.pi.dist(batch['obs']).entropy().mean().item():.3f} | Time {(time.time()-start)/60:.1f}m")
    env.close()
    return steps, rets

# ============ Aggregate 5 seeds ============
plt.figure()
all = []
for s in [1,2,3,4,5]:
    st,r = run(s)
    all.append(r)
    plt.plot(st,r, alpha=0.3)
mean = np.mean(all,0)
std  = np.std(all,0)
plt.plot(st,mean, linewidth=2, color='red')
plt.fill_between(st,mean-std,mean+std, alpha=0.2)
plt.title("SBEED (lunarlander perfect mixed 1m)")
plt.xlabel("Gradient Updates")
plt.ylabel("Average Episode Reward")
plt.savefig("sbeed_mixed_1m.png")
plt.show()