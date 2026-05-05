
#!/usr/bin/env python3
"""
human_balance_complete_stable.py

"""

import os
import random
import math
from copy import deepcopy
from pathlib import Path
import json
import time
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cosine
from scipy.spatial import ConvexHull
from scipy.signal import welch, spectrogram
from scipy.stats import entropy as scipy_entropy
from fastdtw import fastdtw
import gymnasium as gym
from gymnasium import spaces
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ====================== STABILIZED CONFIGURATION ======================
CSV_PATH = "all_excel_measurements.csv"
GROUP_COL = "fn_index"
TIME_COL = "n"
X_COL = "X"
Y_COL = "Y"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Environment parameters
DT = 0.05
MAX_THETA = np.radians(12)
MAX_PHI = np.radians(12)
MAX_ANG_VEL = 15.0
GRAVITY = 9.81
MASS = 71.83
LENGTH = 0.85
MAX_TORQUE = 75.0
INERTIA = MASS * LENGTH**2 / 3.0
DAMPING = 4.0

# Training parameters (stabilized)
TOTAL_ITERS = 10000
STEPS_PER_ITER = 2048
DISCRIM_BATCH = 128
DISCRIM_EPOCHS = 1
POLICY_PPO_EPOCHS = 8
POLICY_MINIBATCH = 128
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.1

# Progressive training
MAX_STEPS_TRAINING_INITIAL = 300
MAX_STEPS_TRAINING_FINAL = 15000
MAX_STEPS_EVALUATION = 15000
PROGRESSIVE_STEP_INTERVAL = 50

# Learning rates (stabilized)
POLICY_LR_INIT = 1e-4
VALUE_LR_INIT = 3e-4
DISCRIM_LR_INIT = 5e-5
DISCRIM_WEIGHT_DECAY = 1e-6

# Entropy
ENT_COEF = 0.06

# Reward shaping/scaling (stabilized)
DISC_SCALE = 0.10
REWARD_CLIP = 100.0
RETURN_CLIP = 100.0

# Label smoothing/noise
EXPERT_LABEL_SMOOTH = 0.85
GEN_LABEL_SMOOTH = 0.15
LABEL_NOISE_P = 0.06

# Discriminator architecture (stabilized)
DISC_HIDDEN = (64, 64)
DISC_DROPOUT = 0.4

# Reward normalization
EMA_ALPHA = 0.05

# BC pretraining
BC_EPOCHS = 30
BC_LR = 1e-3
BC_BATCH_SIZE = 128

FIXED_OBS_LEN = 120

# Action smoothing & penalties
ACTION_SMOOTH_ALPHA = 0.4
VEL_PENALTY_WEIGHT = 0
TORQUE_PENALTY = 1e-2
ACTION_CHANGE_PENALTY = 0
SURVIVAL_BONUS = 1.0
ANGLE_REWARD_SCALE = 3.0
ANGLE_REWARD_SIGMA = np.radians(1.0)

# Enhanced metrics
SPECTRUM_NFFT = 256
SPECTRUM_NPERSEC = 2.0
SPECTRUM_FREQ_RANGE = (0.1, 3.0)
PCA_N_COMPONENTS = 4
TSNE_N_COMPONENTS = 2
SAMPEN_EMBED_DIM = 2
SAMPEN_TOLERANCE = 0.2

# Early stopping (we'll switch to survival-based tracking)
EARLY_STOP_PATIENCE = 10000
EARLY_STOP_MIN_IMPROV = 1e-6

# Survival early stopping params (tunable)
SURVIVAL_MIN_IMPROV = 1.0  # in steps
SURVIVAL_PATIENCE = EARLY_STOP_PATIENCE

# ====================== STABILIZATION UTILITIES ======================

class RunningNorm:
    """Online mean/std with EMA smoothing for stability."""
    def __init__(self, eps=1e-6):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.eps = eps
        self.ema_mean = 0.0
        self.ema_var = 1.0
        self.ema_initialized = False

    def update_batch(self, arr):
        for v in np.ravel(arr):
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self.M2 += delta * delta2

        if self.n > 1:
            var = self.var
        else:
            var = self.eps

        if not self.ema_initialized:
            self.ema_mean = self.mean
            self.ema_var = var
            self.ema_initialized = True
        else:
            self.ema_mean = (1.0 - EMA_ALPHA) * self.ema_mean + EMA_ALPHA * self.mean
            self.ema_var = (1.0 - EMA_ALPHA) * self.ema_var + EMA_ALPHA * var

    @property
    def var(self):
        return (self.M2 / (self.n - 1)) if self.n > 1 else self.eps

    @property
    def std(self):
        v = self.var
        return math.sqrt(v) if v > 0 else self.eps

    @property
    def ema_std(self):
        return math.sqrt(self.ema_var) if self.ema_var > 0 else self.eps

    def normalize_with_ema(self, x):
        return (x - self.ema_mean) / max(self.ema_std, self.eps)

def to_torch(x, device=DEVICE, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype, device=device)

def _json_serial(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return str(obj)

# ====================== STABILIZED ENVIRONMENT ======================

class HumanBalanceEnv(gym.Env):
    def __init__(self, dt=DT, max_steps=600, trajectories=None, trajectory_prob=0.3,
                 enable_noise=True, enable_delay=True, action_threshold=0.01,
                 discrete_actions=True, survival_bonus=SURVIVAL_BONUS,
                 angle_reward_scale=ANGLE_REWARD_SCALE, angle_reward_sigma=ANGLE_REWARD_SIGMA,
                 vel_penalty_weight=VEL_PENALTY_WEIGHT, torque_penalty=TORQUE_PENALTY,
                 action_change_penalty=ACTION_CHANGE_PENALTY, max_noise=1.5, max_delay=0.4):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.dt = dt
        self.max_steps = max_steps
        self.state = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.trajectories = trajectories if trajectories is not None else []
        self.trajectory_prob = trajectory_prob
        self.enable_noise = enable_noise
        self.enable_delay = enable_delay
        self.max_noise = max_noise
        self.max_delay = max_delay
        self.current_noise = 0.0
        self.current_delay = 0.0
        self.last_torque = np.zeros(2, dtype=np.float32)
        self.termination_reason = None

        self.action_threshold = action_threshold
        self.discrete_actions = discrete_actions
        self.survival_bonus = survival_bonus
        self.angle_reward_scale = angle_reward_scale
        self.angle_reward_sigma = angle_reward_sigma
        self.vel_penalty_weight = vel_penalty_weight
        self.torque_penalty = torque_penalty
        self.action_change_penalty = action_change_penalty

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        progress = np.random.rand()
        self.current_noise = progress * self.max_noise if self.enable_noise else 0.0
        self.current_delay = progress * self.max_delay if self.enable_delay else 0.0
        if self.trajectories and np.random.rand() < self.trajectory_prob:
            self._reset_from_trajectory()
        else:
            self._random_reset()
        self.step_count = 0
        self.last_torque = np.zeros(2, dtype=np.float32)
        self.termination_reason = None
        return self._get_observation(), {}

    def _discretize_action(self, action):
        if not self.discrete_actions:
            return action
        discrete_action = np.zeros_like(action)
        for i in range(len(action)):
            if action[i] > self.action_threshold:
                discrete_action[i] = 1.0
            elif action[i] < -self.action_threshold:
                discrete_action[i] = -1.0
            else:
                discrete_action[i] = 0.0
        return discrete_action

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        discrete_action = self._discretize_action(a)
        torque_theta = discrete_action[0] * MAX_TORQUE if self.discrete_actions else np.clip(a[0], -1, 1) * MAX_TORQUE
        torque_phi = discrete_action[1] * MAX_TORQUE if self.discrete_actions else np.clip(a[1], -1, 1) * MAX_TORQUE
        torques = np.array([torque_theta, torque_phi], dtype=np.float32)

        θ, φ, dθ, dφ = self.state
        θ_acc = -(GRAVITY/LENGTH)*math.sin(θ) + torque_theta/INERTIA - DAMPING * dθ
        φ_acc = -(GRAVITY/LENGTH)*math.sin(φ) + torque_phi/INERTIA - DAMPING * dφ
        new_dθ = float(np.clip(dθ + θ_acc*self.dt, -MAX_ANG_VEL, MAX_ANG_VEL))
        new_dφ = float(np.clip(dφ + φ_acc*self.dt, -MAX_ANG_VEL, MAX_ANG_VEL))
        new_θ = float(θ + new_dθ*self.dt)
        new_φ = float(φ + new_dφ*self.dt)

        terminated_theta = abs(new_θ) > MAX_THETA
        terminated_phi = abs(new_φ) > MAX_PHI
        terminated = terminated_theta or terminated_phi

        if terminated:
            if terminated_theta and terminated_phi:
                self.termination_reason = 'both_angles'
            elif terminated_theta:
                self.termination_reason = 'theta'
            else:
                self.termination_reason = 'phi'

        env_reward = self.compute_reward(new_θ, new_φ, new_dθ, new_dφ, torques, terminated)

        self.state = np.array([new_θ, new_φ, new_dθ, new_dφ], dtype=np.float32)
        self.last_torque = torques.copy()
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        if truncated:
            self.termination_reason = 'time_limit'
        return self._get_observation(), float(env_reward), bool(terminated), bool(truncated), {}

    def compute_reward(self, θ, φ, dθ, dφ, torques, terminated):
        ang2 = θ*θ + φ*φ
        angle_reward = self.angle_reward_scale * math.exp(-ang2 / (2 * (self.angle_reward_sigma**2 + 1e-12)))
        vel_penalty = self.vel_penalty_weight * (dθ**2 + dφ**2)
        torque_pen = self.torque_penalty * (((torques[0] / MAX_TORQUE)**2) + ((torques[1] / MAX_TORQUE)**2))
        action_change = self.action_change_penalty * float(np.linalg.norm(torques - self.last_torque))
        survival = float(self.survival_bonus) if not terminated else -5.0
        r = survival + angle_reward - vel_penalty - torque_pen - action_change
        r = float(np.clip(r, -REWARD_CLIP, REWARD_CLIP))
        return r

    def _reset_from_trajectory(self):
        traj = random.choice(self.trajectories)
        if len(traj) < 2:
            self._random_reset()
            return
        idx = np.random.randint(0, len(traj)-1)
        x, y = traj[idx]; nx, ny = traj[idx+1]
        θ = float(math.asin(np.clip(x/LENGTH, -1.0, 1.0)))
        φ = float(math.asin(np.clip(y/LENGTH, -1.0, 1.0)))
        next_θ = float(math.asin(np.clip(nx/LENGTH, -1.0, 1.0)))
        next_φ = float(math.asin(np.clip(ny/LENGTH, -1.0, 1.0)))
        dθ = float((next_θ - θ)/self.dt)
        dφ = float((next_φ - φ)/self.dt)
        self.state = np.array([θ, φ, np.clip(dθ, -MAX_ANG_VEL, MAX_ANG_VEL), np.clip(dφ, -MAX_ANG_VEL, MAX_ANG_VEL)], dtype=np.float32)

    def _random_reset(self):
        self.state = np.array([
            np.random.uniform(-np.radians(0.5), np.radians(0.5)),
            np.random.uniform(-np.radians(0.5), np.radians(0.5)),
            0.0, 0.0
        ], dtype=np.float32)

    def _get_observation(self):
        θ, φ, dθ, dφ = self.state
        noisy_θ = θ + math.radians(np.random.normal(0, self.current_noise)) if self.enable_noise else θ
        noisy_φ = φ + math.radians(np.random.normal(0, self.current_noise)) if self.enable_noise else φ
        θ_scaled = np.clip(noisy_θ / MAX_THETA, -1.0, 1.0)
        φ_scaled = np.clip(noisy_φ / MAX_PHI, -1.0, 1.0)
        dθ_scaled = np.clip(dθ / MAX_ANG_VEL, -1.0, 1.0)
        dφ_scaled = np.clip(dφ / MAX_ANG_VEL, -1.0, 1.0)
        return np.array([
            θ_scaled, φ_scaled,
            math.cos(noisy_θ), math.sin(noisy_θ),
            math.cos(noisy_φ), math.sin(noisy_φ),
            dθ_scaled, dφ_scaled,
            self.current_delay / self.max_delay,
            self.current_noise / self.max_noise
        ], dtype=np.float32)

# ====================== STABILIZED NETWORKS ======================

def mlp(in_dim, out_dim, hidden=(256,256), activation=nn.ReLU, dropout=0.0):
    layers = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        layers.append(nn.LayerNorm(h))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, obs_dim=10, act_dim=2, hidden=DISC_HIDDEN, dropout=DISC_DROPOUT):
        super().__init__()
        self.net = mlp(8 + act_dim, 1, hidden=hidden, activation=nn.ReLU, dropout=dropout)

    def forward(self, obs, act):
        x = torch.cat([obs[:, :8], act], dim=1)
        return self.net(x).squeeze(-1)

class PolicyValue(nn.Module):
    def __init__(self, obs_dim=10, act_dim=2, hidden=(256,256)):
        super().__init__()
        self.actor = nn.Sequential(mlp(obs_dim, act_dim, hidden), nn.Tanh())
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.5))
        self.critic = mlp(obs_dim, 1, hidden)

    def forward(self, obs):
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        return mu, std

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

# ====================== STABILIZED DISCRIMINATOR TRAINING ======================

bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

def train_discriminator_stable(discriminator, optimizer, expert_obs, expert_acts, gen_obs, gen_acts,
                              epochs=1, batch_size=DISCRIM_BATCH):
    device = next(discriminator.parameters()).device
    n_ex = len(expert_obs)
    n_gen = len(gen_obs)
    if n_ex == 0 or n_gen == 0:
        return 0.0, 0.0, 0.0

    idx_ex = np.arange(n_ex)
    idx_gen = np.arange(n_gen)
    losses = []
    acc_ex_total = 0.0
    acc_gen_total = 0.0
    total_batches = 0

    for e in range(epochs):
        np.random.shuffle(idx_ex)
        np.random.shuffle(idx_gen)
        steps = max(1, min(max(1, n_ex // batch_size), max(1, n_gen // batch_size)))

        for i in range(steps):
            be = idx_ex[i*batch_size:(i+1)*batch_size]
            bg = idx_gen[i*batch_size:(i+1)*batch_size]
            obs_e = to_torch(expert_obs[be], device=device)
            acts_e = to_torch(expert_acts[be], device=device)
            obs_g = to_torch(gen_obs[bg], device=device)
            acts_g = to_torch(gen_acts[bg], device=device)

            logits_e = discriminator(obs_e, acts_e)
            logits_g = discriminator(obs_g, acts_g)

            labels_e = torch.full_like(logits_e, EXPERT_LABEL_SMOOTH, device=device, dtype=torch.float32)
            labels_g = torch.full_like(logits_g, GEN_LABEL_SMOOTH, device=device, dtype=torch.float32)

            if LABEL_NOISE_P > 0:
                flip_mask_e = (torch.rand_like(labels_e) < LABEL_NOISE_P)
                flip_mask_g = (torch.rand_like(labels_g) < LABEL_NOISE_P)
                labels_e = torch.where(flip_mask_e, 1.0 - labels_e, labels_e)
                labels_g = torch.where(flip_mask_g, 1.0 - labels_g, labels_g)

            loss = bce_loss(logits_e, labels_e) + bce_loss(logits_g, labels_g)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                preds_e = (torch.sigmoid(logits_e) > 0.5).float()
                preds_g = (torch.sigmoid(logits_g) > 0.5).float()
                target_e = (labels_e > 0.5).float()
                target_g = (labels_g > 0.5).float()
                acc_e = preds_e.eq(target_e).float().mean().item()
                acc_g = preds_g.eq(target_g).float().mean().item()

                acc_ex_total += acc_e
                acc_gen_total += acc_g

            losses.append(loss.item())
            total_batches += 1

    if total_batches > 0:
        return float(np.mean(losses)), float(acc_ex_total/total_batches), float(acc_gen_total/total_batches)
    else:
        return 0.0, 0.0, 0.0

# ====================== DATA PREPARATION ======================

def load_trajectories(csv_path, group_col='path', time_col='n', x_col='X', y_col='y'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path, encoding='utf-8')
    df = df[df['tp'] == "ROMBERG"].copy()

    for c in [x_col, y_col, time_col]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=[group_col, time_col, x_col, y_col], inplace=True)

    groups = []
    labels = []

    print(f"Grouping by '{group_col}' to create trajectories...")

    for name, g in df.groupby(group_col):
        arr = g.sort_values(time_col)[[x_col, y_col]].values / 1000.0

        if len(arr) >= 100:
            groups.append(arr)
            participant = g['name'].iloc[0] if 'name' in g.columns else 'Unknown'
            test_type = g['tp'].iloc[0] if 'tp' in g.columns else 'Unknown'
            sensitivity = g['fn_sens'].iloc[0] if 'fn_sens' in g.columns else 'Unknown'
            labels.append(f"{participant}_{test_type}_sens{sensitivity}")

    print(f"✅ Loaded {len(groups)} trajectories from {len(df[group_col].unique())} unique test sessions")
    return groups, labels

def positions_to_states(traj_xy):
    xs, ys = traj_xy[:, 0], traj_xy[:, 1]
    thetas = np.arcsin(np.clip(xs / LENGTH, -1.0, 1.0))
    phis = np.arcsin(np.clip(ys / LENGTH, -1.0, 1.0))
    dtheta = np.zeros_like(thetas)
    dphi = np.zeros_like(phis)
    for i in range(1, len(thetas)-1):
        dtheta[i] = (thetas[i+1] - thetas[i-1])/(2*DT)
        dphi[i] = (phis[i+1] - phis[i-1])/(2*DT)
    if len(thetas) > 1:
        dtheta[0] = (thetas[1] - thetas[0]) / DT
        dphi[0] = (phis[1] - phis[0]) / DT
        dtheta[-1] = (thetas[-1] - thetas[-2]) / DT
        dphi[-1] = (phis[-1] - phis[-2]) / DT
    dtheta = np.clip(dtheta, -MAX_ANG_VEL, MAX_ANG_VEL)
    dphi = np.clip(dphi, -MAX_ANG_VEL, MAX_ANG_VEL)
    return np.stack([thetas, phis, dtheta, dphi], axis=1)

def state_to_obs(state):
    θ, φ, dθ, dφ = state
    θ_scaled = np.clip(θ / MAX_THETA, -1.0, 1.0)
    φ_scaled = np.clip(φ / MAX_PHI, -1.0, 1.0)
    dθ_scaled = np.clip(dθ / MAX_ANG_VEL, -1.0, 1.0)
    dφ_scaled = np.clip(dφ / MAX_ANG_VEL, -1.0, 1.0)
    return np.array([
        θ_scaled, φ_scaled, math.cos(θ), math.sin(θ),
        math.cos(φ), math.sin(φ), dθ_scaled, dφ_scaled, 0.0, 0.0
    ], dtype=np.float32)

def _discretize_expert_action(action, threshold=0.01):
    discrete_action = np.zeros_like(action)
    for i in range(len(action)):
        if action[i] > threshold:
            discrete_action[i] = 1.0
        elif action[i] < -threshold:
            discrete_action[i] = -1.0
        else:
            discrete_action[i] = 0.0
    return discrete_action

def estimate_actions(states, expert_threshold=0.01):
    obs_raw = []
    acts = []
    for i in range(len(states)-1):
        θ, φ, dθ, dφ = states[i]
        θn, φn, dθn, dφn = states[i+1]
        aθ = (dθn - dθ)/DT
        aφ = (dφn - dφ)/DT
        torque_theta = INERTIA * (aθ + DAMPING*dθ + (GRAVITY/LENGTH)*math.sin(θ))
        torque_phi = INERTIA * (aφ + DAMPING*dφ + (GRAVITY/LENGTH)*math.sin(φ))
        cont_action_theta = float(np.clip(torque_theta / MAX_TORQUE, -1.0, 1.0))
        cont_action_phi = float(np.clip(torque_phi / MAX_TORQUE, -1.0, 1.0))
        discrete_action = _discretize_expert_action([cont_action_theta, cont_action_phi], threshold=expert_threshold)
        acts.append(discrete_action)
        obs_raw.append([θ, φ, dθ, dφ])
    return np.array(obs_raw, dtype=np.float32), np.array(acts, dtype=np.float32)

def build_expert_dataset(trajectories, fixed_len=FIXED_OBS_LEN, expert_threshold=0.01):
    obs_list = []
    act_list = []
    seqs_for_vis = []
    for t in trajectories:
        states = positions_to_states(t)
        obs_raw, acts = estimate_actions(states, expert_threshold=expert_threshold)
        if len(obs_raw) == 0:
            continue
        obs_env = np.array([state_to_obs(s) for s in obs_raw], dtype=np.float32)
        obs_list.append(obs_env)
        act_list.append(acts)
        seqs_for_vis.append(states)

    if not obs_list:
        raise RuntimeError("No expert data produced.")

    expert_obs = np.concatenate(obs_list, axis=0)
    expert_acts = np.concatenate(act_list, axis=0)
    print(f"Built expert dataset: {expert_obs.shape[0]} (s,a) pairs, obs dim {expert_obs.shape[1]}")

    unique_actions, counts = np.unique(expert_acts, return_counts=True, axis=0)
    print("Expert discrete action distribution:")
    for action, count in zip(unique_actions, counts):
        print(f"  Action {action}: {count} samples ({count/len(expert_acts)*100:.1f}%)")

    return expert_obs, expert_acts, seqs_for_vis

# ====================== TRAJECTORY SAVING FUNCTIONS ======================

def save_expert_trajectories_csv(trajectories, labels, filename="expert_trajectories.csv"):
    """
    Save expert trajectories to CSV with exact coordinates and metadata.

    Args:
        trajectories: List of numpy arrays with shape (n_points, 2) containing (x, y) coordinates
        labels: List of trajectory labels
        filename: Output CSV filename
    """
    print(f"Saving expert trajectories to {filename}...")

    all_data = []
    for traj_idx, (traj, label) in enumerate(zip(trajectories, labels)):
        # Convert trajectory to states
        states = positions_to_states(traj)

        for point_idx, (point, state) in enumerate(zip(traj, states)):
            x, y = point
            theta, phi, dtheta, dphi = state

            all_data.append({
                'trajectory_id': traj_idx,
                'trajectory_label': label,
                'point_index': point_idx,
                'time_seconds': point_idx * DT,
                'x_coordinate': x,
                'y_coordinate': y,
                'theta_radians': theta,
                'phi_radians': phi,
                'dtheta_radians_per_sec': dtheta,
                'dphi_radians_per_sec': dphi
            })

    # Create DataFrame and save
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(trajectories)} expert trajectories with {len(df)} total points to {filename}")

    return df

def save_agent_trajectories_csv(agent_trajectories, filename="agent_trajectories.csv"):
    """
    Save agent trajectories to CSV with exact coordinates and metadata.

    Args:
        agent_trajectories: List of numpy arrays with shape (n_points, 4) containing states (theta, phi, dtheta, dphi)
        filename: Output CSV filename
    """
    print(f"Saving agent trajectories to {filename}...")

    all_data = []
    for traj_idx, states in enumerate(agent_trajectories):
        for point_idx, state in enumerate(states):
            theta, phi, dtheta, dphi = state

            # Convert angles back to coordinates
            x = LENGTH * math.sin(theta)
            y = LENGTH * math.sin(phi)

            all_data.append({
                'trajectory_id': traj_idx,
                'point_index': point_idx,
                'time_seconds': point_idx * DT,
                'x_coordinate': x,
                'y_coordinate': y,
                'theta_radians': theta,
                'phi_radians': phi,
                'dtheta_radians_per_sec': dtheta,
                'dphi_radians_per_sec': dphi
            })

    # Create DataFrame and save
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(agent_trajectories)} agent trajectories with {len(df)} total points to {filename}")

    return df

def save_comparison_trajectories_csv(expert_trajectories, expert_labels, agent_trajectories,
                                   filename="comparison_trajectories.csv"):
    """
    Save both expert and agent trajectories in a single CSV for easy comparison.

    Args:
        expert_trajectories: List of expert trajectory arrays (x, y)
        expert_labels: List of expert trajectory labels
        agent_trajectories: List of agent trajectory arrays (states)
        filename: Output CSV filename
    """
    print(f"Saving comparison trajectories to {filename}...")

    all_data = []

    # Save expert trajectories
    for traj_idx, (traj, label) in enumerate(zip(expert_trajectories, expert_labels)):
        states = positions_to_states(traj)

        for point_idx, (point, state) in enumerate(zip(traj, states)):
            x, y = point
            theta, phi, dtheta, dphi = state

            all_data.append({
                'source': 'expert',
                'trajectory_id': traj_idx,
                'trajectory_label': label,
                'point_index': point_idx,
                'time_seconds': point_idx * DT,
                'x_coordinate': x,
                'y_coordinate': y,
                'theta_radians': theta,
                'phi_radians': phi,
                'dtheta_radians_per_sec': dtheta,
                'dphi_radians_per_sec': dphi
            })

    # Save agent trajectories
    for traj_idx, states in enumerate(agent_trajectories):
        for point_idx, state in enumerate(states):
            theta, phi, dtheta, dphi = state
            x = LENGTH * math.sin(theta)
            y = LENGTH * math.sin(phi)

            all_data.append({
                'source': 'agent',
                'trajectory_id': traj_idx,
                'trajectory_label': f'agent_{traj_idx}',
                'point_index': point_idx,
                'time_seconds': point_idx * DT,
                'x_coordinate': x,
                'y_coordinate': y,
                'theta_radians': theta,
                'phi_radians': phi,
                'dtheta_radians_per_sec': dtheta,
                'dphi_radians_per_sec': dphi
            })

    # Create DataFrame and save
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(expert_trajectories)} expert + {len(agent_trajectories)} agent trajectories to {filename}")

    return df

# ====================== ROLLOUT COLLECTION ======================

def collect_rollouts(env, policy, n_steps):
    obs_buf, act_buf, logp_buf, val_buf, done_buf, state_buf, rew_buf, termination_reasons = [], [], [], [], [], [], [], []
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.float32)
    t = 0
    prev_action = np.zeros(env.action_space.shape, dtype=np.float32)
    last_obs = obs.copy()

    while t < n_steps:
        obs_tensor = to_torch(obs[None,:])
        with torch.no_grad():
            mu, std = policy(obs_tensor)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(axis=-1)
            action_np = action.cpu().numpy()[0]
            value = policy.value(obs_tensor).cpu().numpy()[0]

        smoothed = ACTION_SMOOTH_ALPHA * prev_action + (1.0 - ACTION_SMOOTH_ALPHA) * action_np
        prev_action = smoothed.copy()
        next_obs, env_reward, done, truncated, _ = env.step(smoothed)

        obs_buf.append(obs.copy())
        act_buf.append(smoothed.copy())
        logp_buf.append(float(logp.cpu().numpy()))
        val_buf.append(float(value))
        done_buf.append(bool(done))
        state_buf.append(deepcopy(env.state))
        rew_buf.append(float(env_reward))
        termination_reasons.append(env.termination_reason)

        obs = np.asarray(next_obs, dtype=np.float32)
        last_obs = obs.copy()
        t += 1
        if done or truncated:
            obs, _ = env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            prev_action = np.zeros(env.action_space.shape, dtype=np.float32)

    return {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "acts": np.asarray(act_buf, dtype=np.float32),
        "logps": np.asarray(logp_buf, dtype=np.float32),
        "vals": np.asarray(val_buf, dtype=np.float32),
        "dones": np.asarray(done_buf),
        "states": np.asarray(state_buf, dtype=np.float32),
        "rewards": np.asarray(rew_buf, dtype=np.float32),
        "last_obs": last_obs,
        "termination_reasons": termination_reasons
    }

# ====================== PPO UTILITIES ======================

def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        mask = 0.0 if dones[t] else 1.0
        next_value = values[t+1] if t+1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * mask * lastgaelam
    returns = advantages + values[:T]
    return advantages, returns

def pretrain_policy_with_bc(policy, expert_obs, expert_acts):
    print("\n--- Starting Behavioral Cloning Pre-training ---")
    bc_dataset = TensorDataset(
        torch.tensor(expert_obs, dtype=torch.float32),
        torch.tensor(expert_acts, dtype=torch.float32)
    )
    bc_loader = DataLoader(bc_dataset, batch_size=BC_BATCH_SIZE, shuffle=True)
    bc_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=BC_LR)
    loss_fn = nn.MSELoss()

    for epoch in range(BC_EPOCHS):
        total_loss = 0
        for obs_batch, act_batch in bc_loader:
            obs_batch, act_batch = obs_batch.to(DEVICE), act_batch.to(DEVICE)
            pred_acts = policy.actor(obs_batch)
            loss = loss_fn(pred_acts, act_batch)
            bc_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 0.5)
            bc_optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(bc_loader)
        if (epoch + 1) % max(1, BC_EPOCHS//10) == 0 or epoch == 0:
            print(f"[BC Epoch {epoch+1}/{BC_EPOCHS}] MSE Loss: {avg_loss:.6f}")
    print("--- Behavioral Cloning Finished ---\n")

# ====================== PROGRESSIVE TRAINING UTILITIES ======================

def update_environment_max_steps(env, current_iter, total_iters,
                               initial_max_steps=MAX_STEPS_TRAINING_INITIAL,
                               final_max_steps=MAX_STEPS_TRAINING_FINAL,
                               step_interval=PROGRESSIVE_STEP_INTERVAL):
    if current_iter % step_interval == 0:
        progress = min(1.0, current_iter / total_iters)
        new_max_steps = int(initial_max_steps + (final_max_steps - initial_max_steps) * progress)
        env.max_steps = new_max_steps
        return new_max_steps
    return env.max_steps

def analyze_survival_data(rollouts, current_iter):
    termination_reasons = rollouts.get("termination_reasons", [])
    if not termination_reasons:
        return

    reasons = {}
    for reason in termination_reasons:
        if reason is None:
            continue
        reasons[reason] = reasons.get(reason, 0) + 1

    total_terminations = sum(reasons.values())
    if total_terminations > 0:
        print(f"  Survival Analysis (Iter {current_iter}):")
        for reason, count in reasons.items():
            percentage = (count / total_terminations) * 100
            print(f"    {reason}: {count} ({percentage:.1f}%)")

# ====================== STABILIZED TRAINING LOOP ======================

def train_gail_stable(env, expert_obs, expert_acts, expert_seqs, num_iters=TOTAL_ITERS, steps_per_iter=STEPS_PER_ITER, return_metrics=False, save_every=50, checkpoint_dir="checkpoints_stable"):
    obs_dim = expert_obs.shape[1]
    act_dim = expert_acts.shape[1]

    # Initialize networks
    disc = Discriminator(obs_dim, act_dim).to(DEVICE)
    policy = PolicyValue(obs_dim, act_dim).to(DEVICE)

    # BC pretraining
    pretrain_policy_with_bc(policy, expert_obs, expert_acts)

    # Optimizers
    policy_params = list(policy.actor.parameters()) + [policy.log_std] + list(policy.critic.parameters())
    opt_policy = torch.optim.Adam([
        {'params': policy.actor.parameters(), 'lr': POLICY_LR_INIT},
        {'params': [policy.log_std], 'lr': POLICY_LR_INIT},
        {'params': policy.critic.parameters(), 'lr': VALUE_LR_INIT}
    ], eps=1e-5)

    opt_discrim = torch.optim.Adam(
        disc.parameters(),
        lr=DISCRIM_LR_INIT,
        weight_decay=DISCRIM_WEIGHT_DECAY,
        eps=1e-5
    )

    # Expert dataset
    expert_dataset = TensorDataset(to_torch(expert_obs), to_torch(expert_acts))
    safe_batch = min(DISCRIM_BATCH, max(1, len(expert_dataset)))
    expert_loader = DataLoader(expert_dataset, batch_size=safe_batch, shuffle=True, drop_last=True)
    expert_batches = list(expert_loader)
    if len(expert_batches) == 0:
        raise RuntimeError("Expert dataset produced zero batches.")
    print(f"Expert batches: {len(expert_batches)} (batch size {safe_batch}).")

    # Reward normalization
    env_norm = RunningNorm()
    disc_norm = RunningNorm()

    # Early stopping and tracking (survival based)
    best_survival = -np.inf
    best_combined_reward = -np.inf
    best_policy_state = None
    best_disc_state = None
    best_iter = 0
    no_improve_iters = 0

    # Metrics and logging
    metrics_history = {
        "iter": [], "mean_combined_reward": [], "mean_D": [], "mean_Ploss": [], "mean_Vloss": [],
        "mean_disc_loss": [], "mean_acc_e": [], "mean_acc_g": [], "entropy": [], "approx_kl": [],
        "grad_norm_actor": [], "grad_norm_critic": [], "best_combined_reward": [], "current_max_steps": [],
        "time_per_iter": [], "mean_survival": []
    }

    # CSV logging
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    csv_file = open(Path(checkpoint_dir) / "training_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "iter", "time_s", "disc_loss", "disc_acc_ex", "disc_acc_gen", "mean_D",
        "env_mean", "disc_mean", "combined_mean", "policy_loss", "value_loss",
        "entropy", "grad_norm_actor", "grad_norm_critic", "mean_survival"
    ])

    for it in range(num_iters):
        iter_start_time = time.time()

        # Progressive training
        current_max_steps = update_environment_max_steps(env, it, num_iters)

        # Collect rollouts
        roll = collect_rollouts(env, policy, steps_per_iter)
        obs_g = roll["obs"]; acts_g = roll["acts"]; old_logps = roll["logps"]
        vals = roll["vals"]; dones = roll["dones"]; env_rewards = roll["rewards"]
        last_obs = roll["last_obs"]

        if len(obs_g) < 2:
            print("Warning: generator produced too few samples; skipping iteration.")
            continue

        # Survival analysis
        analyze_survival_data(roll, it + 1)

        # Discriminator scores with stabilization
        with torch.no_grad():
            obs_t = to_torch(obs_g)
            acts_t = to_torch(acts_g)
            logits = disc(obs_t, acts_t)
            d_scores = torch.sigmoid(logits).cpu().numpy()

            # Stabilized shaping: tanh squashing
            raw_shaping = -np.log(np.clip(1.0 - d_scores, 1e-8, 1.0))
            rewards_disc = np.tanh(raw_shaping) * 5.0

        # Update running stats
        env_norm.update_batch(env_rewards)
        disc_norm.update_batch(rewards_disc)

        # Normalize rewards using EMA stats
        rewards_env_norm = env_norm.normalize_with_ema(env_rewards)
        rewards_disc_norm = disc_norm.normalize_with_ema(rewards_disc)

        # Combined rewards with clipping
        combined_rewards = rewards_env_norm + (DISC_SCALE * rewards_disc_norm)
        combined_rewards = np.clip(combined_rewards, -REWARD_CLIP, REWARD_CLIP)

        # Last value for bootstrap
        with torch.no_grad():
            if last_obs is not None:
                last_obs_t = to_torch(np.asarray(last_obs, dtype=np.float32)[None, :])
                last_value = float(policy.value(last_obs_t).cpu().numpy()[0])
            else:
                last_value = 0.0

        values_arr = np.concatenate([vals, np.array([last_value], dtype=np.float32)], axis=0)
        advantages, returns = compute_gae(combined_rewards, values_arr, dones)

        # Advantage normalization
        adv_mean, adv_std = advantages.mean(), max(advantages.std(), 1e-8)
        advantages = (advantages - adv_mean) / adv_std

        # Clip returns for critic stability
        returns = np.clip(returns, -RETURN_CLIP, RETURN_CLIP)

        # Train discriminator
        try:
            disc_loss, disc_acc_ex, disc_acc_gen = train_discriminator_stable(
                disc, opt_discrim, expert_obs, expert_acts, obs_g, acts_g,
                epochs=DISCRIM_EPOCHS, batch_size=DISCRIM_BATCH
            )
        except Exception as e:
            print(f"Discriminator training error: {e}")
            disc_loss, disc_acc_ex, disc_acc_gen = 0.0, 0.0, 0.0

        # PPO updates
        dataset_size = len(obs_g)
        b_obs = to_torch(obs_g)
        b_acts = to_torch(acts_g)
        b_oldlogp = to_torch(old_logps)
        b_adv = to_torch(advantages)
        b_ret = to_torch(returns)
        b_vals = to_torch(vals[:dataset_size])

        idxs = np.arange(dataset_size)
        ppo_policy_losses = []
        ppo_value_losses = []
        iter_entropies = []
        iter_approx_kl = []
        grad_norms_actor = []
        grad_norms_critic = []

        for epoch in range(POLICY_PPO_EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, POLICY_MINIBATCH):
                mb_idx = idxs[start:start+POLICY_MINIBATCH]
                if len(mb_idx) == 0:
                    continue

                mb_obs = b_obs[mb_idx]; mb_acts = b_acts[mb_idx]
                mb_oldlogp = b_oldlogp[mb_idx]; mb_adv = b_adv[mb_idx]
                mb_ret = b_ret[mb_idx]; mb_vals = b_vals[mb_idx]

                mu, std = policy(mb_obs)
                dist = torch.distributions.Normal(mu, std)
                newlogp = dist.log_prob(mb_acts).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1).mean()

                # Diagnostics
                iter_entropies.append(float(entropy.item()))
                approx_kl = float((mb_oldlogp - newlogp).mean().item())
                iter_approx_kl.append(approx_kl)

                ratio = torch.exp(newlogp - mb_oldlogp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                value_pred = policy.value(mb_obs)
                value_pred_clipped = mb_vals + torch.clamp(value_pred - mb_vals, -0.2, 0.2)
                value_loss_unclipped = (value_pred - mb_ret).pow(2)
                value_loss_clipped = (value_pred_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = -ENT_COEF * entropy
                total_loss = policy_loss + value_loss + entropy_loss

                opt_policy.zero_grad()
                total_loss.backward()

                # Gradient monitoring
                actor_params = list(policy.actor.parameters()) + [policy.log_std]
                critic_params = list(policy.critic.parameters())

                actor_grad_norm = 0.0
                for p in actor_params:
                    if p.grad is not None:
                        actor_grad_norm += p.grad.norm().item() ** 2
                actor_grad_norm = math.sqrt(actor_grad_norm)
                grad_norms_actor.append(actor_grad_norm)

                critic_grad_norm = 0.0
                for p in critic_params:
                    if p.grad is not None:
                        critic_grad_norm += p.grad.norm().item() ** 2
                critic_grad_norm = math.sqrt(critic_grad_norm)
                grad_norms_critic.append(critic_grad_norm)

                torch.nn.utils.clip_grad_norm_(policy_params, 0.5)
                opt_policy.step()

                ppo_policy_losses.append(float(policy_loss.item()))
                ppo_value_losses.append(float(value_loss.item()))

        # Metrics calculation
        mean_Ploss = float(np.mean(ppo_policy_losses)) if ppo_policy_losses else 0.0
        mean_Vloss = float(np.mean(ppo_value_losses)) if ppo_value_losses else 0.0
        mean_D = float(d_scores.mean()) if d_scores.size > 0 else 0.0
        current_mean_combined_reward = np.mean(combined_rewards) if combined_rewards.size > 0 else -np.inf

        # ---------------- Survival-based early stopping & best-model tracking ----------------
        def compute_episode_lengths_from_roll(roll):
            dones = np.asarray(roll.get("dones", []), dtype=np.bool_)
            if dones.size == 0:
                return []
            lengths = []
            cur_len = 0
            for d in dones:
                cur_len += 1
                if bool(d):
                    lengths.append(cur_len)
                    cur_len = 0
            # if last episode didn't end within the rollout, include its partial length
            if cur_len > 0:
                lengths.append(cur_len)
            return lengths

        episode_lengths = compute_episode_lengths_from_roll(roll)
        mean_survival = float(np.mean(episode_lengths)) if len(episode_lengths) > 0 else 0.0

        # New logic incorporating combined reward as tie-breaker
        is_new_best_by_survival = mean_survival > best_survival + SURVIVAL_MIN_IMPROV
        is_best_by_reward_tie_breaker = False

        # Check for tie or marginal improvement in survival (using a small tolerance, e.g., 1e-6)
        if mean_survival >= best_survival - 1e-6:
            # If survival is maintained or improved marginally, check if combined reward is strictly better
            if current_mean_combined_reward > best_combined_reward + EARLY_STOP_MIN_IMPROV:
                is_best_by_reward_tie_breaker = True

        if is_new_best_by_survival or is_best_by_reward_tie_breaker:
            best_survival = mean_survival
            best_combined_reward = current_mean_combined_reward
            best_iter = it + 1
            best_policy_state = deepcopy(policy.state_dict())
            best_disc_state = deepcopy(disc.state_dict())
            no_improve_iters = 0
            print(f"  🏆 New best model! Mean survival: {mean_survival:.2f} steps, Combined Reward: {best_combined_reward:.4f} (iter {best_iter})")
        else:
            no_improve_iters += 1

        # ---------------- End of modified block ----------------

        # Update metrics history
        iter_time = time.time() - iter_start_time
        metrics_history["iter"].append(it+1)
        metrics_history["mean_combined_reward"].append(current_mean_combined_reward)
        metrics_history["mean_D"].append(mean_D)
        metrics_history["mean_Ploss"].append(mean_Ploss)
        metrics_history["mean_Vloss"].append(mean_Vloss)
        metrics_history["mean_disc_loss"].append(disc_loss)
        metrics_history["mean_acc_e"].append(disc_acc_ex)
        metrics_history["mean_acc_g"].append(disc_acc_gen)
        metrics_history["entropy"].append(float(np.mean(iter_entropies)) if iter_entropies else 0.0)
        metrics_history["approx_kl"].append(float(np.mean(iter_approx_kl)) if iter_approx_kl else 0.0)
        metrics_history["grad_norm_actor"].append(float(np.mean(grad_norms_actor)) if grad_norms_actor else 0.0)
        metrics_history["grad_norm_critic"].append(float(np.mean(grad_norms_critic)) if grad_norms_critic else 0.0)
        metrics_history["best_combined_reward"].append(best_combined_reward)
        metrics_history["current_max_steps"].append(current_max_steps)
        metrics_history["time_per_iter"].append(iter_time)
        metrics_history["mean_survival"].append(mean_survival)

        # CSV logging
        csv_writer.writerow([
            it+1, iter_time, disc_loss, disc_acc_ex, disc_acc_gen, mean_D,
            float(np.mean(env_rewards)) if env_rewards.size > 0 else 0.0,
            float(np.mean(rewards_disc)) if rewards_disc.size > 0 else 0.0,
            current_mean_combined_reward,
            mean_Ploss, mean_Vloss,
            float(np.mean(iter_entropies)) if iter_entropies else 0.0,
            metrics_history["grad_norm_actor"][-1],
            metrics_history["grad_norm_critic"][-1],
            mean_survival
        ])
        csv_file.flush()

        # Print iteration summary
        print(f"[Iter {it+1}/{num_iters}] time={iter_time:.2f}s")
        print(f"  Disc: loss={disc_loss:.5f}, acc_e={disc_acc_ex:.3f}, acc_g={disc_acc_gen:.3f}")
        print(f"  PPO: Ploss={mean_Ploss:.5f}, Vloss={mean_Vloss:.5f}")
        print(f"  Rewards: env={np.mean(env_rewards):.4f}, disc={np.mean(rewards_disc):.4f}, combined={current_mean_combined_reward:.4f}")
        print(f"  D_scores: min={d_scores.min():.4f}, mean={mean_D:.4f}, max={d_scores.max():.4f}")
        print(f"  Grad norms: actor={metrics_history['grad_norm_actor'][-1]:.4f}, critic={metrics_history['grad_norm_critic'][-1]:.4f}")
        print(f"  Current max_steps: {current_max_steps}, Best survival: {best_survival:.4f} (iter {best_iter})")
        print(f"  No improvement iters: {no_improve_iters}/{SURVIVAL_PATIENCE}")

        # Early stopping by survival
        if no_improve_iters >= SURVIVAL_PATIENCE:
            print(f"Early stopping: no survival improvement for {SURVIVAL_PATIENCE} iterations.")
            break

        # Periodic checkpointing
        if (it+1) % save_every == 0:
            torch.save(policy.state_dict(), Path(checkpoint_dir)/f"policy_iter_{it+1}.pt")
            torch.save(disc.state_dict(), Path(checkpoint_dir)/f"disc_iter_{it+1}.pt")

            if best_policy_state is not None:
                torch.save(best_policy_state, Path(checkpoint_dir)/"policy_best.pt")
                torch.save(best_disc_state, Path(checkpoint_dir)/"disc_best.pt")

            save_metrics_history(metrics_history, out_dir=checkpoint_dir, name=f"metrics_iter_{it+1}.json")

    # Final save
    csv_file.close()
    if best_policy_state is not None:
        torch.save(best_policy_state, Path(checkpoint_dir)/"policy_best.pt")
        torch.save(best_disc_state, Path(checkpoint_dir)/"disc_best.pt")
        save_metrics_history(metrics_history, out_dir=checkpoint_dir, name="metrics_best.json")
        print(f"\n=== Training Completed ===")
        print(f"Best model: iteration {best_iter}, mean survival: {best_survival:.4f}, combined reward: {best_combined_reward:.4f}")
    else:
        torch.save(policy.state_dict(), Path(checkpoint_dir)/"policy_best.pt")
        torch.save(disc.state_dict(), Path(checkpoint_dir)/"disc_best.pt")
        print(f"\n=== Training Completed ===")
        print("Saved final model as best")

    if return_metrics:
        return policy, disc, metrics_history
    return policy, disc



# ====================== ENHANCED METRICS MODULE ======================

class EnhancedTrajectoryMetrics:
    def __init__(self, dt=DT, spectrum_freq_range=SPECTRUM_FREQ_RANGE,
                 pca_components=PCA_N_COMPONENTS, tsne_components=TSNE_N_COMPONENTS):
        self.dt = dt
        self.spectrum_freq_range = spectrum_freq_range
        self.pca_components = pca_components
        self.tsne_components = tsne_components

    def compute_spectral_features(self, signal, nperseg=None):
        if len(signal) < 2:
            return {'dom_freq': 0.0, 'centroid': 0.0, 'power': 0.0, 'bandwidth': 0.0, 'freqs': np.array([]), 'psd': np.array([])}

        if nperseg is None:
            nperseg = min(len(signal), max(16, int(SPECTRUM_NPERSEC / self.dt)))
            nperseg = max(nperseg, 16)

        try:
            freqs, psd = welch(signal, fs=1/self.dt, nperseg=nperseg, nfft=SPECTRUM_NFFT)
            freq_mask = (freqs >= self.spectrum_freq_range[0]) & (freqs <= self.spectrum_freq_range[1])
            freqs = freqs[freq_mask]
            psd = psd[freq_mask]

            if len(psd) == 0:
                return {'dom_freq': 0.0, 'centroid': 0.0, 'power': 0.0, 'bandwidth': 0.0, 'freqs': np.array([]), 'psd': np.array([])}

            dom_freq = freqs[np.argmax(psd)]
            centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-12)
            total_power = np.sum(psd)
            bandwidth = np.sqrt(np.sum(psd * (freqs - centroid)**2) / (total_power + 1e-12))

            return {
                'dom_freq': float(dom_freq),
                'centroid': float(centroid),
                'power': float(total_power),
                'bandwidth': float(bandwidth),
                'freqs': freqs,
                'psd': psd
            }
        except Exception as e:
            print(f"Spectral analysis error: {e}")
            return {'dom_freq': 0.0, 'centroid': 0.0, 'power': 0.0, 'bandwidth': 0.0, 'freqs': np.array([]), 'psd': np.array([])}

    def compute_sample_entropy(self, signal, m=SAMPEN_EMBED_DIM, r=SAMPEN_TOLERANCE):
        if len(signal) < m + 1:
            return 0.0

        try:
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)

            def _maxdist(xi, xj):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])

            def _phi(m):
                x = [[signal[j] for j in range(i, i + m)] for i in range(len(signal) - m + 1)]
                C = [len([1 for xj in x if _maxdist(xi, xj) <= r]) for xi in x]
                return sum(C) / (len(signal) - m + 1.0)

            return float(-np.log(_phi(m + 1) / (_phi(m) + 1e-12)))
        except Exception:
            return 0.0

    def trajectory_to_feature_vector(self, states):
        if len(states) < 2:
            return np.zeros(20)

        features = []

        # Basic statistical moments
        for i in range(4):
            sig = states[:, i]
            features.extend([np.mean(sig), np.std(sig), np.var(sig), scipy_entropy(np.histogram(sig, bins=10)[0] + 1e-12)])

        # Spectral features for theta and phi
        for i in [0, 1]:
            spec_feats = self.compute_spectral_features(states[:, i])
            features.extend([spec_feats['dom_freq'], spec_feats['centroid'], spec_feats['power']])

        # Complexity measures
        for i in [0, 1]:
            features.append(self.compute_sample_entropy(states[:, i]))

        # Ensure fixed length (pad or trim)
        fv = np.array(features[:20], dtype=np.float32)
        if fv.shape[0] < 20:
            fv = np.pad(fv, (0, 20 - fv.shape[0]), 'constant')
        return fv

    def compare_trajectory_sets(self, traj_set1, traj_set2, labels1=None, labels2=None):
        if labels1 is None:
            labels1 = ['set1'] * len(traj_set1)
        if labels2 is None:
            labels2 = ['set2'] * len(traj_set2)

        results = {}

        # Feature-based comparison
        features1 = [self.trajectory_to_feature_vector(traj) for traj in traj_set1 if len(traj) >= 10]
        features2 = [self.trajectory_to_feature_vector(traj) for traj in traj_set2 if len(traj) >= 10]

        if len(features1) > 0 and len(features2) > 0:
            features1 = np.array(features1)
            features2 = np.array(features2)

            mean_feat1 = np.mean(features1, axis=0)
            mean_feat2 = np.mean(features2, axis=0)
            results['feature_euclidean'] = float(np.linalg.norm(mean_feat1 - mean_feat2))
            # handle cosine with potential zero vectors
            try:
                results['feature_cosine'] = float(cosine(mean_feat1, mean_feat2))
            except Exception:
                results['feature_cosine'] = 0.0
            results['feature_std_ratio'] = float(np.mean(np.std(features1, axis=0) / (np.std(features2, axis=0) + 1e-12)))

            # PCA on combined features
            try:
                scaler = StandardScaler()
                all_feats = np.vstack([features1, features2])
                all_scaled = scaler.fit_transform(all_feats)
                pca = PCA(n_components=min(self.pca_components, all_scaled.shape[1]))
                pca.fit(all_scaled)
                comps = pca.components_
                exp_var = pca.explained_variance_ratio_.tolist()
                results['pca_explained_variance'] = exp_var
                # compute projection distances between means on principal axes
                proj1 = pca.transform(scaler.transform(features1))
                proj2 = pca.transform(scaler.transform(features2))
                results['pca_mean_distance'] = float(np.linalg.norm(proj1.mean(axis=0) - proj2.mean(axis=0)))
                # principal axes similarity: cosine between first components
                try:
                    results['pca_first_axis_cosine'] = float(cosine(comps[0], comps[0])) if comps.shape[0] > 0 else 0.0
                except Exception:
                    results['pca_first_axis_cosine'] = 0.0
                results['pca_components'] = comps.tolist()
            except Exception as e:
                print(f"PCA comparison error: {e}")

            # t-SNE embedding difference (small)
            try:
                tsne = TSNE(n_components=min(2, self.tsne_components), init='random', learning_rate='auto', perplexity=30)
                # combine a small balanced subset to save time
                n1 = min(50, len(features1)); n2 = min(50, len(features2))
                sub = np.vstack([features1[:n1], features2[:n2]])
                emb = tsne.fit_transform(sub)
                emb1 = emb[:n1]; emb2 = emb[n1:]
                results['tsne_mean_distance'] = float(np.linalg.norm(emb1.mean(axis=0) - emb2.mean(axis=0)))
            except Exception as e:
                results['tsne_mean_distance'] = 0.0

        return results

# ====================== EVALUATION AND VISUALIZATION ======================

def save_metrics_history(metrics_history, out_dir="runs", name="metrics_history.json"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / name, "w") as f:
        json.dump(metrics_history, f, indent=2, default=_json_serial)

def plot_training_metrics(metrics_history, keys=("mean_combined_reward","mean_D","mean_Ploss","mean_Vloss","mean_disc_loss")):
    n = len(metrics_history.get("iter", []))
    if n == 0:
        print("No metrics to plot.")
        return
    iters = metrics_history["iter"]
    plt.figure(figsize=(10,6))
    for k in keys:
        if k in metrics_history:
            plt.plot(iters, metrics_history[k], label=k)
    plt.xlabel("iteration")
    plt.ylabel("value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title("Training metrics over iterations")
    plt.show()

def load_best_model(checkpoint_dir="checkpoints_stable", obs_dim=10, act_dim=2):
    policy_path = Path(checkpoint_dir) / "policy_best.pt"
    disc_path = Path(checkpoint_dir) / "disc_best.pt"

    if not policy_path.exists():
        raise FileNotFoundError(f"Best policy model not found at {policy_path}")

    policy = PolicyValue(obs_dim, act_dim).to(DEVICE)
    policy.load_state_dict(torch.load(policy_path, map_location=DEVICE))
    policy.eval()

    disc = None
    if disc_path.exists():
        disc = Discriminator(obs_dim, act_dim).to(DEVICE)
        disc.load_state_dict(torch.load(disc_path, map_location=DEVICE))
        disc.eval()

    print(f"Loaded best model from {policy_path}")

    # Try to load metrics
    metrics_path = Path(checkpoint_dir) / "metrics_best.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            if "best_combined_reward" in metrics and len(metrics["best_combined_reward"]) > 0:
                best_reward = metrics["best_combined_reward"][-1]
                best_iter = metrics["iter"][-1] if "iter" in metrics else "unknown"
                print(f"Best model performance: reward {best_reward:.4f} at iteration {best_iter}")
        except Exception as e:
            print(f"Could not load best metrics: {e}")

    return policy, disc

def collect_agent_trajectories(policy, env, n_episodes=10, max_steps=15000):
    trajs = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        states = []
        done = False
        steps = 0
        while not done and steps < max_steps:
            obs_t = to_torch(obs[None,:])
            with torch.no_grad():
                mu, _ = policy(obs_t)
                act = mu.cpu().numpy()[0]
            obs, _, terminated, truncated, _ = env.step(act)
            states.append(env.state.copy())
            done = terminated or truncated
            steps += 1
        if len(states) > 0:
            trajs.append(np.array(states, dtype=np.float32))

    if trajs:
        traj_lengths = [len(traj) for traj in trajs]
        print(f"Collected {len(trajs)} agent trajectories:")
        print(f"  Lengths: min={min(traj_lengths)}, max={max(traj_lengths)}, mean={np.mean(traj_lengths):.1f}")
    else:
        print("No agent trajectories collected.")

    return trajs

def path_length(theta_phi):
    if theta_phi.shape[0] < 2:
        return 0.0
    diffs = np.diff(theta_phi, axis=0)
    steps = np.sqrt((diffs**2).sum(axis=1))
    return float(steps.sum())

def sway_area_convex_hull(theta_phi):
    if theta_phi.shape[0] < 3:
        return 0.0
    try:
        hull = ConvexHull(theta_phi)
        return float(hull.volume)
    except Exception:
        return 0.0

def traj_basic_moments(states):
    out = {}
    for i, name in enumerate(['theta','phi','dtheta','dphi']):
        x = states[:,i].astype(np.float64)
        m = x.mean()
        v = x.var(ddof=0)
        if v <= 0:
            skew = 0.0; kurt = -3.0
        else:
            std = np.sqrt(v)
            m3 = np.mean((x - m)**3)
            m4 = np.mean((x - m)**4)
            skew = float(m3 / (std**3 + 1e-12))
            kurt = float(m4 / (std**4 + 1e-12) - 3.0)
        out[name] = {'mean':float(m), 'var':float(v), 'skew':skew, 'kurt':kurt}
    return out

def compute_reproducibility_metrics(list_of_states, dt=DT, verbose=True):
    valid_trajectories = [traj for traj in list_of_states if len(traj) >= 2]

    if len(valid_trajectories) < 2:
        print(f"Warning: Only {len(valid_trajectories)} valid trajectories")
        return {
            'n_traj': len(valid_trajectories),
            'combined_score': 0.0
        }

    n = len(valid_trajectories)

    try:
        # Calculate basic metrics
        path_lengths = [path_length(traj[:, :2]) for traj in valid_trajectories]
        sway_areas = [sway_area_convex_hull(traj[:, :2]) for traj in valid_trajectories]

        # Calculate feature variability
        moments_list = [traj_basic_moments(traj) for traj in valid_trajectories]

        # Simple combined score based on consistency
        path_length_cv = np.std(path_lengths) / (np.mean(path_lengths) + 1e-12)
        sway_area_cv = np.std(sway_areas) / (np.mean(sway_areas) + 1e-12)

        # Lower CV is better (more reproducible)
        combined_score = 1.0 / (1.0 + path_length_cv + sway_area_cv)

        out = {
            'n_traj': n,
            'combined_score': float(combined_score),
            'path_length_mean': float(np.mean(path_lengths)),
            'path_length_std': float(np.std(path_lengths)),
            'sway_area_mean': float(np.mean(sway_areas)),
            'sway_area_std': float(np.std(sway_areas))
        }

        if verbose:
            print(f"Reproducibility summary for {n} trajectories:")
            print(f"  Combined score: {combined_score:.4f}")
            print(f"  Path length: {np.mean(path_lengths):.3f} ± {np.std(path_lengths):.3f}")
            print(f"  Sway area: {np.mean(sway_areas):.6f} ± {np.std(sway_areas):.6f}")
        return out

    except Exception as e:
        print(f"Error computing reproducibility metrics: {e}")
        return {
            'n_traj': n,
            'combined_score': 0.0
        }

def eval_and_visualize(policy, test_trajs, n_episodes=6):
    env = HumanBalanceEnv(trajectories=test_trajs, enable_noise=True, enable_delay=True)
    rmses = []; dtws = []; survival_steps = []; lengths = []; areas = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        traj_agent_states = []
        done = False; steps = 0

        while not done and steps < 1000:
            obs_t = to_torch(obs[None,:])
            with torch.no_grad():
                mu, _ = policy(obs_t)
                act = mu.cpu().numpy()[0]
            obs, _, terminated, truncated, _ = env.step(act)
            traj_agent_states.append(env.state.copy())
            done = terminated or truncated
            steps += 1

        if len(traj_agent_states) == 0:
            continue

        traj_agent = np.array(traj_agent_states)
        expert_raw = random.choice(test_trajs)
        expert_states = positions_to_states(expert_raw)
        L = min(len(traj_agent), len(expert_states))
        agent_angles = traj_agent[:L,:2]; expert_angles = expert_states[:L,:2]

        rmse = float(np.sqrt(np.mean((agent_angles - expert_angles)**2)))
        dtw, _ = fastdtw(agent_angles, expert_angles, dist=euclidean)
        survival_steps.append(len(traj_agent))
        rmses.append(rmse); dtws.append(float(dtw)/max(1,L))
        lengths.append(path_length(agent_angles))
        areas.append(sway_area_convex_hull(agent_angles))

        if ep < 3:
            t = np.arange(L)*DT
            plt.figure(figsize=(10,4))
            plt.plot(t, expert_angles[:,0], label='expert θ')
            plt.plot(t, agent_angles[:,0], '--', label='agent θ')
            plt.plot(t, expert_angles[:,1], label='expert φ')
            plt.plot(t, agent_angles[:,1], '--', label='agent φ')
            plt.legend(); plt.xlabel("time [s]"); plt.title(f"Episode {ep+1} angles")
            plt.show()

    print("Evaluation summary:")
    if rmses:
        print(f"  Mean RMSE (angles): {np.mean(rmses):.6f}")
        print(f"  Mean DTW (angles): {np.mean(dtws):.6f}")
        print(f"  Avg survival steps: {np.mean(survival_steps):.2f}")
        print(f"  Mean path length: {np.mean(lengths):.6f}")
        print(f"  Mean sway area: {np.mean(areas):.6f}")

        plt.figure(figsize=(7,4))
        plt.hist(survival_steps, bins=10, edgecolor='k')
        plt.title("Agent survival steps")
        plt.show()

# ====================== NEW ADDITIONS ======================
# ====================== SURROGATE MODEL ======================
class SurrogateDynamics(nn.Module):
    """MLP that predicts next state (theta, phi, dtheta, dphi) from (obs, action)."""
    def __init__(self, obs_dim=10, act_dim=2, state_dim=4, hidden=(64, 64)):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, state_dim, hidden=hidden, activation=nn.ReLU, dropout=0.0)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)

def extract_expert_transitions_continuous(expert_states_seqs, dt=DT, max_ang_vel=MAX_ANG_VEL):
    """
    From expert state sequences (list of arrays [T,4]), create observation, continuous action,
    and next state tuples. Continuous actions are computed from inverse dynamics.
    Returns: (obs_array [N,10], actions_array [N,2], next_states_array [N,4])
    """
    obs_list, act_list, next_list = [], [], []
    for states in expert_states_seqs:
        if len(states) < 2:
            continue
        # Convert states to observations
        obs_seq = np.array([state_to_obs(s) for s in states], dtype=np.float32)
        # Estimate actions (continuous) using inverse dynamics
        for i in range(len(states) - 1):
            s = states[i]; s_next = states[i+1]
            θ, φ, dθ, dφ = s; θn, φn, dθn, dφn = s_next
            aθ = (dθn - dθ)/dt
            aφ = (dφn - dφ)/dt
            torque_theta = INERTIA * (aθ + DAMPING*dθ + (GRAVITY/LENGTH)*math.sin(θ))
            torque_phi = INERTIA * (aφ + DAMPING*dφ + (GRAVITY/LENGTH)*math.sin(φ))
            cont_action = np.array([np.clip(torque_theta / MAX_TORQUE, -1.0, 1.0),
                                    np.clip(torque_phi / MAX_TORQUE, -1.0, 1.0)], dtype=np.float32)
            obs_list.append(obs_seq[i])
            act_list.append(cont_action)
            next_list.append(s_next)
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32), np.array(next_list, dtype=np.float32)

def train_surrogate(model, obs, act, next_state, epochs=20, batch_size=128, lr=1e-3, device=DEVICE):
    """Train surrogate dynamics model on provided transitions."""
    dataset = TensorDataset(to_torch(obs, device), to_torch(act, device), to_torch(next_state, device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for o, a, n in loader:
            pred = model(o, a)
            loss = loss_fn(pred, n)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  Surrogate epoch {epoch+1}/{epochs}, loss: {total_loss/len(loader):.6f}")
    return model

def evaluate_surrogate(model, obs, act, next_state, device=DEVICE):
    """Return MSE between predicted and true next states."""
    model.eval()
    with torch.no_grad():
        pred = model(to_torch(obs, device), to_torch(act, device))
        mse = nn.MSELoss()(pred, to_torch(next_state, device)).item()
    return mse

# ====================== HUMAN-AGENT COMPARISON METRICS ======================
def compute_kinematic_metrics(states_seq, dt):
    """Kinematic metrics from a trajectory: sway (RMS angle), angular velocity profiles, path length."""
    if len(states_seq) < 2:
        return {}
    thetas = states_seq[:, 0]
    phis   = states_seq[:, 1]
    dthetas = states_seq[:, 2]
    dphis   = states_seq[:, 3]
    rms_theta = float(np.sqrt(np.mean(thetas**2)))
    rms_phi   = float(np.sqrt(np.mean(phis**2)))
    avg_vel_theta = float(np.mean(np.abs(dthetas)))
    avg_vel_phi   = float(np.mean(np.abs(dphis)))
    path_len = float(np.sum(np.sqrt(np.diff(thetas)**2 + np.diff(phis)**2)))
    return {
        'rms_theta': rms_theta, 'rms_phi': rms_phi,
        'avg_ang_vel_theta': avg_vel_theta, 'avg_ang_vel_phi': avg_vel_phi,
        'path_length': path_len
    }

def compute_control_metrics(actions_seq):
    """Action magnitude, smoothness, control effort."""
    if len(actions_seq) < 1:
        return {}
    mag = np.linalg.norm(actions_seq, axis=1)
    smoothness = np.mean(np.linalg.norm(np.diff(actions_seq, axis=0), axis=1)) if len(actions_seq) > 1 else 0.0
    effort = np.sum(mag**2)
    return {
        'action_magnitude_mean': float(np.mean(mag)),
        'action_smoothness': float(smoothness),
        'control_effort': float(effort)
    }

def compute_statistical_metrics(expert_list, agent_list):
    """Mean, var, RMSE between expert and agent lists of metrics across trajectories."""
    # expert_list and agent_list are lists of metric dictionaries (one per trajectory)
    # aggregate means per metric and compute RMSE per metric over trajectories
    all_keys = set()
    for d in expert_list + agent_list:
        all_keys.update(d.keys())
    stats = {}
    for key in all_keys:
        expert_vals = [d.get(key, np.nan) for d in expert_list]
        agent_vals  = [d.get(key, np.nan) for d in agent_list]
        expert_vals = np.array(expert_vals, dtype=np.float64)
        agent_vals  = np.array(agent_vals, dtype=np.float64)
        valid_ex = ~np.isnan(expert_vals)
        valid_ag = ~np.isnan(agent_vals)
        if valid_ex.sum() == 0 or valid_ag.sum() == 0:
            continue
        stats[key] = {
            'expert_mean': float(np.mean(expert_vals[valid_ex])),
            'expert_std':  float(np.std(expert_vals[valid_ex])),
            'agent_mean':  float(np.mean(agent_vals[valid_ag])),
            'agent_std':   float(np.std(agent_vals[valid_ag])),
            'rmse':        float(np.sqrt(np.mean((expert_vals[valid_ex] - agent_vals[valid_ag])**2)))
        }
    return stats

def compute_information_metrics(actions_list):
    """Entropy of action distribution (histogram-based)."""
    if len(actions_list) == 0:
        return {}
    all_acts = np.vstack(actions_list)
    # Joint histogram 2D
    hist, _, _ = np.histogram2d(all_acts[:,0], all_acts[:,1], bins=10, range=[[-1,1],[-1,1]])
    prob = hist / (hist.sum() + 1e-12)
    entropy = -np.sum(prob * np.log(prob + 1e-12))
    return {'action_entropy': float(entropy)}

def compute_temporal_metrics(states_seq, dt):
    """Autocorrelation of theta/phi and oscillation measure (damping)."""
    if len(states_seq) < 2:
        return {}
    def autocorr(x, lag=1):
        x = x - np.mean(x)
        return np.corrcoef(x[:-lag], x[lag:])[0,1] if len(x) > lag else 0.0

    ac_theta = autocorr(states_seq[:,0], 1)
    ac_phi   = autocorr(states_seq[:,1], 1)
    # Damping: fit exponential to peaks? simple ratio of successive peak amplitudes
    # We'll approximate by how quickly the amplitude decays.
    return {'autocorr_theta_lag1': float(ac_theta), 'autocorr_phi_lag1': float(ac_phi)}

def comprehensive_comparison(expert_states_list, agent_states_list, expert_actions_list, agent_actions_list, dt):
    """
    Compute per-trajectory metrics and aggregate comparisons.
    expert_states_list: list of arrays [T,4]
    agent_states_list: list of arrays [T,4]
    expert_actions_list: list of arrays [T,2] (continuous)
    agent_actions_list: list of arrays [T,2] (continuous)
    Returns a dictionary with all metrics.
    """
    expert_kinematic = [compute_kinematic_metrics(s, dt) for s in expert_states_list]
    agent_kinematic  = [compute_kinematic_metrics(s, dt) for s in agent_states_list]
    expert_control = [compute_control_metrics(a) for a in expert_actions_list]
    agent_control  = [compute_control_metrics(a) for a in agent_actions_list]
    expert_temporal = [compute_temporal_metrics(s, dt) for s in expert_states_list]
    agent_temporal  = [compute_temporal_metrics(s, dt) for s in agent_states_list]

    all_expert_metrics = [{**k, **c, **t} for k,c,t in zip(expert_kinematic, expert_control, expert_temporal)]
    all_agent_metrics  = [{**k, **c, **t} for k,c,t in zip(agent_kinematic, agent_control, agent_temporal)]

    stats = compute_statistical_metrics(all_expert_metrics, all_agent_metrics)
    info_expert = compute_information_metrics(expert_actions_list)
    info_agent  = compute_information_metrics(agent_actions_list)

    return {
        'statistical_comparison': stats,
        'expert_action_entropy': info_expert.get('action_entropy', 0.0),
        'agent_action_entropy': info_agent.get('action_entropy', 0.0)
    }

# ====================== VISUALIZATION ======================
def plot_trajectory_overlays(expert_states_list, agent_states_list, dt, save_dir, max_plots=4):
    """Overlay expert and agent angle time series for a few episodes."""
    os.makedirs(save_dir, exist_ok=True)
    for i, (exp_s, ag_s) in enumerate(zip(expert_states_list[:max_plots], agent_states_list[:max_plots])):
        plt.figure(figsize=(10,4))
        t_exp = np.arange(len(exp_s)) * dt
        t_ag  = np.arange(len(ag_s)) * dt
        plt.plot(t_exp, exp_s[:,0], label='expert θ', alpha=0.7)
        plt.plot(t_exp, exp_s[:,1], label='expert φ', alpha=0.7)
        plt.plot(t_ag,  ag_s[:,0], '--', label='agent θ', alpha=0.7)
        plt.plot(t_ag,  ag_s[:,1], '--', label='agent φ', alpha=0.7)
        plt.title(f'Trajectory overlay {i+1}')
        plt.xlabel('Time [s]'); plt.ylabel('Angle [rad]')
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'trajectory_overlay_{i+1}.png'), dpi=150)
        plt.close()

def plot_phase_portraits(expert_states_list, agent_states_list, save_dir, max_plots=4):
    """Phase plots: θ vs dθ, φ vs dφ."""
    os.makedirs(save_dir, exist_ok=True)
    for i, (exp_s, ag_s) in enumerate(zip(expert_states_list[:max_plots], agent_states_list[:max_plots])):
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
        ax1.plot(exp_s[:,0], exp_s[:,2], label='expert')
        ax1.plot(ag_s[:,0],  ag_s[:,2],  '--', label='agent')
        ax1.set_title(f'θ phase portrait {i+1}'); ax1.set_xlabel('θ [rad]'); ax1.set_ylabel('dθ [rad/s]')
        ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(exp_s[:,1], exp_s[:,3], label='expert')
        ax2.plot(ag_s[:,1],  ag_s[:,3],  '--', label='agent')
        ax2.set_title(f'φ phase portrait {i+1}'); ax2.set_xlabel('φ [rad]'); ax2.set_ylabel('dφ [rad/s]')
        ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'phase_portrait_{i+1}.png'), dpi=150)
        plt.close()

def plot_action_distributions(expert_actions_list, agent_actions_list, save_dir):
    """Histograms of action components and magnitude."""
    if not expert_actions_list or not agent_actions_list:
        return
    expert_acts = np.vstack(expert_actions_list)
    agent_acts  = np.vstack(agent_actions_list)
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    axes[0].hist(expert_acts[:,0], bins=30, alpha=0.5, density=True, label='expert')
    axes[0].hist(agent_acts[:,0], bins=30, alpha=0.5, density=True, label='agent')
    axes[0].set_title('Action θ component'); axes[0].legend()
    axes[1].hist(expert_acts[:,1], bins=30, alpha=0.5, density=True, label='expert')
    axes[1].hist(agent_acts[:,1], bins=30, alpha=0.5, density=True, label='agent')
    axes[1].set_title('Action φ component'); axes[1].legend()
    mag_exp = np.linalg.norm(expert_acts, axis=1)
    mag_ag  = np.linalg.norm(agent_acts, axis=1)
    axes[2].hist(mag_exp, bins=30, alpha=0.5, density=True, label='expert')
    axes[2].hist(mag_ag, bins=30, alpha=0.5, density=True, label='agent')
    axes[2].set_title('Action magnitude'); axes[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'action_distributions.png'), dpi=150)
    plt.close()

def plot_metric_comparison_bar(stats_dict, save_dir):
    """Bar chart comparing expert vs agent mean for each metric with RMSE."""
    if not stats_dict:
        return
    metrics = list(stats_dict.keys())
    expert_means = [stats_dict[m]['expert_mean'] for m in metrics]
    agent_means  = [stats_dict[m]['agent_mean'] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(metrics)*0.8), 5))
    bars1 = ax.bar(x - width/2, expert_means, width, label='Expert')
    bars2 = ax.bar(x + width/2, agent_means, width, label='Agent')
    ax.set_ylabel('Mean value')
    ax.set_title('Expert vs Agent metric means')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metric_comparison_bar.png'), dpi=150)
    plt.close()

def generate_comparison_visualizations(expert_states_list, agent_states_list, expert_actions_list, agent_actions_list, dt, save_dir="human_agent_comparison"):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Generating visualizations in '{save_dir}'...")
    plot_trajectory_overlays(expert_states_list, agent_states_list, dt, save_dir)
    plot_phase_portraits(expert_states_list, agent_states_list, save_dir)
    plot_action_distributions(expert_actions_list, agent_actions_list, save_dir)
    # Metric bar chart will be generated inside comprehensive_comparison call after we have stats
    print("Visualizations saved.")

# ====================== AUXILIARY FUNCTION: COLLECT AGENT SEQUENCES WITH ACTIONS ======================
def collect_agent_sequences(policy, env, n_episodes=30, max_steps=15000):
    """
    Collect agent episodes returning states, observations, and continuous actions.
    Returns: list of dicts with keys 'states', 'obs', 'actions'
    """
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        states = []
        obs_seq = []
        actions = []
        done = False
        steps = 0
        while not done and steps < max_steps:
            obs_t = to_torch(obs[None,:])
            with torch.no_grad():
                mu, _ = policy(obs_t)
                act = mu.cpu().numpy()[0]   # continuous action [-1,1]^2
            next_obs, _, terminated, truncated, _ = env.step(act)
            states.append(env.state.copy())
            obs_seq.append(obs.copy())
            actions.append(act.copy())
            obs = next_obs
            done = terminated or truncated
            steps += 1
        if len(states) > 0:
            episodes.append({
                'states': np.array(states, dtype=np.float32),
                'obs': np.array(obs_seq, dtype=np.float32),
                'actions': np.array(actions, dtype=np.float32)
            })
    return episodes

# ====================== HUMAN-AGENT ANALYSIS FUNCTION ======================
def add_human_agent_analysis(policy, expert_vis_seqs, train_trajs):
    """
    Performs surrogate model training, human-agent comparison, and visualization.
    Called from main() after training and basic evaluation.
    """
    print("\n===== SURROGATE MODEL & HUMAN-AGENT COMPARISON =====")

    # 1. Prepare expert continuous transitions
    print("Preparing expert continuous transitions...")
    expert_obs_cont, expert_acts_cont, expert_nexts = extract_expert_transitions_continuous(expert_vis_seqs, DT)
    print(f"Expert transitions: {expert_obs_cont.shape[0]} samples")

    # 2. Train surrogate dynamics model
    print("Training surrogate dynamics model on expert data...")
    surrogate = SurrogateDynamics(obs_dim=10, act_dim=2).to(DEVICE)
    surrogate = train_surrogate(surrogate, expert_obs_cont, expert_acts_cont, expert_nexts, epochs=30, lr=1e-3)

    # 3. Collect agent sequences
    print("Collecting agent rollouts for analysis...")
    env_analysis = HumanBalanceEnv(
        trajectories=train_trajs,   # same structure as training
        enable_noise=True, enable_delay=True,
        discrete_actions=False,     # continuous actions for analysis
        action_threshold=0.05,
        max_steps=MAX_STEPS_EVALUATION
    )
    agent_episodes = collect_agent_sequences(policy, env_analysis, n_episodes=30)

    # Extract agent sequences
    agent_states = [ep['states'] for ep in agent_episodes]
    agent_actions = [ep['actions'] for ep in agent_episodes]

    # 4. Evaluate surrogate on agent data
    print("Evaluating surrogate on agent data...")
    # Build agent transitions from episodes
    agent_obs_list, agent_act_list, agent_next_list = [], [], []
    for ep in agent_episodes:
        obs_seq = ep['obs']      # [T,10]
        act_seq = ep['actions']  # [T,2]
        state_seq = ep['states'] # [T,4]
        for i in range(len(obs_seq)-1):
            agent_obs_list.append(obs_seq[i])
            agent_act_list.append(act_seq[i])
            agent_next_list.append(state_seq[i+1])
    if agent_obs_list:
        agent_obs_arr = np.array(agent_obs_list, dtype=np.float32)
        agent_act_arr = np.array(agent_act_list, dtype=np.float32)
        agent_next_arr = np.array(agent_next_list, dtype=np.float32)
        agent_mse = evaluate_surrogate(surrogate, agent_obs_arr, agent_act_arr, agent_next_arr)
        expert_mse = evaluate_surrogate(surrogate, expert_obs_cont, expert_acts_cont, expert_nexts)
        print(f"Surrogate MSE: expert = {expert_mse:.6f}, agent = {agent_mse:.6f}")
        # Ratio: higher agent MSE may indicate less human-like dynamics
        print(f"Agent/expert MSE ratio: {agent_mse/expert_mse:.3f}" if expert_mse > 0 else "infinite")
    else:
        print("No agent transitions collected for surrogate evaluation.")

    # 5. Prepare expert state sequences for metrics (use the same expert_vis_seqs)
    expert_states_for_metrics = expert_vis_seqs   # list of [T,4] states
    # Extract continuous expert actions per trajectory for metrics
    expert_actions_for_metrics = []
    for states in expert_states_for_metrics:
        if len(states) < 2: continue
        acts_traj = []
        for i in range(len(states)-1):
            s, s_next = states[i], states[i+1]
            θ, φ, dθ, dφ = s; θn, φn, dθn, dφn = s_next
            aθ = (dθn - dθ)/DT
            aφ = (dφn - dφ)/DT
            torque_theta = INERTIA * (aθ + DAMPING*dθ + (GRAVITY/LENGTH)*math.sin(θ))
            torque_phi = INERTIA * (aφ + DAMPING*dφ + (GRAVITY/LENGTH)*math.sin(φ))
            acts_traj.append([np.clip(torque_theta/MAX_TORQUE, -1,1), np.clip(torque_phi/MAX_TORQUE, -1,1)])
        if acts_traj:
            expert_actions_for_metrics.append(np.array(acts_traj, dtype=np.float32))
    # Keep only trajectories that have actions
    expert_states_trimmed = [expert_states_for_metrics[i] for i in range(len(expert_states_for_metrics)) if i < len(expert_actions_for_metrics) and len(expert_actions_for_metrics[i])>0]
    expert_actions_trimmed = expert_actions_for_metrics

    # 6. Compute comprehensive comparison metrics
    print("Computing human-agent comparison metrics...")
    comp = comprehensive_comparison(expert_states_trimmed, agent_states, expert_actions_trimmed, agent_actions, DT)
    print("Statistical comparison:")
    for metric, vals in comp['statistical_comparison'].items():
        print(f"  {metric}: expert={vals['expert_mean']:.4f}±{vals['expert_std']:.4f}, agent={vals['agent_mean']:.4f}±{vals['agent_std']:.4f}, RMSE={vals['rmse']:.4f}")
    print(f"Expert action entropy: {comp['expert_action_entropy']:.4f}, Agent action entropy: {comp['agent_action_entropy']:.4f}")

    # Save metrics to JSON
    out_dir = Path("analysis_results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "human_agent_comparison_metrics.json", "w") as f:
        json.dump(comp, f, indent=2, default=_json_serial)

    # 7. Generate visualizations
    viz_dir = "human_agent_comparison"
    generate_comparison_visualizations(expert_states_trimmed, agent_states, expert_actions_trimmed, agent_actions, DT, save_dir=viz_dir)
    # Add metric bar chart
    if comp['statistical_comparison']:
        plot_metric_comparison_bar(comp['statistical_comparison'], viz_dir)

    print("Human-agent analysis complete.")
    return surrogate, comp

# ====================== MAIN EXECUTION ======================

def main():
    """Main execution function."""
    # Set seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        print("=== Human Balance GAIL Training ===")
        print(f"Device: {DEVICE}")
        print(f"Seed: {SEED}")

        # Load and prepare data
        print("\n1. Loading trajectories...")
        all_trajs, labels = load_trajectories(CSV_PATH, GROUP_COL, TIME_COL, X_COL, Y_COL)

        # Shuffle and split
        combined = list(zip(all_trajs, labels))
        random.shuffle(combined)
        all_trajs, labels = zip(*combined)
        all_trajs = list(all_trajs)
        labels = list(labels)

        n_test = max(1, int(len(all_trajs) * 0.2))
        train_trajs = all_trajs[:-n_test]
        test_trajs = all_trajs[-n_test:]
        print(f"Split data into {len(train_trajs)} train and {len(test_trajs)} test trajectories.")

        # Build expert dataset
        print("\n2. Building expert dataset...")
        expert_obs, expert_acts, expert_vis_seqs = build_expert_dataset(train_trajs, expert_threshold=0.05)

        # Create environment
        print("\n3. Creating environment...")
        env = HumanBalanceEnv(
            trajectories=train_trajs,
            enable_noise=True,
            enable_delay=True,
            discrete_actions=True,
            action_threshold=0.05,
            survival_bonus=SURVIVAL_BONUS,
            angle_reward_scale=ANGLE_REWARD_SCALE,
            angle_reward_sigma=ANGLE_REWARD_SIGMA,
            vel_penalty_weight=VEL_PENALTY_WEIGHT,
            torque_penalty=TORQUE_PENALTY,
            action_change_penalty=ACTION_CHANGE_PENALTY,
            max_steps=MAX_STEPS_TRAINING_INITIAL
        )

        # Train with stabilized version
        print("\n4. Starting stabilized GAIL training...")
        trained_policy, trained_disc, metrics_history = train_gail_stable(
            env, expert_obs, expert_acts, expert_vis_seqs,
            num_iters=TOTAL_ITERS,
            steps_per_iter=STEPS_PER_ITER,
            return_metrics=True,
            save_every=50,
            checkpoint_dir="checkpoints_stable"
        )

        # Save and plot training metrics
        save_metrics_history(metrics_history, out_dir="runs_stable", name="metrics_history_final.json")
        plot_training_metrics(metrics_history)

        # Load best model for evaluation
        print("\n5. Loading best model for evaluation...")
        try:
            best_policy, _ = load_best_model("checkpoints_stable", expert_obs.shape[1], expert_acts.shape[1])
            print("Using BEST model for evaluation")
        except FileNotFoundError as e:
            print(f"Best model not found: {e}, using final trained model")
            best_policy = trained_policy

        # Compute reproducibility metrics (expert subset)
        print("\n6. Computing reproducibility metrics...")
        expert_subset = expert_vis_seqs[:min(50, len(expert_vis_seqs))]
        expert_repro = compute_reproducibility_metrics(expert_subset)

        print("\n7. Collecting agent trajectories...")
        eval_env = HumanBalanceEnv(
            trajectories=train_trajs,
            enable_noise=True,
            enable_delay=True,
            discrete_actions=False,
            action_threshold=0.05,
            max_steps=MAX_STEPS_EVALUATION
        )
        agent_trajs = collect_agent_trajectories(best_policy, eval_env, n_episodes=30)

        agent_repro = compute_reproducibility_metrics(agent_trajs)

        print("\nComparison (expert vs agent):")
        print(f"  Expert combined score: {expert_repro.get('combined_score', np.nan):.4f}")
        print(f"  Agent  combined score: {agent_repro.get('combined_score', np.nan):.4f}")

        # ====================== 8. Save trajectories to CSV ======================
        print("\n8. Saving trajectories to CSV files...")

        # Create output directory for trajectories
        trajectory_dir = "trajectory_data"
        Path(trajectory_dir).mkdir(exist_ok=True)

        # Save expert trajectories
        expert_csv_path = os.path.join(trajectory_dir, "expert_trajectories.csv")
        save_expert_trajectories_csv(train_trajs, labels[:len(train_trajs)], expert_csv_path)

        # Save agent trajectories
        agent_csv_path = os.path.join(trajectory_dir, "agent_trajectories.csv")
        save_agent_trajectories_csv(agent_trajs, agent_csv_path)

        # Save comparison file
        comparison_csv_path = os.path.join(trajectory_dir, "comparison_trajectories.csv")
        save_comparison_trajectories_csv(train_trajs, labels[:len(train_trajs)], agent_trajs, comparison_csv_path)

        print(f"✅ All trajectories saved to {trajectory_dir}/ directory")

        # ====================== 9. Enhanced comparison metrics (spectrum, PCA, t-SNE) ======================
        print("\n9. Computing enhanced comparison metrics (spectrum/PCA/t-SNE)...")
        metrics_calculator = EnhancedTrajectoryMetrics()
        enhanced_metrics = metrics_calculator.compare_trajectory_sets(expert_subset, agent_trajs)

        # Additionally compute aggregated spectral comparisons for theta and phi
        def aggregate_spectral_stats(traj_set):
            doms = []; cents = []; powers = []
            for traj in traj_set:
                if len(traj) < 4:
                    continue
                sf_theta = metrics_calculator.compute_spectral_features(traj[:,0])
                sf_phi = metrics_calculator.compute_spectral_features(traj[:,1])
                # use theta and phi centroid/power averages
                if 'centroid' in sf_theta and sf_theta['centroid'] != 0:
                    doms.append(sf_theta['dom_freq'])
                    cents.append(sf_theta['centroid'])
                    powers.append(sf_theta['power'])
                if 'centroid' in sf_phi and sf_phi['centroid'] != 0:
                    doms.append(sf_phi['dom_freq'])
                    cents.append(sf_phi['centroid'])
                    powers.append(sf_phi['power'])
            return {
                'dom_freq_mean': float(np.mean(doms)) if doms else 0.0,
                'centroid_mean': float(np.mean(cents)) if cents else 0.0,
                'power_mean': float(np.mean(powers)) if powers else 0.0
            }

        spec_ex = aggregate_spectral_stats(expert_subset)
        spec_ag = aggregate_spectral_stats(agent_trajs)
        enhanced_metrics['expert_spectral'] = spec_ex
        enhanced_metrics['agent_spectral'] = spec_ag

        print("Enhanced metrics:")
        for k, v in enhanced_metrics.items():
            try:
                print(f"  {k}: {v}")
            except Exception:
                print(f"  {k}: <complex>")

        # Save enhanced metrics
        Path("analysis_results").mkdir(exist_ok=True)
        with open("analysis_results/enhanced_comparison_metrics.json", "w") as f:
            json.dump(enhanced_metrics, f, indent=2, default=_json_serial)

        # Also save feature vectors and PCA results for offline inspection
        def compute_feature_matrix(trajs):
            fv = []
            for t in trajs:
                if len(t) < 10:
                    continue
                fv.append(metrics_calculator.trajectory_to_feature_vector(t))
            if len(fv) == 0:
                return np.zeros((0,20))
            return np.vstack(fv)

        feat_ex = compute_feature_matrix(expert_subset)
        feat_ag = compute_feature_matrix(agent_trajs)
        np.save("analysis_results/features_expert.npy", feat_ex)
        np.save("analysis_results/features_agent.npy", feat_ag)

        # PCA summary
        try:
            if feat_ex.shape[0] > 0 and feat_ag.shape[0] > 0:
                scaler = StandardScaler()
                combined_feats = np.vstack([feat_ex, feat_ag])
                combined_scaled = scaler.fit_transform(combined_feats)
                pca = PCA(n_components=min(6, combined_scaled.shape[1]))
                proj = pca.fit_transform(combined_scaled)
                n_ex = feat_ex.shape[0]
                proj_ex = proj[:n_ex]
                proj_ag = proj[n_ex:]
                pca_summary = {
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'pca_mean_distance': float(np.linalg.norm(proj_ex.mean(axis=0) - proj_ag.mean(axis=0)))
                }
                with open("analysis_results/pca_summary.json", "w") as f:
                    json.dump(pca_summary, f, indent=2, default=_json_serial)
                print("Saved PCA summary to analysis_results/pca_summary.json")
                # Plot first two PCA axes if available
                if proj.shape[1] >= 2:
                    plt.figure(figsize=(6,5))
                    plt.scatter(proj_ex[:,0], proj_ex[:,第1], label='expert', alpha=0.7)
                    plt.scatter(proj_ag[:,0], proj_ag[:,第1], label='agent', alpha=0.7)
                    plt.legend(); plt.title("PCA projection (first 2 components)")
                    plt.xlabel("PC1"); plt.ylabel("PC2")
                    plt.savefig("analysis_results/pca_scatter.png", dpi=150)
                    plt.close()
        except Exception as e:
            print(f"PCA save/plot error: {e}")

        # t-SNE summary (small)
        try:
            if feat_ex.shape[0] > 2 and feat_ag.shape[0] > 2:
                combined_small = np.vstack([feat_ex[:100], feat_ag[:100]])
                tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=30)
                emb = tsne.fit_transform(combined_small)
                n_ex = min(feat_ex.shape[0], 100)
                emb_ex = emb[:n_ex]; emb_ag = emb[n_ex:]
                np.save("analysis_results/tsne_emb.npy", emb)
                plt.figure(figsize=(6,5))
                plt.scatter(emb_ex[:,0], emb_ex[:,第1], label='expert', alpha=0.7)
                plt.scatter(emb_ag[:,0], emb_ag[:,第1], label='agent', alpha=0.7)
                plt.legend(); plt.title("t-SNE embedding (subset)")
                plt.savefig("analysis_results/tsne_scatter.png", dpi=150)
                plt.close()
        except Exception as e:
            print(f"t-SNE error: {e}")

        # Final evaluation
        print("\n10. Final evaluation on test trajectories...")
        eval_and_visualize(best_policy, test_trajs, n_episodes=6)

        # ====================== HUMAN-AGENT ANALYSIS (ADDITION) ======================
        surrogate_model, comparison_metrics = add_human_agent_analysis(best_policy, expert_vis_seqs, train_trajs)
        print(f"Surrogate model trained. Agent dynamics MSE: see above.")

        print("\n=== Training and Evaluation Complete ===")
        print("Results saved to:")
        print("  - checkpoints_stable/ (models and training logs)")
        print("  - runs_stable/ (metrics history)")
        print("  - trajectory_data/ (expert and agent trajectory CSV files)")
        print("  - analysis_results/ (spectral, PCA, t-SNE, enhanced metrics)")
        print("  - human_agent_comparison/ (trajectory overlays, phase plots, action distributions, metric bar chart)")
        print("  - analysis_results/ (human_agent_comparison_metrics.json)")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure the dataset file exists in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
