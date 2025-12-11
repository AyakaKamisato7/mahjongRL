import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import glob
from torch.distributions import Categorical
from config import MahjongConfig as Cfg

# 检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MahjongFeatureExtractor:
    """
    [特征工程层]
    负责将 Env 返回的 dict 转换为神经网络可吃的 Tensor
    """

    @staticmethod
    def encode_obs(obs_list):
        """
        批处理将 observation 列表转为 Tensor
        :param obs_list: List of dict (batch_size 个 obs)
        :return: (cnn_input, lstm_input, masks)
        """
        batch_size = len(obs_list)

        # --- 1. 构建 CNN 空间特征 (Batch, Channel, 34) ---
        # Channel 0: 手牌计数
        # Channel 1: 赖子指示 (哪些牌是赖子)
        # Channel 2: 副露 (碰/杠的牌)
        # Channel 3: 花牌赖子数 (全图填充)
        # Channel 4: 自身花牌数 (全图填充) - 辅助判断胡牌倍率

        spatial_features = np.zeros((batch_size, Cfg.CNN_CHANNEL_IN, 34), dtype=np.float32)

        # --- 2. 构建 LSTM 时序特征 (Batch, Seq_Len, Input_Dim) ---
        lstm_features = np.zeros((batch_size, Cfg.HISTORY_LEN, Cfg.LSTM_INPUT_DIM), dtype=np.float32)

        # --- 3. 动作掩码 ---
        masks = np.zeros((batch_size, Cfg.ACTION_DIM), dtype=np.float32)

        for i, obs in enumerate(obs_list):
            # [CNN] Channel 0: Hand
            spatial_features[i, 0, :] = obs['hand']

            # [CNN] Channel 1: Laizi Set
            for l_id in obs['laizi_set']:
                if l_id < 34:
                    spatial_features[i, 1, l_id] = 1.0

            # [CNN] Channel 2: Melds (简单的计数编码)
            for m_type, m_tile in obs['melds']:
                if m_tile < 34:
                    spatial_features[i, 2, m_tile] += 1.0

            # [CNN] Channel 3 & 4: Scalar info broadcast to map
            spatial_features[i, 3, :] = obs['flower_laizis']
            spatial_features[i, 4, :] = len(obs['flowers'])

            # [LSTM] Encoding
            # history 格式: [(rel_p, action), ...]
            for t, (rel_p, act) in enumerate(obs['history']):
                if rel_p == -1: continue  # Padding

                # One-hot encoding
                # 前 41 维是动作，后 4 维是玩家相对位置
                lstm_features[i, t, act] = 1.0
                lstm_features[i, t, 41 + rel_p] = 1.0

            # [Mask]
            masks[i, :] = obs['mask']

        return (
            torch.tensor(spatial_features).to(device),
            torch.tensor(lstm_features).to(device),
            torch.tensor(masks).to(device)
        )


class MahjongNet(nn.Module):
    """
    [神经网络模型]
    结构：CNN (空间) + LSTM (时序) -> Fusion -> Actor/Critic
    """

    def __init__(self):
        super(MahjongNet, self).__init__()

        # 1. CNN 部分 (提取牌型结构)
        # Input: (B, 5, 34) -> Output: (B, 64, 34) -> Flatten
        self.cnn = nn.Sequential(
            nn.Conv1d(Cfg.CNN_CHANNEL_IN, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, Cfg.CNN_CHANNEL_OUT, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # CNN Flatten 大小 = 64 * 34 = 2176
        self.cnn_out_dim = Cfg.CNN_CHANNEL_OUT * 34

        # 2. LSTM 部分 (提取出牌流向)
        self.lstm = nn.LSTM(
            input_size=Cfg.LSTM_INPUT_DIM,
            hidden_size=Cfg.LSTM_HIDDEN_DIM,
            num_layers=Cfg.LSTM_LAYERS,
            batch_first=True
        )

        # 3. 融合层
        self.fusion_dim = self.cnn_out_dim + Cfg.LSTM_HIDDEN_DIM
        self.fc = nn.Sequential(
            nn.Linear(self.fusion_dim, Cfg.FC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1)  # 防止过拟合
        )

        # 4. Heads
        self.actor = nn.Linear(Cfg.FC_HIDDEN_DIM, Cfg.ACTION_DIM)  # Policy Logits
        self.critic = nn.Linear(Cfg.FC_HIDDEN_DIM, 1)  # State Value

    def forward(self, cnn_input, lstm_input):
        # CNN Forward
        x_cnn = self.cnn(cnn_input)

        # LSTM Forward
        # output: (Batch, Seq, Hidden), h_n: (Layers, Batch, Hidden)
        # 我们只取最后一个时间步的 hidden state
        _, (h_n, _) = self.lstm(lstm_input)
        x_lstm = h_n[-1]  # 取最后一层的 hidden state

        # Fusion
        x_combined = torch.cat([x_cnn, x_lstm], dim=1)
        x_features = self.fc(x_combined)

        return self.actor(x_features), self.critic(x_features)


class PPOAgent:
    """
    [智能体] PPO 算法实现 + 自博弈管理
    """

    def __init__(self, run_name="run_01", output_dir=None, load_path=None):
        self.policy = MahjongNet().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=Cfg.LR_START)
        self.policy_old = MahjongNet().to(device)  # PPO 需要旧策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # 训练状态管理
        self.run_name = run_name

        # [修改] 如果外部指定了 output_dir，就用外部的；否则默认 ./checkpoints
        if output_dir:
            self.checkpoint_dir = output_dir
        else:
            self.checkpoint_dir = f"./checkpoints/{run_name}"

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 记忆池 (Memory)
        self.buffer = []

        # 自博弈对手池
        self.opponent_pool = []
        self.update_step_counter = 0

        # 加载旧模型 (断点续训)
        if load_path:
            self.load_model(load_path)

    def select_action(self, obs, eval_mode=False):
        """
        选择动作
        :param obs: 单个 observation 字典
        :param eval_mode: 如果 True，选择概率最大的动作 (Greedy)
        :return: action_id, log_prob, state_value
        """
        # 转 Batch=1 的 Tensor
        cnn_in, lstm_in, mask = MahjongFeatureExtractor.encode_obs([obs])

        with torch.no_grad():
            logits, value = self.policy_old(cnn_in, lstm_in)

            # --- Action Masking (关键) ---
            # 将非法动作的 logits 设为负无穷
            # mask 为 0 的位置是非法动作
            masked_logits = logits.clone()
            masked_logits[mask == 0] = -1e9

            # Softmax 归一化
            probs = F.softmax(masked_logits, dim=-1)

            # 建立分布
            dist = Categorical(probs)

            if eval_mode:
                action = torch.argmax(probs, dim=1)
            else:
                action = dist.sample()

            action_logprob = dist.log_prob(action)

        return action.item(), action_logprob.item(), value.item()

    def store_transition(self, transition):
        """存储一条轨迹 (s, a, log_prob, r, done, val, mask)"""
        self.buffer.append(transition)

    def update(self):
        """
        [PPO 核心更新] 使用 GPU 加速训练
        """
        # 1. 解压 buffer
        obs_list = [t[0] for t in self.buffer]
        actions = torch.tensor([t[1] for t in self.buffer], dtype=torch.long).to(device)
        old_logprobs = torch.tensor([t[2] for t in self.buffer], dtype=torch.float32).to(device)
        rewards = [t[3] for t in self.buffer]
        dones = [t[4] for t in self.buffer]
        old_values = torch.tensor([t[5] for t in self.buffer], dtype=torch.float32).to(device)

        # 2. 计算 Monte Carlo Returns & Advantages (GAE)
        returns = []
        discounted_reward = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done:
                discounted_reward = 0
            discounted_reward = reward + (Cfg.GAMMA * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # Normalize returns (加速收敛)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        advantages = returns - old_values.detach()

        # 3. 准备数据特征
        cnn_in, lstm_in, masks = MahjongFeatureExtractor.encode_obs(obs_list)

        # 4. PPO Update Epochs
        for _ in range(Cfg.EPOCHS):
            # 获取当前策略的评估
            logits, values = self.policy(cnn_in, lstm_in)
            values = values.squeeze()

            # Masking
            masked_logits = logits.clone()
            masked_logits[masks == 0] = -1e9
            probs = F.softmax(masked_logits, dim=-1)
            dist = Categorical(probs)

            logprobs = dist.log_prob(actions)
            dist_entropy = dist.entropy()

            # Ratio
            ratios = torch.exp(logprobs - old_logprobs)

            # Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - Cfg.EPS_CLIP, 1 + Cfg.EPS_CLIP) * advantages

            # Loss Function
            # Loss = -min(surr) + 0.5 * MSE(val, ret) - 0.01 * Entropy
            loss = -torch.min(surr1, surr2) + \
                   Cfg.VF_COEF * self.MseLoss(values, returns) - \
                   Cfg.ENTROPY_COEF * dist_entropy

            # Backprop
            self.optimizer.zero_grad()
            loss.mean().backward()
            # 梯度裁剪 (防止爆炸)
            nn.utils.clip_grad_norm_(self.policy.parameters(), Cfg.MAX_GRAD_NORM)
            self.optimizer.step()

        # 5. 更新旧策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        # 6. 自博弈：尝试将当前模型加入对手池
        self._update_opponent_pool()

    def _update_opponent_pool(self):
        """定期保存当前模型作为未来的对手"""
        self.update_step_counter += 1
        if self.update_step_counter % Cfg.UPDATE_OPPONENT_FREQ == 0:
            save_path = os.path.join(self.checkpoint_dir, f"opponent_{self.update_step_counter}.pth")
            self.save_model(save_path)
            self.opponent_pool.append(save_path)
            # 保持池子大小
            if len(self.opponent_pool) > Cfg.OPPONENT_POOL_SIZE:
                oldest = self.opponent_pool.pop(0)
                if os.path.exists(oldest): os.remove(oldest)
            print(f"[Self-Play] Added new model to opponent pool: {save_path}")

    def save_model(self, path=None):
        if path is None:
            path = os.path.join(self.checkpoint_dir, "latest_model.pth")
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.policy_old.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded model from {path}")
        else:
            print(f"No checkpoint found at {path}, starting fresh.")

    def get_opponent_agent(self):
        """
        [自博弈] 获取一个对手 Agent
        可以是当前最新的，也可以是历史版本 (Frozen Policy)
        """
        # 20% 概率打历史版本，80% 概率打当前版本
        if self.opponent_pool and np.random.rand() < 0.2:
            random_model_path = np.random.choice(self.opponent_pool)
            opponent = PPOAgent(run_name="opponent_temp")
            # 只加载权重，不加载优化器，因为对手不训练
            ckpt = torch.load(random_model_path, map_location=device)
            opponent.policy.load_state_dict(ckpt['model_state_dict'])
            opponent.policy_old.load_state_dict(ckpt['model_state_dict'])
            opponent.policy.eval()  # 设为评估模式
            return opponent
        else:
            # 返回自己 (共享参数，即 self-play against current policy)
            return self