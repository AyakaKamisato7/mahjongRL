import os
import torch
import numpy as np
import time
from collections import deque, Counter
from config import MahjongConfig as Cfg
from env import MahjongEnv, TileUtils
from agent import PPOAgent

WORK_DIR = r"D:/pyworksp/mahjongRL/"
SAVE_DIR = os.path.join(WORK_DIR, "pth")
REPLAY_DIR = os.path.join(WORK_DIR, "replays")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(REPLAY_DIR, exist_ok=True)


def save_game_replay(env, episode_id, winner_id, reward, final_hand_type):
    """
    [功能] 保存回放，输出每一轮所有 Agent 的手牌
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"replay_ep{episode_id}_{timestamp}_win{winner_id}.txt"
    filepath = os.path.join(REPLAY_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"=== Mahjong Replay [Episode {episode_id}] ===\n")
        f.write(f"Time: {timestamp} | Winner: Agent {winner_id} | Reward: {reward:.2f} | Type: {final_hand_type}\n")
        f.write(
            f"Laizi: {TileUtils.to_string(env.indicator_tile)} -> {[TileUtils.to_string(l) for l in env.laizi_set]}\n")
        f.write("=" * 60 + "\n\n")

        # 遍历历史记录
        # env.action_history 结构: [{'pid': int, 'action': int, 'snapshot': list}, ...]
        for step_i, record in enumerate(env.action_history):
            pid = record['pid']
            act = record['action']
            snapshot = record['snapshot']  # 4个玩家的 {hand, melds, flower_laizis}

            # 解析动作名称
            if act <= 33:
                act_str = f"Discard {TileUtils.to_string(act)}"
            else:
                act_map = {34: "PASS", 35: "HU", 36: "PON", 37: "GANG", 38: "CHI_L", 39: "CHI_M", 40: "CHI_R"}
                act_str = act_map.get(act, "UNKNOWN")

            f.write(f"--- Step {step_i:03d} | Agent {pid} performs: {act_str} ---\n")

            # 打印该时刻所有人的手牌
            for p_i in range(4):
                p_state = snapshot[p_i]

                # 转换手牌
                hand_str = []
                for t_id, count in enumerate(p_state['hand']):
                    if count > 0: hand_str.extend([TileUtils.to_string(t_id)] * count)
                if p_state['flower_laizis'] > 0:
                    hand_str.append(f"[花赖x{p_state['flower_laizis']}]")

                # 转换副露
                melds_str = [(m[0], TileUtils.to_string(m[1])) for m in p_state['melds']]

                # 标记当前行动者
                prefix = "👉 " if p_i == pid else "   "
                f.write(f"{prefix}A{p_i}: {hand_str} | {melds_str}\n")
            f.write("\n")

        f.write("=== End of Replay ===\n")


def train():
    # ... (主体训练逻辑保持不变，只需直接复制之前的 train 代码) ...
    print(f"🚀 Starting Training from SCRATCH...")
    print(f"📂 Weights dir: {SAVE_DIR}")
    print(f"📂 Replays dir: {REPLAY_DIR}")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"⚙️  Device: {device_name}")

    env = MahjongEnv()
    agent = PPOAgent(run_name="mahjong_v1", output_dir=SAVE_DIR)

    total_timesteps = 0
    i_episode = 0
    reward_history = deque(maxlen=200)
    win_counts = Counter()
    win_types = Counter()
    best_avg_reward = -float('inf')

    while total_timesteps < Cfg.MAX_TIMESTEPS:
        batch_start_time = time.time()
        while len(agent.buffer) < Cfg.BATCH_SIZE:
            obs = env.reset()
            done = False
            ep_reward = 0
            while not done:
                action, log_prob, val = agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                mask = obs['mask']
                agent.store_transition((obs, action, log_prob, reward, done, val, mask))
                obs = next_obs
                ep_reward += reward
                total_timesteps += 1

            i_episode += 1
            reward_history.append(ep_reward)

            if 'winner' in info:
                winner_id = info['winner']
                win_counts[winner_id] += 1
                w_type = "Unknown"
                try:
                    is_win, w_type = env.rules.is_winning(
                        env.players[winner_id]['hand'],
                        env.laizi_set,
                        extra_laizi_cnt=env.players[winner_id]['flower_laizis']
                    )
                except:
                    pass
                win_types[w_type] += 1

                if i_episode % 50 == 0 or ep_reward > 5.0:
                    save_game_replay(env, i_episode, winner_id, ep_reward, w_type)

        avg_reward = np.mean(reward_history) if reward_history else 0
        total_games = sum(win_counts.values()) + 1e-9
        win_dist = {k: v / total_games for k, v in win_counts.items()}

        print("\n" + "=" * 60)
        print(f"🔥 [Update] Step {total_timesteps} / {Cfg.MAX_TIMESTEPS}")
        print(f"   ⏱️  Time Cost: {time.time() - batch_start_time:.2f}s")
        print(f"   📊 Avg Reward: {avg_reward:.4f}")
        print(
            f"   🏆 Win Dist  : A0:{win_dist.get(0, 0):.2f} | A1:{win_dist.get(1, 0):.2f} | A2:{win_dist.get(2, 0):.2f} | A3:{win_dist.get(3, 0):.2f}")
        print(f"   🀄 Top Hands : {win_types.most_common(3)}")
        print("=" * 60)

        agent.update()
        win_counts.clear()
        win_types.clear()

        save_interval = 20
        update_count = total_timesteps // Cfg.BATCH_SIZE
        if update_count % save_interval == 0:
            save_name = f"mahjong_agent_step{total_timesteps}.pth"
            agent.save_model(os.path.join(SAVE_DIR, save_name))
            print(f"💾 Checkpoint saved: {save_name}")

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_path = os.path.join(SAVE_DIR, "best_model.pth")
            agent.save_model(best_path)
            print(f"🌟 New Record! Best model saved with Reward: {best_avg_reward:.4f}")

    print("✅ Training Completed!")
    agent.save_model(os.path.join(SAVE_DIR, "mahjong_agent_final.pth"))


if __name__ == "__main__":
    train()