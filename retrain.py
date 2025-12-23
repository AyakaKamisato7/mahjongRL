import os
import torch
import numpy as np
import glob
import re
import time
from collections import deque, Counter
from config import MahjongConfig as Cfg
from env import MahjongEnv, TileUtils
from agent import PPOAgent

# --- 1. 路径配置 ---
WORK_DIR = r"D:/pyworksp/mahjongRL/"
SAVE_DIR = os.path.join(WORK_DIR, "pth")
REPLAY_DIR = os.path.join(WORK_DIR, "replays")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(REPLAY_DIR, exist_ok=True)


def find_latest_checkpoint(path_dir):
    files = glob.glob(os.path.join(path_dir, "*.pth"))
    step_files = [f for f in files if "step" in f]
    if not step_files: return None, 0
    latest_file = None
    max_step = -1
    pattern = re.compile(r"mahjong_agent_step(\d+).pth")
    for f in step_files:
        match = pattern.search(f)
        if match:
            step_num = int(match.group(1))
            if step_num > max_step:
                max_step = step_num
                latest_file = f
    return latest_file, max_step


def save_game_replay(env, episode_id, winner_id, reward, final_hand_type):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"replay_retrain_ep{episode_id}_{timestamp}_win{winner_id}.txt"
    filepath = os.path.join(REPLAY_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"=== Mahjong Replay [Episode {episode_id}] (Retrain) ===\n")
        f.write(f"Time: {timestamp} | Winner: Agent {winner_id} | Reward: {reward:.2f} | Type: {final_hand_type}\n")
        f.write(
            f"Laizi: {TileUtils.to_string(env.indicator_tile)} -> {[TileUtils.to_string(l) for l in env.laizi_set]}\n")
        f.write("=" * 80 + "\n\n")

        for step_i, record in enumerate(env.action_history):
            pid = record['pid']
            act = record['action']
            snapshot = record['snapshot']

            if act == Cfg.ACT_DRAW:
                act_str = "DRAWS a tile"
            elif act <= 33:
                act_str = f"Discard {TileUtils.to_string(act)}"
            else:
                act_map = {34: "PASS", 35: "HU", 36: "PON", 37: "GANG", 38: "CHI_L", 39: "CHI_M", 40: "CHI_R"}
                act_str = act_map.get(act, "UNKNOWN")

            f.write(f"--- Step {step_i:03d} | Agent {pid} performs: {act_str} ---\n")

            for p_i in range(4):
                p_state = snapshot[p_i]
                hand_str = []
                for t_id, count in enumerate(p_state['hand']):
                    if count > 0: hand_str.extend([TileUtils.to_string(t_id)] * count)
                if p_state['flower_laizis'] > 0:
                    hand_str.append(f"[花赖x{p_state['flower_laizis']}]")
                melds_str = [(m[0], TileUtils.to_string(m[1])) for m in p_state['melds']]

                prefix = "👉 " if p_i == pid else "   "
                f.write(f"{prefix}A{p_i}: {hand_str} | Melds: {melds_str}\n")
            f.write("\n")

        f.write("=== End of Replay ===\n")


def retrain():
    print(f"🔄 Starting Re-Training (Resume Mode)...")
    print(f"📂 Weights dir: {SAVE_DIR}")
    print(f"📂 Replays dir: {REPLAY_DIR}")

    latest_ckpt, start_timestep = find_latest_checkpoint(SAVE_DIR)

    if latest_ckpt:
        print(f"🔍 Found latest checkpoint: {latest_ckpt}")
        print(f"⏱️  Resuming from timestep: {start_timestep}")
    else:
        print(f"⚠️  No checkpoint found! Starting from SCRATCH.")
        start_timestep = 0

    env = MahjongEnv()
    agent = PPOAgent(run_name="mahjong_retrain", output_dir=SAVE_DIR)
    if latest_ckpt: agent.load_model(latest_ckpt)

    total_timesteps = start_timestep
    i_episode = 0
    reward_history = deque(maxlen=200)
    win_counts = Counter()
    win_types = Counter()
    best_avg_reward = -float('inf')

    # ==========================================
    # [核心修改] 强制增量训练逻辑
    # ==========================================
    EXTRA_STEPS = 1000000  # 你想要额外训练的步数 (这里设为10万)
    target_timesteps = start_timestep + EXTRA_STEPS

    print(f"🎯 Target Timesteps set to: {target_timesteps}")
    print(f"   (Base: {start_timestep} + Extra: {EXTRA_STEPS})")
    # ==========================================

    while total_timesteps < target_timesteps:
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

            if info.get('winner') is not None:
                winner_id = info['winner']
                win_counts[winner_id] += 1
                w_type = "Unknown"
                try:
                    is_win, calculated_type = env.rules.is_winning(
                        env.players[winner_id]['hand'],
                        env.laizi_set,
                        extra_laizi_cnt=env.players[winner_id]['flower_laizis']
                    )
                    if is_win:
                        w_type = calculated_type
                    else:
                        w_type = "FalseHu"
                except Exception as e:
                    w_type = f"Err: {str(e)[:10]}"

                win_types[w_type] += 1

                if i_episode % 50 == 0 or ep_reward > 5.0:
                    save_game_replay(env, i_episode, winner_id, ep_reward, w_type)

        avg_reward = np.mean(reward_history) if reward_history else 0
        total_games = sum(win_counts.values()) + 1e-9
        win_dist = {k: v / total_games for k, v in win_counts.items()}

        print("\n" + "=" * 60)
        print(f"🔥 [Resume Update] Step {total_timesteps} / {target_timesteps}")
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
        update_count = (total_timesteps - start_timestep) // Cfg.BATCH_SIZE
        if len(agent.buffer) == 0:
            if update_count > 0 and update_count % save_interval == 0:
                save_name = f"mahjong_agent_step{total_timesteps}.pth"
                save_path = os.path.join(SAVE_DIR, save_name)
                agent.save_model(save_path)
                print(f"💾 Checkpoint saved: {save_path}")

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_path = os.path.join(SAVE_DIR, "best_model.pth")
                agent.save_model(best_path)
                print(f"🌟 New Session Record! Best model saved: {best_avg_reward:.4f}")

    print("✅ Re-Training Completed!")
    agent.save_model(os.path.join(SAVE_DIR, "mahjong_agent_retrain_final.pth"))


if __name__ == "__main__":
    retrain()