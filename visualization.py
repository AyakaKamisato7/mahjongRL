import time
import os
import glob
import re
import argparse
import torch
import numpy as np
from env import MahjongEnv, TileUtils
from config import MahjongConfig as Cfg
from agent import PPOAgent

# --- 1. 路径配置 ---
WORK_DIR = r"D:/pyworksp/mahjongRL/"
SAVE_DIR = os.path.join(WORK_DIR, "pth")


def find_latest_checkpoint(path_dir):
    """
    [工具] 自动寻找目录下最新的模型文件
    优先找 step 最大的，其次找 best_model
    """
    if not os.path.exists(path_dir):
        return None

    files = glob.glob(os.path.join(path_dir, "*.pth"))
    if not files:
        return None

    # 优先找带 step 的
    step_files = [f for f in files if "step" in f]

    if not step_files:
        # 如果没有 step 文件，尝试找 best 或 final
        if os.path.join(path_dir, "best_model.pth") in files:
            return os.path.join(path_dir, "best_model.pth")
        if os.path.join(path_dir, "mahjong_agent_final.pth") in files:
            return os.path.join(path_dir, "mahjong_agent_final.pth")
        return files[-1]  # 实在没有就随便返回一个

    # 找 step 最大的
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
    return latest_file


def render_game_state(env):
    """
    [可视化] 打印当前牌桌状态 (支持所有 Agent 手牌显示)
    """
    print("\n" + "=" * 60)
    print(f"Phase: {env.phase} | Current Turn: Agent {env.current_player}")

    # 显示赖子信息
    laizi_str = [TileUtils.to_string(l) for l in env.laizi_set]
    print(f"🀄 Indicator: {TileUtils.to_string(env.indicator_tile)} | Laizi: {laizi_str}")

    # 显示上一张打出的牌
    last_discard_str = TileUtils.to_string(env.last_discard)
    if env.last_discard is None: last_discard_str = "None"
    print(f"🗑️  Last Discard: {last_discard_str} (by Agent {env.last_discard_pid})")
    print("-" * 60)

    for pid in range(4):
        p = env.players[pid]

        # 排序手牌方便观看
        hand_list = []
        for t_id, count in enumerate(p['hand']):
            if count > 0:
                hand_list.extend([TileUtils.to_string(t_id)] * count)

        # 花牌赖子
        if p['flower_laizis'] > 0:
            hand_list.append(f"[花赖x{p['flower_laizis']}]")

        melds_str = str([(m[0], TileUtils.to_string(m[1])) for m in p['melds']])
        flowers_str = str([TileUtils.to_string(f) for f in p['flowers']])

        # 高亮当前玩家
        prefix = "👉 " if pid == env.current_player else "   "
        role = "[DEALER]" if pid == env.dealer else ""

        print(f"{prefix}Agent {pid} {role}")
        print(f"      Hand   : {hand_list}")
        print(f"      Melds  : {melds_str}")
        print(f"      Flowers: {flowers_str}")

    print("=" * 60)


def watch_agent_play(agent, env_config=None, delay=1.0):
    """
    观看 Agent 自我对弈 (Live Inference)
    """
    env = MahjongEnv(config=env_config)
    obs = env.reset()
    done = False

    print("\n🎥 Starting Live Replay...")
    render_game_state(env)

    steps = 0
    while not done:
        # 使用 Agent 预测动作 (Eval 模式 - 贪婪策略)
        # 注意：这里我们让 Agent 控制 env.current_player
        # 因为是 Self-Play，Agent 扮演所有角色
        action, _, _ = agent.select_action(obs, eval_mode=True)

        # 动作名称解析
        if action <= 33:
            act_str = f"Discard {TileUtils.to_string(action)}"
        else:
            special_acts = {
                34: "PASS", 35: "HU (Win)", 36: "PON", 37: "GANG",
                38: "CHI Left", 39: "CHI Mid", 40: "CHI Right"
            }
            act_str = special_acts.get(action, "UNKNOWN")

        print(f"\n⚡ [Step {steps}] Agent {env.current_player} performs: {act_str}")

        obs, reward, done, info = env.step(action)
        render_game_state(env)

        if done:
            winner = info.get('winner', 'None')
            print(f"\n🏆 Game Over! Winner: Agent {winner} | Final Reward: {reward:.2f}")

            if winner != 'None':
                try:
                    # [关键] 这里必须传入 flower_laizis，否则判断不准
                    is_win, w_type = env.rules.is_winning(
                        env.players[winner]['hand'],
                        env.laizi_set,
                        extra_laizi_cnt=env.players[winner]['flower_laizis']
                    )
                    print(f"🎉 Winning Hand Type: {w_type}")
                except Exception as e:
                    print(f"⚠️ Error checking hand type: {e}")

        time.sleep(delay)
        steps += 1


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="指定要加载的模型路径")
    parser.add_argument("--delay", type=float, default=1.0, help="每步延迟时间(秒)")
    args = parser.parse_args()

    model_path = args.ckpt

    # 1. 如果没指定，自动找最新的
    if model_path is None:
        print("🔍 Searching for latest checkpoint in:", SAVE_DIR)
        model_path = find_latest_checkpoint(SAVE_DIR)

    # 2. 加载模型并运行
    if model_path and os.path.exists(model_path):
        print(f"👀 Loading model: {model_path}")

        # 这里的 output_dir 不重要，因为只是推理
        agent = PPOAgent()
        agent.load_model(model_path)

        # 开始看戏
        watch_agent_play(agent, delay=args.delay)
    else:
        print(f"❌ Could not find model at: {model_path}")
        print(f"Please check your SAVE_DIR: {SAVE_DIR} or train some episodes first.")