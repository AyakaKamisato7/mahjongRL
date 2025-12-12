import os

# 屏蔽干扰信息
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import warnings

warnings.filterwarnings("ignore")

import pygame
import sys
import time
import traceback  # 用于打印报错
from env import MahjongEnv, TileUtils
from agent import PPOAgent
from config import MahjongConfig as Cfg
from retrain import find_latest_checkpoint, SAVE_DIR

# --- 基础配置 ---
BG_COLOR = (34, 139, 34)
TILE_WIDTH = 42
TILE_HEIGHT = 62
FONT_SIZE = 24
RIVER_SCALE = 0.75


def get_chinese_font_path():
    """
    [稳定版] 只返回字体路径字符串，不返回对象
    """
    font_names = ["simhei.ttf", "msyh.ttc", "simsun.ttc", "PingFang.ttc", "Arial Unicode.ttf"]
    font_dirs = ["C:\\Windows\\Fonts", "/System/Library/Fonts", "/usr/share/fonts"]
    for folder in font_dirs:
        for name in font_names:
            path = os.path.join(folder, name)
            if os.path.exists(path): return path
    return None


class MahjongGUI:
    def __init__(self, agent_path=None):
        pygame.init()

        # [修改 1] 获取屏幕尺寸，但不强制全屏，使用无边框窗口或普通窗口
        info = pygame.display.Info()
        self.W = int(info.current_w * 0.9)  # 占用 90% 屏幕大小，防止撑爆
        self.H = int(info.current_h * 0.9)

        # 使用可缩放窗口，避免分辨率不匹配导致的闪退
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        pygame.display.set_caption("奉化麻将 AI 对战回放 (Stable)")

        # [修改 2] 预加载三种大小的字体，避免在循环中动态创建导致内存泄漏或报错
        self.font_path = get_chinese_font_path()
        if self.font_path:
            self.font_large = pygame.font.Font(self.font_path, FONT_SIZE)
            self.font_small = pygame.font.Font(self.font_path, int(FONT_SIZE * 0.7))
            self.font_tiny = pygame.font.Font(self.font_path, int(FONT_SIZE * 0.5))
        else:
            print("⚠️ 未找到中文字体，使用默认字体")
            self.font_large = pygame.font.SysFont("microsoftyahei", FONT_SIZE)
            self.font_small = pygame.font.SysFont("microsoftyahei", int(FONT_SIZE * 0.7))
            self.font_tiny = pygame.font.SysFont("microsoftyahei", int(FONT_SIZE * 0.5))

        self.clock = pygame.time.Clock()

        # 加载环境
        print("正在初始化环境...")
        self.env = MahjongEnv()
        self.agent = PPOAgent()

        if agent_path:
            try:
                self.agent.load_model(agent_path)
                print(f"✅ 模型加载成功: {agent_path}")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                print("将使用随机策略运行")
        else:
            print("⚠️ 未指定模型，使用随机策略")

        self.obs = self.env.reset()
        self.done = False
        self.auto_play = False
        self.step_delay = 0.5
        self.last_step_time = 0
        self.steps = 0
        self.info_text = "空格:单步 | P:自动 | R:重置 | ESC:退出"

    def _draw_tile(self, tile_id, x, y, scale=1.0, is_laizi=False, special_text=None):
        w = int(TILE_WIDTH * scale)
        h = int(TILE_HEIGHT * scale)
        rect = pygame.Rect(x, y, w, h)

        pygame.draw.rect(self.screen, (250, 248, 235), rect, border_radius=4)
        pygame.draw.rect(self.screen, (80, 80, 80), rect, 1, border_radius=4)

        text_str = ""
        color = (20, 20, 20)

        if special_text:
            text_str = special_text
            color = (255, 0, 255)
        elif tile_id == -1:
            text_str = "花赖"
            color = (200, 100, 0)
        else:
            raw_str = TileUtils.to_string(tile_id)
            text_str = raw_str
            if "万" in raw_str or raw_str == "中":
                color = (180, 0, 0)
            elif "筒" in raw_str or raw_str == "白":
                color = (0, 0, 160)
            elif "索" in raw_str or raw_str == "发":
                color = (0, 120, 0)
            elif tile_id >= 34:
                color = (160, 32, 240)

        # [修改 3] 根据情况选择预设字体，不再动态 new Font
        if scale >= 0.9:
            use_font = self.font_large
        elif scale >= 0.7:
            use_font = self.font_small
        else:
            use_font = self.font_tiny

        # 如果字数多（如"花赖"）且空间小，强制用最小字体
        if len(text_str) > 1 and scale < 1.0:
            use_font = self.font_tiny

        text_surf = use_font.render(text_str, True, color)
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)

        if is_laizi:
            pygame.draw.rect(self.screen, (255, 215, 0), rect, 3, border_radius=4)

    def _draw_melds(self, player, start_x, start_y):
        for i, (m_type, m_tile) in enumerate(player['melds']):
            offset_x = i * (TILE_WIDTH * 2.8)
            tiles = []
            label = ""

            if m_type == 'PON':
                tiles = [m_tile] * 3; label = "碰"
            elif m_type == 'GANG':
                tiles = [m_tile] * 4; label = "杠"
            elif m_type == 'CHI_L':
                tiles = [m_tile, m_tile + 1, m_tile + 2]; label = "吃"
            elif m_type == 'CHI_M':
                tiles = [m_tile - 1, m_tile, m_tile + 1]; label = "吃"
            elif m_type == 'CHI_R':
                tiles = [m_tile - 2, m_tile - 1, m_tile]; label = "吃"
            else:
                tiles = [m_tile] * 3; label = "吃"

            for k, tid in enumerate(tiles):
                self._draw_tile(tid, start_x + offset_x + k * (TILE_WIDTH * 0.7), start_y, scale=0.7)

            lbl = self.font_small.render(label, True, (255, 200, 0))
            self.screen.blit(lbl, (start_x + offset_x, start_y - 18))

    def _draw_player_hand(self, pid, cx, cy):
        player = self.env.players[pid]
        hand_tiles = []
        for t_id, count in enumerate(player['hand']):
            hand_tiles.extend([t_id] * count)
        hand_tiles.extend([-1] * player['flower_laizis'])

        hand_width = len(hand_tiles) * (TILE_WIDTH + 2)

        if pid == 0:  # 自己 (下)
            start_x = cx - hand_width // 2
            start_y = self.H - 120
            meld_x = start_x + hand_width + 20
            meld_y = start_y + 10
            flower_x = start_x - 120
            flower_y = start_y

        elif pid == 1:  # 右家
            start_x = self.W - hand_width - 50
            start_y = cy - 60
            meld_x = start_x
            meld_y = start_y + TILE_HEIGHT + 15
            flower_x = start_x
            flower_y = start_y - 60

        elif pid == 2:  # 对家 (上)
            start_x = cx - hand_width // 2
            start_y = 60
            meld_x = start_x - 20 - (len(player['melds']) * TILE_WIDTH * 2.8)
            meld_y = start_y + 10
            flower_x = start_x + hand_width + 50
            flower_y = start_y

        elif pid == 3:  # 左家
            start_x = 50
            start_y = cy - 60
            meld_x = start_x
            meld_y = start_y + TILE_HEIGHT + 15
            flower_x = start_x
            flower_y = start_y - 60

        for i, tid in enumerate(hand_tiles):
            dx = i * (TILE_WIDTH + 2)
            is_lz = (tid in self.env.laizi_set)

            txt = None
            if tid == -1:
                if self.env.laizi_set:
                    first_lz = list(self.env.laizi_set)[0]
                    txt = "花赖" if first_lz >= 34 else "赖"
                else:
                    txt = "花"

            self._draw_tile(tid, start_x + dx, start_y, is_laizi=is_lz, special_text=txt)

        # 补花
        for i, fid in enumerate(player['flowers']):
            row = i // 4;
            col = i % 4
            fx = flower_x + col * 35
            fy = flower_y + row * 45
            self._draw_tile(fid, fx, fy, scale=0.8)

        # 花赖
        offset = len(player['flowers'])
        for i in range(player['flower_laizis']):
            idx = offset + i
            row = idx // 4;
            col = idx % 4
            fx = flower_x + col * 35
            fy = flower_y + row * 45
            self._draw_tile(-1, fx, fy, scale=0.8, special_text="花赖", is_laizi=True)

        self._draw_melds(player, meld_x, meld_y)

        if self.env.current_player == pid:
            pygame.draw.circle(self.screen, (255, 50, 50), (start_x - 15, start_y + TILE_HEIGHT // 2), 8)

    def _draw_river(self, pid, cx, cy):
        history = self.env.action_history
        discards = []
        for i, rec in enumerate(history):
            if rec['action'] == Cfg.ACT_DRAW: continue
            if rec['pid'] == pid and rec['action'] <= 33:
                is_claimed = False
                if i + 1 < len(history):
                    next_act = history[i + 1]['action']
                    if next_act in [Cfg.ACT_PON, Cfg.ACT_GANG, Cfg.ACT_HU,
                                    Cfg.ACT_CHI_LEFT, Cfg.ACT_CHI_MID, Cfg.ACT_CHI_RIGHT]:
                        is_claimed = True
                if not is_claimed: discards.append(rec['action'])

        w = int(TILE_WIDTH * RIVER_SCALE)
        h = int(TILE_HEIGHT * RIVER_SCALE)
        cols = 6

        if pid == 0:
            sx, sy = cx - 100, cy + 90
        elif pid == 1:
            sx, sy = cx + 220, cy - 80
        elif pid == 2:
            sx, sy = cx - 100, cy - 200
        elif pid == 3:
            sx, sy = cx - 400, cy - 80

        for i, tid in enumerate(discards):
            r = i // cols
            c = i % cols
            self._draw_tile(tid, sx + c * (w + 2), sy + r * (h + 2), scale=RIVER_SCALE,
                            is_laizi=(tid in self.env.laizi_set))

    def _draw_hud(self):
        panel = pygame.Surface((300, 160))
        panel.set_alpha(180)
        panel.fill((0, 0, 0))
        self.screen.blit(panel, (10, 10))

        lz_str = " ".join([TileUtils.to_string(l) for l in self.env.laizi_set])
        last_str = "-"
        if self.env.last_discard is not None:
            last_str = f"A{self.env.last_discard_pid} 打 {TileUtils.to_string(self.env.last_discard)}"

        texts = [
            f"剩余牌数: {len(self.env.wall)}",
            f"本局赖子: {lz_str}",
            f"当前步数: {self.steps}",
            f"上一动作: {last_str}",
            f"状态: {self.env.phase}"
        ]

        for i, t in enumerate(texts):
            s = self.font_large.render(t, True, (255, 255, 255))
            self.screen.blit(s, (30, 30 + i * 28))

    def render(self):
        self.screen.fill(BG_COLOR)
        cx, cy = self.W // 2, self.H // 2
        for i in range(4): self._draw_river(i, cx, cy)
        for i in range(4): self._draw_player_hand(i, cx, cy)
        self._draw_hud()
        tip = self.font_large.render(self.info_text, True, (200, 200, 200))
        self.screen.blit(tip, (20, self.H - 40))
        pygame.display.flip()

    def step_game(self):
        if self.done:
            self.info_text = "游戏结束! 按 R 重开"
            return

        action, _, _ = self.agent.select_action(self.obs, eval_mode=True)

        act_str = ""
        if action <= 33:
            act_str = f"打出 {TileUtils.to_string(action)}"
        else:
            acts = {34: "过", 35: "胡!", 36: "碰", 37: "杠", 38: "左吃", 39: "中吃", 40: "右吃"}
            act_str = acts.get(action, "未知")

        self.info_text = f"Agent {self.env.current_player} -> {act_str}"

        self.obs, reward, self.done, info = self.env.step(action)
        self.steps += 1

        if self.done:
            winner = info.get('winner')
            if winner is not None:
                w_type = "Check Log"
                try:
                    _, w_type = self.env.rules.is_winning(
                        self.env.players[winner]['hand'],
                        self.env.laizi_set,
                        extra_laizi_cnt=self.env.players[winner]['flower_laizis']
                    )
                except:
                    pass
                self.info_text = f"WINNER: A{winner} [{w_type}] Rew:{reward:.1f}"
            else:
                self.info_text = "流局 (荒牌)"

    def run(self):
        running = True
        while running:
            now = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.step_game()
                    elif event.key == pygame.K_p:
                        self.auto_play = not self.auto_play
                    elif event.key == pygame.K_r:
                        self.obs = self.env.reset()
                        self.done = False
                        self.steps = 0
            if self.auto_play and not self.done:
                if now - self.last_step_time > self.step_delay:
                    self.step_game()
                    self.last_step_time = now
            self.render()
            self.clock.tick(30)
        pygame.quit()


if __name__ == "__main__":
    print("🚀 GUI 启动中...")

    # --- [防闪退核心] 捕获所有异常 ---
    try:
        ckpt_path, _ = find_latest_checkpoint(SAVE_DIR)
        if ckpt_path:
            print(f"📂 加载模型: {ckpt_path}")
            gui = MahjongGUI(agent_path=ckpt_path)
            gui.run()
        else:
            print("⚠️ 未找到模型，运行随机策略")
            gui = MahjongGUI()
            gui.run()

    except Exception as e:
        print("\n" + "=" * 40)
        print("❌ 发生错误，程序已暂停!")
        print("错误信息:", e)
        print("=" * 40)
        traceback.print_exc()  # 打印完整报错堆栈
        input("\n按回车键退出...")  # 暂停，让你有时间看报错