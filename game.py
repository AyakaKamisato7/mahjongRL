import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import warnings

warnings.filterwarnings("ignore")

import pygame
import sys
import time
from env import MahjongEnv, TileUtils
from agent import PPOAgent
from config import MahjongConfig as Cfg
from retrain import find_latest_checkpoint, SAVE_DIR

# --- 配色方案 ---
BG_COLOR = (34, 139, 34)
BTN_COLOR = (70, 130, 180)  # 按钮蓝
BTN_HOVER_COLOR = (100, 149, 237)
BTN_TEXT_COLOR = (255, 255, 255)
TILE_BACK_COLOR = (30, 100, 60)  # 牌背颜色 (深绿)

# 尺寸配置
TILE_WIDTH = 46
TILE_HEIGHT = 66
FONT_SIZE = 26
RIVER_SCALE = 0.75


def get_chinese_font_path():
    """寻找中文字体"""
    font_names = ["simhei.ttf", "msyh.ttc", "simsun.ttc", "PingFang.ttc", "Arial Unicode.ttf"]
    font_dirs = ["C:\\Windows\\Fonts", "/System/Library/Fonts", "/usr/share/fonts"]
    for folder in font_dirs:
        for name in font_names:
            path = os.path.join(folder, name)
            if os.path.exists(path): return path
    return None


class InteractiveMahjong:
    def __init__(self, agent_path=None):
        pygame.init()

        # [修改] 获取屏幕真实尺寸 (最大化窗口)
        info = pygame.display.Info()
        self.W = info.current_w
        self.H = info.current_h - 60  # 减去任务栏高度，防止底部被遮挡

        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        pygame.display.set_caption("奉化麻将: 人机大战 (You vs 3 Agents)")

        # 字体初始化
        self.font_path = get_chinese_font_path()
        if self.font_path:
            self.font = pygame.font.Font(self.font_path, FONT_SIZE)
            self.font_small = pygame.font.Font(self.font_path, int(FONT_SIZE * 0.7))
            self.font_btn = pygame.font.Font(self.font_path, 30)
        else:
            self.font = pygame.font.SysFont("microsoftyahei", FONT_SIZE)
            self.font_small = pygame.font.SysFont("microsoftyahei", int(FONT_SIZE * 0.7))
            self.font_btn = pygame.font.SysFont(None, 30)

        self.clock = pygame.time.Clock()

        # 游戏核心
        self.env = MahjongEnv()
        self.agent = PPOAgent()

        if agent_path:
            try:
                self.agent.load_model(agent_path)
                print(f"✅ AI 模型已加载: {agent_path}")
            except:
                print("❌ 模型加载失败，AI 将随机行动")
        else:
            print("⚠️ 未找到模型，AI 将随机行动")

        # 交互状态初始化
        self.human_pid = 0
        self.obs = self.env.reset()
        self.done = False
        self.info_text = "等待游戏开始..."

        # 状态变量补全
        self.steps = 0
        self.auto_play = False
        self.step_delay = 0.5
        self.last_step_time = 0
        self.active_buttons = []
        self.human_hand_rects = []

    def _draw_tile(self, tile_id, x, y, scale=1.0, is_laizi=False, special_text=None, is_hidden=False):
        w = int(TILE_WIDTH * scale)
        h = int(TILE_HEIGHT * scale)
        rect = pygame.Rect(x, y, w, h)

        if is_hidden:
            pygame.draw.rect(self.screen, (240, 240, 230), rect, border_radius=4)
            inner_rect = pygame.Rect(x + 2, y + 2, w - 4, h - 4)
            pygame.draw.rect(self.screen, TILE_BACK_COLOR, inner_rect, border_radius=2)
            pygame.draw.rect(self.screen, (40, 120, 70), inner_rect, 1)
            return

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

        use_font = self.font if scale >= 0.9 else self.font_small
        if len(text_str) > 1 and scale < 1.0:
            try:
                use_font = pygame.font.Font(self.font_path, int(FONT_SIZE * scale * 0.7))
            except:
                pass

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
        is_human = (pid == self.human_pid)

        for t_id, count in enumerate(player['hand']):
            hand_tiles.extend([t_id] * count)
        hand_tiles.extend([-1] * player['flower_laizis'])

        hand_width = len(hand_tiles) * (TILE_WIDTH + 2)

        if pid == 0:  # Human (Bottom)
            start_x = cx - hand_width // 2
            start_y = self.H - 140
            meld_x = start_x + hand_width + 20
            meld_y = start_y + 10
            flower_x = start_x - 120
            flower_y = start_y
            self.human_hand_rects = []

        elif pid == 1:  # AI Right
            start_x = self.W - hand_width - 50
            start_y = cy - 60

            # [关键修复] 右家副露固定在屏幕右侧向左偏移的位置，不随手牌移动
            # 这样保证副露永远在屏幕内，且大概在手牌下方
            meld_x = self.W - 680
            meld_y = start_y + TILE_HEIGHT + 15

            flower_x = start_x
            flower_y = start_y - 60

        elif pid == 2:  # AI Top
            start_x = cx - hand_width // 2
            start_y = 60
            meld_x = start_x - 20 - (len(player['melds']) * TILE_WIDTH * 2.8)
            meld_y = start_y + 10
            flower_x = start_x + hand_width + 50
            flower_y = start_y

        elif pid == 3:  # AI Left
            start_x = 50
            start_y = cy - 60
            meld_x = start_x
            meld_y = start_y + TILE_HEIGHT + 15
            flower_x = start_x
            flower_y = start_y - 60

        # 绘制立牌
        for i, tid in enumerate(hand_tiles):
            dx = i * (TILE_WIDTH + 2)
            should_hide = (not is_human)
            is_lz = (tid in self.env.laizi_set)

            txt = None
            if tid == -1:
                if self.env.laizi_set:
                    first_lz = list(self.env.laizi_set)[0]
                    txt = "花赖" if first_lz >= 34 else "赖"
                else:
                    txt = "花"

            self._draw_tile(tid, start_x + dx, start_y, is_laizi=is_lz, special_text=txt, is_hidden=should_hide)

            if is_human:
                rect = pygame.Rect(start_x + dx, start_y, TILE_WIDTH, TILE_HEIGHT)
                self.human_hand_rects.append((rect, tid))

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
        self.screen.blit(panel, (20, 20))

        lz_str = " ".join([TileUtils.to_string(l) for l in self.env.laizi_set])
        last_str = "-"
        if self.env.last_discard is not None:
            last_str = f"A{self.env.last_discard_pid} 打 {TileUtils.to_string(self.env.last_discard)}"

        texts = [
            f"剩余牌数: {len(self.env.wall)}",
            f"本局赖子: {lz_str}",
            f"我的状态: {self.env.phase}",
            f"上一动作: {last_str}",
            "ESC退出 | R重开"
        ]

        for i, t in enumerate(texts):
            color = (255, 255, 255)
            if "我的状态" in t and self.env.current_player == self.human_pid:
                color = (255, 215, 0)
            s = self.font.render(t, True, color)
            self.screen.blit(s, (30, 30 + i * 28))

    def _draw_interaction_panel(self):
        if self.env.current_player != self.human_pid: return
        mask = self.obs['mask']

        special_acts = {
            Cfg.ACT_PASS: "过", Cfg.ACT_HU: "胡", Cfg.ACT_GANG: "杠", Cfg.ACT_PON: "碰",
            Cfg.ACT_CHI_LEFT: "左吃", Cfg.ACT_CHI_MID: "中吃", Cfg.ACT_CHI_RIGHT: "右吃"
        }

        actions_available = []
        for act_id, label in special_acts.items():
            if mask[act_id] == 1.0: actions_available.append((act_id, label))

        if not actions_available: return

        self.active_buttons = []
        btn_w, btn_h = 100, 50
        gap = 20
        total_w = len(actions_available) * (btn_w + gap)
        start_x = (self.W - total_w) // 2
        start_y = self.H - 220

        for i, (act_id, label) in enumerate(actions_available):
            bx = start_x + i * (btn_w + gap)
            rect = pygame.Rect(bx, start_y, btn_w, btn_h)

            mouse_pos = pygame.mouse.get_pos()
            color = BTN_HOVER_COLOR if rect.collidepoint(mouse_pos) else BTN_COLOR

            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=8)

            txt_surf = self.font_btn.render(label, True, BTN_TEXT_COLOR)
            txt_rect = txt_surf.get_rect(center=rect.center)
            self.screen.blit(txt_surf, txt_rect)

            self.active_buttons.append((rect, act_id))

    def handle_human_click(self, pos):
        if self.env.current_player != self.human_pid: return False
        mask = self.obs['mask']

        for rect, act_id in self.active_buttons:
            if rect.collidepoint(pos):
                print(f"Clicked Button: {act_id}")
                self._execute_step(act_id)
                return True

        if self.env.phase == 'DISCARD':
            for rect, tile_id in self.human_hand_rects:
                if rect.collidepoint(pos):
                    if mask[tile_id] == 1.0:
                        print(f"Discard: {TileUtils.to_string(tile_id)}")
                        self._execute_step(tile_id)
                        return True
                    else:
                        print(f"Invalid: {TileUtils.to_string(tile_id)}")
        return False

    def _execute_step(self, action):
        self.obs, reward, self.done, info = self.env.step(action)
        self.steps += 1
        if self.done:
            self.active_buttons = []
            winner = info.get('winner')
            if winner is not None:
                if winner == self.human_pid:
                    self.info_text = f"你赢了! 奖励: {reward:.1f}"
                else:
                    self.info_text = f"你输了! 赢家: A{winner} 奖励: {reward:.1f}"
            else:
                self.info_text = "流局!"

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.obs = self.env.reset()
                        self.done = False
                        self.steps = 0
                        self.info_text = "游戏开始"
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.done and event.button == 1:
                        self.handle_human_click(event.pos)

            if not self.done:
                if self.env.current_player != self.human_pid:
                    pygame.time.wait(1000)  # AI 思考延迟
                    action, _, _ = self.agent.select_action(self.obs, eval_mode=True)
                    self._execute_step(action)

            self.screen.fill(BG_COLOR)
            cx, cy = self.W // 2, self.H // 2

            for i in range(4): self._draw_river(i, cx, cy)
            for i in range(4): self._draw_player_hand(i, cx, cy)

            self._draw_hud()
            self._draw_interaction_panel()

            if self.env.current_player == self.human_pid:
                status = "轮到你了! 请出牌或选择操作"
                color = (255, 255, 0)
            else:
                status = f"AI (A{self.env.current_player}) 思考中..."
                color = (200, 200, 200)

            if self.done: status = self.info_text

            tip = self.font.render(status, True, color)
            self.screen.blit(tip, (20, self.H - 40))

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    print("🚀 人机对战模式启动...")
    ckpt_path, _ = find_latest_checkpoint(SAVE_DIR)
    if ckpt_path and os.path.exists(ckpt_path):
        app = InteractiveMahjong(agent_path=ckpt_path)
        app.run()
    else:
        print("⚠️ 未找到模型，对手将随机行动")
        app = InteractiveMahjong()
        app.run()

# import os
#
# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# import warnings
#
# warnings.filterwarnings("ignore")
#
# import pygame
# import sys
# import time
# from env import MahjongEnv, TileUtils
# from agent import PPOAgent
# from config import MahjongConfig as Cfg
# from retrain import find_latest_checkpoint, SAVE_DIR
#
# # --- 配色方案 ---
# BG_COLOR = (34, 139, 34)
# BTN_COLOR = (70, 130, 180)  # 按钮蓝
# BTN_HOVER_COLOR = (100, 149, 237)
# BTN_TEXT_COLOR = (255, 255, 255)
# TILE_BACK_COLOR = (30, 100, 60)  # 牌背颜色 (深绿)
#
# # 尺寸配置
# TILE_WIDTH = 46
# TILE_HEIGHT = 66
# FONT_SIZE = 26
# RIVER_SCALE = 0.7
#
#
# def get_chinese_font_path():
#     """寻找中文字体"""
#     font_names = ["simhei.ttf", "msyh.ttc", "simsun.ttc", "PingFang.ttc", "Arial Unicode.ttf"]
#     font_dirs = ["C:\\Windows\\Fonts", "/System/Library/Fonts", "/usr/share/fonts"]
#     for folder in font_dirs:
#         for name in font_names:
#             path = os.path.join(folder, name)
#             if os.path.exists(path): return path
#     return None
#
#
# class InteractiveMahjong:
#     def __init__(self, agent_path=None):
#         pygame.init()
#
#         # 窗口设置
#         info = pygame.display.Info()
#         self.W = int(info.current_w * 0.9)
#         self.H = int(info.current_h * 0.9)
#         self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
#         pygame.display.set_caption("奉化麻将: 人机大战 (You vs 3 Agents)")
#
#         # 字体初始化
#         self.font_path = get_chinese_font_path()
#         if self.font_path:
#             self.font = pygame.font.Font(self.font_path, FONT_SIZE)
#             self.font_small = pygame.font.Font(self.font_path, int(FONT_SIZE * 0.7))
#             self.font_btn = pygame.font.Font(self.font_path, 30)  # 按钮大字体
#         else:
#             self.font = pygame.font.SysFont("microsoftyahei", FONT_SIZE)
#             self.font_small = pygame.font.SysFont("microsoftyahei", int(FONT_SIZE * 0.7))
#             self.font_btn = pygame.font.SysFont(None, 30)
#
#         self.clock = pygame.time.Clock()
#
#         # 游戏核心
#         self.env = MahjongEnv()
#         self.agent = PPOAgent()
#
#         if agent_path:
#             try:
#                 self.agent.load_model(agent_path)
#                 print(f"✅ AI 模型已加载: {agent_path}")
#             except:
#                 print("❌ 模型加载失败，AI 将随机行动")
#         else:
#             print("⚠️ 未找到模型，AI 将随机行动")
#
#         # 交互状态
#         self.human_pid = 0  # 人类固定坐在 0 号位 (下方)
#         self.obs = self.env.reset()
#         self.done = False
#         self.info_text = "等待游戏开始..."
#
#         # [修复] 补全缺失的状态变量初始化
#         self.steps = 0
#         self.auto_play = False  # 虽然人机模式一般不自动，但保留逻辑防止报错
#         self.step_delay = 0.5
#         self.last_step_time = 0
#
#         # 按钮区域缓存 (Rect, ActionID, Text)
#         self.active_buttons = []
#         # 手牌区域缓存 (Rect, TileID) 用于点击检测
#         self.human_hand_rects = []
#
#     def _draw_tile(self, tile_id, x, y, scale=1.0, is_laizi=False, special_text=None, is_hidden=False):
#         """绘制单张牌 (支持背面)"""
#         w = int(TILE_WIDTH * scale)
#         h = int(TILE_HEIGHT * scale)
#         rect = pygame.Rect(x, y, w, h)
#
#         if is_hidden:
#             # 绘制牌背
#             pygame.draw.rect(self.screen, (240, 240, 230), rect, border_radius=4)  # 侧面白边
#             inner_rect = pygame.Rect(x + 2, y + 2, w - 4, h - 4)
#             pygame.draw.rect(self.screen, TILE_BACK_COLOR, inner_rect, border_radius=2)
#             # 画个简单的花纹
#             pygame.draw.rect(self.screen, (40, 120, 70), inner_rect, 1)
#             return
#
#         # 绘制正面
#         pygame.draw.rect(self.screen, (250, 248, 235), rect, border_radius=4)
#         pygame.draw.rect(self.screen, (80, 80, 80), rect, 1, border_radius=4)
#
#         text_str = ""
#         color = (20, 20, 20)
#
#         if special_text:
#             text_str = special_text
#             color = (255, 0, 255)
#         elif tile_id == -1:
#             text_str = "花赖"
#             color = (200, 100, 0)
#         else:
#             raw_str = TileUtils.to_string(tile_id)
#             text_str = raw_str
#             if "万" in raw_str or raw_str == "中":
#                 color = (180, 0, 0)
#             elif "筒" in raw_str or raw_str == "白":
#                 color = (0, 0, 160)
#             elif "索" in raw_str or raw_str == "发":
#                 color = (0, 120, 0)
#             elif tile_id >= 34:
#                 color = (160, 32, 240)
#
#         use_font = self.font if scale >= 0.9 else self.font_small
#         if len(text_str) > 1 and scale < 1.0:
#             # 动态生成一个小字体
#             try:
#                 use_font = pygame.font.Font(self.font_path, int(FONT_SIZE * scale * 0.7))
#             except:
#                 pass
#
#         text_surf = use_font.render(text_str, True, color)
#         text_rect = text_surf.get_rect(center=rect.center)
#         self.screen.blit(text_surf, text_rect)
#
#         if is_laizi:
#             pygame.draw.rect(self.screen, (255, 215, 0), rect, 3, border_radius=4)
#
#     def _draw_melds(self, player, start_x, start_y):
#         """绘制副露"""
#         for i, (m_type, m_tile) in enumerate(player['melds']):
#             offset_x = i * (TILE_WIDTH * 2.8)
#             tiles = []
#             label = ""
#
#             if m_type == 'PON':
#                 tiles = [m_tile] * 3; label = "碰"
#             elif m_type == 'GANG':
#                 tiles = [m_tile] * 4; label = "杠"
#             elif m_type == 'CHI_L':
#                 tiles = [m_tile, m_tile + 1, m_tile + 2]; label = "吃"
#             elif m_type == 'CHI_M':
#                 tiles = [m_tile - 1, m_tile, m_tile + 1]; label = "吃"
#             elif m_type == 'CHI_R':
#                 tiles = [m_tile - 2, m_tile - 1, m_tile]; label = "吃"
#             else:
#                 tiles = [m_tile] * 3; label = "吃"
#
#             for k, tid in enumerate(tiles):
#                 self._draw_tile(tid, start_x + offset_x + k * (TILE_WIDTH * 0.7), start_y, scale=0.7)
#
#             lbl = self.font_small.render(label, True, (255, 200, 0))
#             self.screen.blit(lbl, (start_x + offset_x, start_y - 18))
#
#     def _draw_player_hand(self, pid, cx, cy):
#         """绘制玩家手牌 (AI的牌盖住)"""
#         player = self.env.players[pid]
#         hand_tiles = []
#
#         # 只有人类 (PID 0) 或者是明牌模式(调试用)才展开手牌
#         is_human = (pid == self.human_pid)
#
#         # 整理手牌数据
#         for t_id, count in enumerate(player['hand']):
#             hand_tiles.extend([t_id] * count)
#         hand_tiles.extend([-1] * player['flower_laizis'])
#
#         # 布局参数
#         hand_width = len(hand_tiles) * (TILE_WIDTH + 2)
#
#         if pid == 0:  # Human (Bottom)
#             start_x = cx - hand_width // 2
#             start_y = self.H - 140
#             meld_x = start_x + hand_width + 20
#             meld_y = start_y + 10
#             flower_x = start_x - 120
#             flower_y = start_y
#             self.human_hand_rects = []  # 重置点击区域
#
#         elif pid == 1:  # AI Right
#             start_x = self.W - hand_width - 50
#             start_y = cy - 60
#             meld_x = start_x
#             meld_y = start_y + TILE_HEIGHT + 15
#             flower_x = start_x
#             flower_y = start_y - 60
#
#         elif pid == 2:  # AI Top
#             start_x = cx - hand_width // 2
#             start_y = 60
#             meld_x = start_x - 20 - (len(player['melds']) * TILE_WIDTH * 2.8)
#             meld_y = start_y + 10
#             flower_x = start_x + hand_width + 50
#             flower_y = start_y
#
#         elif pid == 3:  # AI Left
#             start_x = 50
#             start_y = cy - 60
#             meld_x = start_x
#             meld_y = start_y + TILE_HEIGHT + 15
#             flower_x = start_x
#             flower_y = start_y - 60
#
#         # --- 绘制立牌 ---
#         for i, tid in enumerate(hand_tiles):
#             dx = i * (TILE_WIDTH + 2)
#
#             # 关键：AI的手牌全画背面，除了花牌和副露
#             should_hide = (not is_human)
#
#             is_lz = (tid in self.env.laizi_set)
#
#             txt = None
#             if tid == -1:
#                 if self.env.laizi_set:
#                     first_lz = list(self.env.laizi_set)[0]
#                     txt = "花赖" if first_lz >= 34 else "赖"
#                 else:
#                     txt = "花"
#
#             # 绘制
#             self._draw_tile(tid, start_x + dx, start_y, is_laizi=is_lz, special_text=txt, is_hidden=should_hide)
#
#             # 如果是人类，记录点击区域
#             if is_human:
#                 # 记录 (Rect, TileID)
#                 rect = pygame.Rect(start_x + dx, start_y, TILE_WIDTH, TILE_HEIGHT)
#                 self.human_hand_rects.append((rect, tid))
#
#         # --- 补花 (所有人都可见) ---
#         for i, fid in enumerate(player['flowers']):
#             row = i // 4;
#             col = i % 4
#             fx = flower_x + col * 35
#             fy = flower_y + row * 45
#             self._draw_tile(fid, fx, fy, scale=0.8)  # 花牌始终正面
#
#         # --- 花牌赖子 (所有人都可见数量，具体内容不可见) ---
#         offset = len(player['flowers'])
#         for i in range(player['flower_laizis']):
#             idx = offset + i
#             row = idx // 4;
#             col = idx % 4
#             fx = flower_x + col * 35
#             fy = flower_y + row * 45
#             self._draw_tile(-1, fx, fy, scale=0.8, special_text="花赖", is_laizi=True)
#
#         # --- 副露 (所有人都可见) ---
#         self._draw_melds(player, meld_x, meld_y)
#
#         # --- 标记出牌人 ---
#         if self.env.current_player == pid:
#             pygame.draw.circle(self.screen, (255, 50, 50), (start_x - 15, start_y + TILE_HEIGHT // 2), 8)
#
#     def _draw_river(self, pid, cx, cy):
#         """绘制牌河"""
#         history = self.env.action_history
#         discards = []
#         for i, rec in enumerate(history):
#             if rec['action'] == Cfg.ACT_DRAW: continue
#             if rec['pid'] == pid and rec['action'] <= 33:
#                 is_claimed = False
#                 if i + 1 < len(history):
#                     next_act = history[i + 1]['action']
#                     if next_act in [Cfg.ACT_PON, Cfg.ACT_GANG, Cfg.ACT_HU,
#                                     Cfg.ACT_CHI_LEFT, Cfg.ACT_CHI_MID, Cfg.ACT_CHI_RIGHT]:
#                         is_claimed = True
#                 if not is_claimed: discards.append(rec['action'])
#
#         w = int(TILE_WIDTH * RIVER_SCALE)
#         h = int(TILE_HEIGHT * RIVER_SCALE)
#         cols = 6
#
#         if pid == 0:
#             sx, sy = cx - 100, cy + 90
#         elif pid == 1:
#             sx, sy = cx + 220, cy - 80
#         elif pid == 2:
#             sx, sy = cx - 100, cy - 200
#         elif pid == 3:
#             sx, sy = cx - 400, cy - 80
#
#         for i, tid in enumerate(discards):
#             r = i // cols
#             c = i % cols
#             self._draw_tile(tid, sx + c * (w + 2), sy + r * (h + 2), scale=RIVER_SCALE,
#                             is_laizi=(tid in self.env.laizi_set))
#
#     def _draw_hud(self):
#         panel = pygame.Surface((300, 160))
#         panel.set_alpha(180)
#         panel.fill((0, 0, 0))
#         self.screen.blit(panel, (20, 20))
#
#         lz_str = " ".join([TileUtils.to_string(l) for l in self.env.laizi_set])
#         last_str = "-"
#         if self.env.last_discard is not None:
#             last_str = f"A{self.env.last_discard_pid} 打 {TileUtils.to_string(self.env.last_discard)}"
#
#         texts = [
#             f"剩余牌数: {len(self.env.wall)}",
#             f"本局赖子: {lz_str}",
#             f"我的状态: {self.env.phase}",  # 对人类来说状态很重要
#             f"上一动作: {last_str}",
#             "ESC退出 | R重开"
#         ]
#
#         for i, t in enumerate(texts):
#             color = (255, 255, 255)
#             if "我的状态" in t and self.env.current_player == self.human_pid:
#                 color = (255, 215, 0)  # 轮到自己时高亮
#             s = self.font.render(t, True, color)
#             self.screen.blit(s, (30, 30 + i * 28))
#
#     def _draw_interaction_panel(self):
#         """
#         绘制交互按钮 (仅当轮到人类且有可操作项时)
#         """
#         if self.env.current_player != self.human_pid:
#             return  # 不是我的回合
#
#         # 获取合法动作 Mask (从 obs 里拿)
#         mask = self.obs['mask']
#
#         # 识别当前可用的操作
#         actions_available = []
#
#         # 1. 特殊操作检查
#         special_acts = {
#             Cfg.ACT_PASS: "过",
#             Cfg.ACT_HU: "胡",
#             Cfg.ACT_GANG: "杠",
#             Cfg.ACT_PON: "碰",
#             Cfg.ACT_CHI_LEFT: "左吃",
#             Cfg.ACT_CHI_MID: "中吃",
#             Cfg.ACT_CHI_RIGHT: "右吃"
#         }
#
#         # 筛选可用动作
#         for act_id, label in special_acts.items():
#             if mask[act_id] == 1.0:
#                 actions_available.append((act_id, label))
#
#         if not actions_available:
#             return
#
#         self.active_buttons = []  # 重置
#
#         # 绘制按钮栏
#         btn_w, btn_h = 100, 50
#         gap = 20
#         total_w = len(actions_available) * (btn_w + gap)
#         start_x = (self.W - total_w) // 2
#         start_y = self.H - 220  # 在手牌上方
#
#         for i, (act_id, label) in enumerate(actions_available):
#             bx = start_x + i * (btn_w + gap)
#             by = start_y
#             rect = pygame.Rect(bx, by, btn_w, btn_h)
#
#             # 检测鼠标悬停
#             mouse_pos = pygame.mouse.get_pos()
#             color = BTN_HOVER_COLOR if rect.collidepoint(mouse_pos) else BTN_COLOR
#
#             # 画按钮
#             pygame.draw.rect(self.screen, color, rect, border_radius=8)
#             pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=8)
#
#             # 画字
#             txt_surf = self.font_btn.render(label, True, BTN_TEXT_COLOR)
#             txt_rect = txt_surf.get_rect(center=rect.center)
#             self.screen.blit(txt_surf, txt_rect)
#
#             # 存入缓存供点击检测
#             self.active_buttons.append((rect, act_id))
#
#     def handle_human_click(self, pos):
#         """处理人类点击事件"""
#         if self.env.current_player != self.human_pid:
#             return False  # 没轮到你，点的无效
#
#         mask = self.obs['mask']
#
#         # 1. 优先检测按钮点击 (吃碰杠胡过)
#         for rect, act_id in self.active_buttons:
#             if rect.collidepoint(pos):
#                 print(f"Human Clicked Button: {act_id}")
#                 self._execute_step(act_id)
#                 return True
#
#         # 2. 检测手牌点击 (打牌)
#         if self.env.phase == 'DISCARD':
#             for rect, tile_id in self.human_hand_rects:
#                 if rect.collidepoint(pos):
#                     # 检查是否合法
#                     if mask[tile_id] == 1.0:
#                         print(f"Human Discard: {TileUtils.to_string(tile_id)}")
#                         self._execute_step(tile_id)
#                         return True
#                     else:
#                         print(f"非法出牌: {TileUtils.to_string(tile_id)}")
#
#         return False
#
#     def _execute_step(self, action):
#         """执行一步环境交互"""
#         self.obs, reward, self.done, info = self.env.step(action)
#         self.steps += 1
#
#         if self.done:
#             self.active_buttons = []  # 清空按钮
#             winner = info.get('winner')
#             if winner is not None:
#                 if winner == self.human_pid:
#                     self.info_text = f"你赢了! 奖励: {reward:.1f}"
#                 else:
#                     self.info_text = f"你输了! 赢家: A{winner} 奖励: {reward:.1f}"
#             else:
#                 self.info_text = "流局!"
#
#     def run(self):
#         running = True
#         while running:
#             # 1. 事件循环
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False
#                 elif event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_ESCAPE:
#                         running = False
#                     elif event.key == pygame.K_r:  # 重置
#                         self.obs = self.env.reset()
#                         self.done = False
#                         self.steps = 0
#                         self.info_text = "游戏开始"
#
#                 elif event.type == pygame.MOUSEBUTTONDOWN:
#                     if not self.done and event.button == 1:  # 左键
#                         self.handle_human_click(event.pos)
#
#             # 2. 游戏逻辑
#             if not self.done:
#                 if self.env.current_player != self.human_pid:
#                     # --- AI 回合 ---
#
#                     # 👇【修改这里】取消注释，并修改数值
#                     # 300 表示延迟 300毫秒 (0.3秒)
#                     # 想慢一点改成 800 或 1000
#                     # 想快一点改成 100
#                     pygame.time.wait(1000)
#
#                     # AI 决策
#                     action, _, _ = self.agent.select_action(self.obs, eval_mode=True)
#                     self._execute_step(action)
#                 else:
#                     # --- 人类回合 ---
#                     pass
#
#             # 3. 渲染
#             self.screen.fill(BG_COLOR)
#             cx, cy = self.W // 2, self.H // 2
#
#             for i in range(4): self._draw_river(i, cx, cy)
#             for i in range(4): self._draw_player_hand(i, cx, cy)
#
#             self._draw_hud()
#             self._draw_interaction_panel()  # 绘制按钮
#
#             # 底部提示
#             if self.env.current_player == self.human_pid:
#                 status = "轮到你了! 请出牌或选择操作"
#                 color = (255, 255, 0)
#             else:
#                 status = f"AI (A{self.env.current_player}) 思考中..."
#                 color = (200, 200, 200)
#
#             if self.done: status = self.info_text
#
#             tip = self.font.render(status, True, color)
#             self.screen.blit(tip, (20, self.H - 40))
#
#             pygame.display.flip()
#             self.clock.tick(30)
#
#         pygame.quit()
#
#
# if __name__ == "__main__":
#     print("🚀 人机对战模式启动...")
#     ckpt_path, _ = find_latest_checkpoint(SAVE_DIR)
#
#     if ckpt_path and os.path.exists(ckpt_path):
#         app = InteractiveMahjong(agent_path=ckpt_path)
#         app.run()
#     else:
#         print("⚠️ 未找到模型，对手将随机行动")
#         app = InteractiveMahjong()
#         app.run()