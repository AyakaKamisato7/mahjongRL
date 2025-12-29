import os

# 屏蔽 Pygame 欢迎信息
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import warnings

warnings.filterwarnings("ignore")

import pygame
import sys
import time
import random
import numpy as np  # 引入 numpy 进行手牌数组对比 
from env import MahjongEnv, TileUtils
from agent import PPOAgent
from config import MahjongConfig as Cfg

# --- 路径配置 ---
WORK_DIR = r"D:/pyworksp/mahjongRL/"
MODEL_PATH = os.path.join(WORK_DIR, "pth", "mahjong_agent_retrain_final.pth")
# MODEL_PATH = os.path.join(WORK_DIR, "pth", "best_model.pth")
IMG_DIR = os.path.join(WORK_DIR, "img")
SOUND_DIR = os.path.join(WORK_DIR, "sounds")  # [新增] 音频路径

# --- 配色方案 ---
BG_COLOR = (34, 139, 34)
BTN_COLOR = (70, 130, 180)
BTN_HOVER_COLOR = (100, 149, 237)
BTN_TEXT_COLOR = (255, 255, 255)
TILE_BACK_COLOR = (30, 100, 60)

# --- 尺寸配置 ---
TILE_WIDTH = 46
TILE_HEIGHT = 66
FONT_SIZE = 26
RIVER_SCALE = 0.90  # 牌河放大


def get_chinese_font_path():
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
        pygame.mixer.init()  # [新增] 初始化混音器

        # 窗口设置
        info = pygame.display.Info()
        self.W = int(info.current_w)
        self.H = int(info.current_h - 60)
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        pygame.display.set_caption("奉化麻将: 人机大战 (Sound Enabled)")

        # 字体
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

        # 加载资源
        self.tile_imgs = {}
        self._load_tile_images()

        self.sounds = {}  # 基础音效 (0-41.WAV)
        self.special_sounds = {}  # 特殊音效 (chi/peng/gang/hu)
        self._load_sounds()  # [新增] 加载音频

        # 初始化 Agent
        self.agent = PPOAgent()
        target_model = agent_path if agent_path else MODEL_PATH
        if os.path.exists(target_model):
            try:
                self.agent.load_model(target_model)
                print(f"✅ 成功加载模型: {target_model}")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
        else:
            print(f"⚠️ 未找到模型 {target_model}，使用随机策略")

        self.human_pid = 0
        self.last_drawn_tile = None  # [新增] 专门记录人类玩家刚摸到的牌 ID

        # 启动游戏
        self.env = None
        self.reset_game()

    def _load_tile_images(self):
        print(f"正在加载图片资源: {IMG_DIR}")
        if not os.path.exists(IMG_DIR):
            return
        for i in range(42):
            fname = f"{i}.png"
            fpath = os.path.join(IMG_DIR, fname)
            if os.path.exists(fpath):
                try:
                    img = pygame.image.load(fpath).convert_alpha()
                    self.tile_imgs[i] = img
                except:
                    pass

    def _load_sounds(self):
        """[新增] 加载所有音频资源"""
        print(f"正在加载音频资源: {SOUND_DIR}")
        if not os.path.exists(SOUND_DIR):
            print("⚠️ 音频目录不存在，将静音运行")
            return

        # 1. 加载编号音频 (0.WAV - 41.WAV)
        # 用于打牌语音 (0-33) 和备用动作语音
        for i in range(42):
            # 尝试大写 .WAV 和小写 .wav
            for ext in [".WAV", ".wav"]:
                fpath = os.path.join(SOUND_DIR, f"{i}{ext}")
                if os.path.exists(fpath):
                    try:
                        self.sounds[i] = pygame.mixer.Sound(fpath)
                        # print(f"Loaded sound: {i}")
                        break
                    except Exception as e:
                        print(f"Error loading {fpath}: {e}")

        # 2. 加载特殊动作音频 (优先使用 chi/peng/gang/hu.wav)
        # 映射配置: 动作ID -> 文件名列表(支持大小写)
        special_map = {
            'chi': ['chi.wav', 'chi.WAV'],  # 对应所有吃操作
            'peng': ['peng.wav', 'peng.WAV'],  # 对应碰
            'gang': ['gang.wav', 'gang.WAV'],  # 对应杠
            'hu': ['hu.wav', 'hu.WAV']  # 对应胡
        }

        # 具体的动作ID映射
        action_mapping = {
            Cfg.ACT_CHI_LEFT: 'chi',
            Cfg.ACT_CHI_MID: 'chi',
            Cfg.ACT_CHI_RIGHT: 'chi',
            Cfg.ACT_PON: 'peng',
            Cfg.ACT_GANG: 'gang',
            Cfg.ACT_HU: 'hu'
        }

        for act_id, key in action_mapping.items():
            for fname in special_map[key]:
                fpath = os.path.join(SOUND_DIR, fname)
                if os.path.exists(fpath):
                    try:
                        self.special_sounds[act_id] = pygame.mixer.Sound(fpath)
                        break  # 找到一个就停止
                    except:
                        pass

    def _play_action_sound(self, action):
        """[新增] 根据动作播放对应音效"""
        if action is None: return

        # 1. 优先播放特殊音效 (吃碰杠胡)
        if action in self.special_sounds:
            self.special_sounds[action].play()
            return

        # 2. 播放打牌音效 (0-33) 或其他编号音效
        # 过滤掉 PASS (34)
        if action == Cfg.ACT_PASS:
            return

        if action in self.sounds:
            self.sounds[action].play()

    def reset_game(self):
        """完全重置游戏逻辑"""
        print("🔄 正在重置游戏...")
        pygame.event.clear()  # 清除积压按键

        # 1. 重建环境
        self.env = MahjongEnv()
        self.obs = self.env.reset()
        self.last_drawn_tile = None

        # 2. 随机庄家 Hack
        last_draw_pid = -1
        if self.env.action_history:
            last_rec = self.env.action_history[-1]
            if last_rec['action'] == Cfg.ACT_DRAW:
                last_draw_pid = last_rec['pid']

        new_dealer = random.randint(0, 3)
        if last_draw_pid != -1 and new_dealer != 0:
            h0 = self.env.players[0]['hand']
            valid_tiles = [t for t in range(34) if h0[t] > 0]
            if valid_tiles:
                move_tile = random.choice(valid_tiles)
                self.env.players[0]['hand'][move_tile] -= 1
                self.env.players[new_dealer]['hand'][move_tile] += 1
                self.env.dealer = new_dealer
                self.env.current_player = new_dealer
                self.env.incoming_tile = move_tile
                self.env.action_history.append({
                    'pid': new_dealer,
                    'action': Cfg.ACT_DRAW,
                    'snapshot': None
                })
                if new_dealer == self.human_pid:
                    self.last_drawn_tile = move_tile

        if self.env.dealer == self.human_pid:
            h_human = self.env.players[self.human_pid]['hand']
            valid_tiles = [t for t in range(34) if h_human[t] > 0]
            if valid_tiles:
                self.last_drawn_tile = valid_tiles[-1]

        self.obs = self.env.get_observation(self.env.current_player)
        self.done = False
        self.steps = 0
        self.info_text = f"游戏开始! 庄家: A{self.env.dealer}"
        self.active_buttons = []
        self.human_hand_rects = []

        print(f"✅ 重置完成. 庄:A{self.env.dealer}, 初始高亮: {TileUtils.to_string(self.last_drawn_tile)}")

    def _draw_tile_img(self, tile_id, x, y, w, h):
        if tile_id in self.tile_imgs:
            img = pygame.transform.smoothscale(self.tile_imgs[tile_id], (w, h))
            self.screen.blit(img, (x, y))
            return True
        return False

    def _draw_tile(self, tile_id, x, y, scale=1.0, is_laizi=False, special_text=None, is_hidden=False, highlight=False):
        w = int(TILE_WIDTH * scale)
        h = int(TILE_HEIGHT * scale)

        offset_y = -20 if highlight else 0
        draw_rect = pygame.Rect(x, y + offset_y, w, h)

        if is_hidden:
            pygame.draw.rect(self.screen, (220, 220, 220), draw_rect, border_radius=4)
            inner = pygame.Rect(x + 2, y + 2 + offset_y, w - 4, h - 4)
            pygame.draw.rect(self.screen, TILE_BACK_COLOR, inner, border_radius=3)
            pygame.draw.rect(self.screen, (50, 150, 80), inner, 1)
            return

        pygame.draw.rect(self.screen, (250, 248, 235), draw_rect, border_radius=4)

        drawn = False
        if special_text is None and tile_id != -1:
            drawn = self._draw_tile_img(tile_id, x, y + offset_y, w, h)

        if not drawn:
            pygame.draw.rect(self.screen, (100, 100, 100), draw_rect, 1, border_radius=4)
            text = special_text if special_text else TileUtils.to_string(tile_id)
            color = (0, 0, 0)
            if "万" in text or text == "中":
                color = (180, 0, 0)
            elif "索" in text or text == "发":
                color = (0, 120, 0)
            elif "筒" in text or text == "白":
                color = (0, 0, 160)
            elif tile_id >= 34:
                color = (160, 32, 240)

            f = self.font if scale >= 0.9 else self.font_small
            if len(text) > 1 and scale < 1.0: f = pygame.font.Font(self.font_path, int(FONT_SIZE * scale * 0.6))
            s = f.render(text, True, color)
            s_r = s.get_rect(center=draw_rect.center)
            self.screen.blit(s, s_r)

        if is_laizi:
            pygame.draw.rect(self.screen, (255, 215, 0), draw_rect, 3, border_radius=4)

        if highlight:
            pygame.draw.rect(self.screen, (255, 30, 30), draw_rect, 3, border_radius=4)

    def _draw_player_hand(self, pid, cx, cy):
        player = self.env.players[pid]
        hand_counts = player['hand'].copy()

        is_human = (pid == self.human_pid)
        should_hide = (not is_human) and (not self.done)

        separate_tile = None

        if self.env.current_player == pid and self.env.phase == 'DISCARD':
            is_fresh_draw = False
            if self.env.action_history:
                last_rec = self.env.action_history[-1]
                if last_rec['pid'] == pid and last_rec['action'] == Cfg.ACT_DRAW:
                    is_fresh_draw = True

            if is_fresh_draw:
                target_tile = -1
                if is_human and self.last_drawn_tile is not None:
                    target_tile = self.last_drawn_tile
                elif not is_human:
                    valid_idx = np.where(hand_counts > 0)[0]
                    if len(valid_idx) > 0: target_tile = valid_idx[-1]

                if target_tile != -1 and 0 <= target_tile < 34:
                    if hand_counts[target_tile] > 0:
                        separate_tile = target_tile
                        hand_counts[target_tile] -= 1

        hand_tiles = []
        for t_id, count in enumerate(hand_counts):
            hand_tiles.extend([t_id] * count)

        if 'flower_laizi_ids' in player:
            hand_tiles.extend(player['flower_laizi_ids'])
        else:
            hand_tiles.extend([-1] * player['flower_laizis'])

        base_width = len(hand_tiles) * (TILE_WIDTH + 2)
        total_width = base_width + (TILE_WIDTH + 25) if separate_tile is not None else base_width

        if pid == 0:
            start_x = cx - total_width // 2
            start_y = self.H - 140
            meld_x = start_x + total_width + 20
            meld_y = start_y + 10
            flower_x = start_x - 120
            flower_y = start_y
            self.human_hand_rects = []
        elif pid == 1:
            start_x = self.W - total_width - 50
            start_y = cy - 60
            meld_x = self.W - 680
            meld_y = start_y + TILE_HEIGHT + 15
            flower_x = start_x
            flower_y = start_y - 60
        elif pid == 2:
            start_x = cx - total_width // 2
            start_y = 60
            meld_x = start_x - 20 - (len(player['melds']) * TILE_WIDTH * 2.8)
            meld_y = start_y + 10
            flower_x = start_x + total_width + 50
            flower_y = start_y
        elif pid == 3:
            start_x = 50
            start_y = cy - 60
            meld_x = start_x
            meld_y = start_y + TILE_HEIGHT + 15
            flower_x = start_x
            flower_y = start_y - 60

        if self.env.dealer == pid:
            z_s = self.font_small.render("庄", True, (255, 0, 0))
            z_x = start_x - 30 if pid in [0, 2, 3] else start_x + total_width + 10
            pygame.draw.circle(self.screen, (255, 255, 255), (z_x + 10, start_y + 10), 12)
            self.screen.blit(z_s, (z_x + 2, start_y))

        for i, tid in enumerate(hand_tiles):
            dx = i * (TILE_WIDTH + 2)
            is_lz = (tid in self.env.laizi_set)
            txt = "花赖" if (tid == -1 and self.env.laizi_set) else ("花" if tid == -1 else None)

            self._draw_tile(tid, start_x + dx, start_y, is_laizi=is_lz, special_text=txt, is_hidden=should_hide)

            if is_human:
                rect = pygame.Rect(start_x + dx, start_y, TILE_WIDTH, TILE_HEIGHT)
                self.human_hand_rects.append((rect, tid))

        if separate_tile is not None:
            sep_x = start_x + base_width + 25
            is_lz = (separate_tile in self.env.laizi_set)
            self._draw_tile(separate_tile, sep_x, start_y, is_laizi=is_lz, is_hidden=should_hide, highlight=True)

            if is_human:
                rect = pygame.Rect(sep_x, start_y - 20, TILE_WIDTH, TILE_HEIGHT)
                self.human_hand_rects.append((rect, separate_tile))

        for i, fid in enumerate(player['flowers']):
            r = i // 4;
            c = i % 4
            self._draw_tile(fid, flower_x + c * 35, flower_y + r * 45, scale=0.8)

        self._draw_melds(player, meld_x, meld_y)

        if self.env.current_player == pid:
            ind_x = start_x - 20
            ind_y = start_y + TILE_HEIGHT // 2
            pygame.draw.circle(self.screen, (255, 0, 0), (ind_x, ind_y), 8)
            pygame.draw.circle(self.screen, (255, 255, 255), (ind_x, ind_y), 10, 2)

    def _draw_melds(self, player, start_x, start_y):
        for i, (m_type, m_tile) in enumerate(player['melds']):
            offset_x = i * (TILE_WIDTH * 2.8)
            tiles = []
            label = ""
            if m_type == 'PON':
                tiles = [m_tile] * 3;
                label = "碰"
            elif m_type == 'GANG':
                tiles = [m_tile] * 4;
                label = "杠"
            elif m_type == 'CHI_L':
                tiles = [m_tile, m_tile + 1, m_tile + 2];
                label = "吃"
            elif m_type == 'CHI_M':
                tiles = [m_tile - 1, m_tile, m_tile + 1];
                label = "吃"
            elif m_type == 'CHI_R':
                tiles = [m_tile - 2, m_tile - 1, m_tile];
                label = "吃"
            else:
                tiles = [m_tile] * 3;
                label = "吃"

            for k, tid in enumerate(tiles):
                self._draw_tile(tid, start_x + offset_x + k * (TILE_WIDTH * 0.7), start_y, scale=0.7)
            lbl = self.font_small.render(label, True, (255, 200, 0))
            self.screen.blit(lbl, (start_x + offset_x, start_y - 18))

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
        panel = pygame.Surface((320, 180))
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
            f"庄家: A{self.env.dealer} | 骰子: {self.env.dice}",
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
        if self.done: return

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
        if self.done: return False
        if self.env.current_player != self.human_pid: return False

        mask = self.obs['mask']
        for rect, act_id in self.active_buttons:
            if rect.collidepoint(pos):
                self._execute_step(act_id)
                return True

        if self.env.phase == 'DISCARD':
            for rect, tile_id in self.human_hand_rects:
                if rect.collidepoint(pos):
                    if mask[tile_id] == 1.0:
                        self._execute_step(tile_id)
                        return True
                    else:
                        print(f"不可出牌: {TileUtils.to_string(tile_id)}")
        return False

    def _execute_step(self, action):
        # [新增] 播放音效 (在逻辑更新前播放，提升响应感)
        self._play_action_sound(action)

        # [核心修复] 在执行 Action 之前，备份当前人类手牌
        prev_hand_count = self.env.players[self.human_pid]['hand'].copy()

        # 执行动作
        self.obs, reward, self.done, info = self.env.step(action)
        self.steps += 1

        # [核心修复] 重新计算 last_drawn_tile
        if not self.done and self.env.current_player == self.human_pid and self.env.phase == 'DISCARD':
            curr_hand_count = self.env.players[self.human_pid]['hand']
            diff = curr_hand_count - prev_hand_count
            added_indices = np.where(diff > 0)[0]
            if len(added_indices) > 0:
                self.last_drawn_tile = added_indices[0]
            else:
                self.last_drawn_tile = None
        elif self.env.current_player != self.human_pid:
            self.last_drawn_tile = None

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
                        self.reset_game()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.done and event.button == 1:
                        self.handle_human_click(event.pos)

            if not self.done:
                if self.env.current_player != self.human_pid:
                    pygame.time.wait(1000)
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
    try:
        app = InteractiveMahjong()
        app.run()
    except Exception as e:
        import traceback

        traceback.print_exc()
        input("Error! Press Enter to exit...")



# import os
#
# # 屏蔽 Pygame 欢迎信息
# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# import warnings
#
# warnings.filterwarnings("ignore")
#
# import pygame
# import sys
# import time
# import random
# import numpy as np
# from env import MahjongEnv, TileUtils
# from agent import PPOAgent
# from config import MahjongConfig as Cfg
#
# # --- 路径配置 ---
# WORK_DIR = r"D:/pyworksp/mahjongRL/"
# # MODEL_PATH = os.path.join(WORK_DIR, "pth", "mahjong_agent_retrain_final.pth")
# MODEL_PATH = os.path.join(WORK_DIR, "pth", "best_model.pth")
# IMG_DIR = os.path.join(WORK_DIR, "img")
#
# # --- 配色方案 ---
# BG_COLOR = (34, 139, 34)
# BTN_COLOR = (70, 130, 180)
# BTN_HOVER_COLOR = (100, 149, 237)
# BTN_TEXT_COLOR = (255, 255, 255)
# TILE_BACK_COLOR = (30, 100, 60)
#
# # --- 尺寸配置 ---
# TILE_WIDTH = 46
# TILE_HEIGHT = 66
# FONT_SIZE = 26
# RIVER_SCALE = 0.90
#
#
# def get_chinese_font_path():
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
#         info = pygame.display.Info()
#         self.W = int(info.current_w)
#         self.H = int(info.current_h - 60)
#         self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
#         pygame.display.set_caption("奉化麻将: 人机大战 (Fixed Version)")
#
#         self.font_path = get_chinese_font_path()
#         if self.font_path:
#             self.font = pygame.font.Font(self.font_path, FONT_SIZE)
#             self.font_small = pygame.font.Font(self.font_path, int(FONT_SIZE * 0.7))
#             self.font_btn = pygame.font.Font(self.font_path, 30)
#         else:
#             self.font = pygame.font.SysFont("microsoftyahei", FONT_SIZE)
#             self.font_small = pygame.font.SysFont("microsoftyahei", int(FONT_SIZE * 0.7))
#             self.font_btn = pygame.font.SysFont(None, 30)
#
#         self.clock = pygame.time.Clock()
#         self.tile_imgs = {}
#         self._load_tile_images()
#
#         self.agent = PPOAgent()
#         target_model = agent_path if agent_path else MODEL_PATH
#         if os.path.exists(target_model):
#             try:
#                 self.agent.load_model(target_model)
#                 print(f"✅ 成功加载模型: {target_model}")
#             except Exception as e:
#                 print(f"❌ 模型加载失败: {e}")
#         else:
#             print(f"⚠️ 未找到模型 {target_model}，使用随机策略")
#
#         self.human_pid = 0
#         self.last_drawn_tile = None
#         self.env = None
#         self.reset_game()
#
#     def _load_tile_images(self):
#         print(f"正在加载图片资源: {IMG_DIR}")
#         if not os.path.exists(IMG_DIR):
#             return
#         for i in range(42):
#             fname = f"{i}.png"
#             fpath = os.path.join(IMG_DIR, fname)
#             if os.path.exists(fpath):
#                 try:
#                     img = pygame.image.load(fpath).convert_alpha()
#                     self.tile_imgs[i] = img
#                 except:
#                     pass
#
#     def reset_game(self):
#         print("🔄 正在重置游戏...")
#         pygame.event.clear()
#         self.env = MahjongEnv()
#         self.obs = self.env.reset()
#         self.last_drawn_tile = None
#
#         last_draw_pid = -1
#         if self.env.action_history:
#             last_rec = self.env.action_history[-1]
#             if last_rec['action'] == Cfg.ACT_DRAW:
#                 last_draw_pid = last_rec['pid']
#
#         new_dealer = random.randint(0, 3)
#         if last_draw_pid != -1 and new_dealer != 0:
#             h0 = self.env.players[0]['hand']
#             valid_tiles = [t for t in range(34) if h0[t] > 0]
#             if valid_tiles:
#                 move_tile = random.choice(valid_tiles)
#                 self.env.players[0]['hand'][move_tile] -= 1
#                 self.env.players[new_dealer]['hand'][move_tile] += 1
#                 self.env.dealer = new_dealer
#                 self.env.current_player = new_dealer
#                 self.env.incoming_tile = move_tile
#                 self.env.action_history.append({'pid': new_dealer, 'action': Cfg.ACT_DRAW, 'snapshot': None})
#                 if new_dealer == self.human_pid:
#                     self.last_drawn_tile = move_tile
#
#         if self.env.dealer == self.human_pid:
#             h_human = self.env.players[self.human_pid]['hand']
#             valid_tiles = [t for t in range(34) if h_human[t] > 0]
#             if valid_tiles:
#                 self.last_drawn_tile = valid_tiles[-1]
#
#         self.obs = self.env.get_observation(self.env.current_player)
#         self.done = False
#         self.steps = 0
#         self.info_text = f"游戏开始! 庄家: A{self.env.dealer}"
#         self.active_buttons = []
#         self.human_hand_rects = []
#         print(f"✅ 重置完成. 庄:A{self.env.dealer}, 初始高亮: {TileUtils.to_string(self.last_drawn_tile)}")
#
#     def _draw_tile_img(self, tile_id, x, y, w, h):
#         if tile_id in self.tile_imgs:
#             img = pygame.transform.smoothscale(self.tile_imgs[tile_id], (w, h))
#             self.screen.blit(img, (x, y))
#             return True
#         return False
#
#     def _draw_tile(self, tile_id, x, y, scale=1.0, is_laizi=False, special_text=None, is_hidden=False, highlight=False):
#         w = int(TILE_WIDTH * scale)
#         h = int(TILE_HEIGHT * scale)
#         offset_y = -20 if highlight else 0
#         draw_rect = pygame.Rect(x, y + offset_y, w, h)
#
#         if is_hidden:
#             pygame.draw.rect(self.screen, (220, 220, 220), draw_rect, border_radius=4)
#             inner = pygame.Rect(x + 2, y + 2 + offset_y, w - 4, h - 4)
#             pygame.draw.rect(self.screen, TILE_BACK_COLOR, inner, border_radius=3)
#             pygame.draw.rect(self.screen, (50, 150, 80), inner, 1)
#             return
#
#         pygame.draw.rect(self.screen, (250, 248, 235), draw_rect, border_radius=4)
#         drawn = False
#         if special_text is None and tile_id != -1:
#             drawn = self._draw_tile_img(tile_id, x, y + offset_y, w, h)
#
#         if not drawn:
#             pygame.draw.rect(self.screen, (100, 100, 100), draw_rect, 1, border_radius=4)
#             text = special_text if special_text else TileUtils.to_string(tile_id)
#             color = (0, 0, 0)
#             if "万" in text or text == "中":
#                 color = (180, 0, 0)
#             elif "索" in text or text == "发":
#                 color = (0, 120, 0)
#             elif "筒" in text or text == "白":
#                 color = (0, 0, 160)
#             elif tile_id >= 34:
#                 color = (160, 32, 240)
#
#             f = self.font if scale >= 0.9 else self.font_small
#             if len(text) > 1 and scale < 1.0: f = pygame.font.Font(self.font_path, int(FONT_SIZE * scale * 0.6))
#             s = f.render(text, True, color)
#             s_r = s.get_rect(center=draw_rect.center)
#             self.screen.blit(s, s_r)
#
#         if is_laizi:
#             pygame.draw.rect(self.screen, (255, 215, 0), draw_rect, 3, border_radius=4)
#         if highlight:
#             pygame.draw.rect(self.screen, (255, 30, 30), draw_rect, 3, border_radius=4)
#
#     def _draw_player_hand(self, pid, cx, cy):
#         player = self.env.players[pid]
#         hand_counts = player['hand'].copy()
#         is_human = (pid == self.human_pid)
#         should_hide = (not is_human) and (not self.done)
#
#         separate_tile = None
#         if self.env.current_player == pid and self.env.phase == 'DISCARD':
#             is_fresh_draw = False
#             if self.env.action_history:
#                 last_rec = self.env.action_history[-1]
#                 if last_rec['pid'] == pid and last_rec['action'] == Cfg.ACT_DRAW:
#                     is_fresh_draw = True
#             if is_fresh_draw:
#                 target_tile = -1
#                 if is_human and self.last_drawn_tile is not None:
#                     target_tile = self.last_drawn_tile
#                 elif not is_human:
#                     valid_idx = np.where(hand_counts > 0)[0]
#                     if len(valid_idx) > 0: target_tile = valid_idx[-1]
#                 if target_tile != -1 and 0 <= target_tile < 34:
#                     if hand_counts[target_tile] > 0:
#                         separate_tile = target_tile
#                         hand_counts[target_tile] -= 1
#
#         hand_tiles = []
#         for t_id, count in enumerate(hand_counts):
#             hand_tiles.extend([t_id] * count)
#
#         # [修改] 使用具体的花赖ID列表，而不是 -1
#         if 'flower_laizi_ids' in player:
#             hand_tiles.extend(player['flower_laizi_ids'])
#         else:
#             # 兼容旧代码或未初始化的情形
#             hand_tiles.extend([-1] * player['flower_laizis'])
#
#         base_width = len(hand_tiles) * (TILE_WIDTH + 2)
#         total_width = base_width + (TILE_WIDTH + 25) if separate_tile is not None else base_width
#
#         if pid == 0:
#             start_x = cx - total_width // 2
#             start_y = self.H - 140
#             meld_x = start_x + total_width + 20
#             meld_y = start_y + 10
#             flower_x = start_x - 120
#             flower_y = start_y
#             self.human_hand_rects = []
#         elif pid == 1:
#             start_x = self.W - total_width - 50
#             start_y = cy - 60
#             meld_x = self.W - 680
#             meld_y = start_y + TILE_HEIGHT + 15
#             flower_x = start_x
#             flower_y = start_y - 60
#         elif pid == 2:
#             start_x = cx - total_width // 2
#             start_y = 60
#             meld_x = start_x - 20 - (len(player['melds']) * TILE_WIDTH * 2.8)
#             meld_y = start_y + 10
#             flower_x = start_x + total_width + 50
#             flower_y = start_y
#         elif pid == 3:
#             start_x = 50
#             start_y = cy - 60
#             meld_x = start_x
#             meld_y = start_y + TILE_HEIGHT + 15
#             flower_x = start_x
#             flower_y = start_y - 60
#
#         if self.env.dealer == pid:
#             z_s = self.font_small.render("庄", True, (255, 0, 0))
#             z_x = start_x - 30 if pid in [0, 2, 3] else start_x + total_width + 10
#             pygame.draw.circle(self.screen, (255, 255, 255), (z_x + 10, start_y + 10), 12)
#             self.screen.blit(z_s, (z_x + 2, start_y))
#
#         for i, tid in enumerate(hand_tiles):
#             dx = i * (TILE_WIDTH + 2)
#             is_lz = (tid in self.env.laizi_set)
#
#             # [修改] 如果 tid 正常 (如34)，TileUtils 会返回 "春"，不再是 "花赖"
#             # 只有当数据结构未更新导致 tid=-1 时才显示 "花赖"
#             txt = "花赖" if (tid == -1 and self.env.laizi_set) else ("花" if tid == -1 else None)
#
#             self._draw_tile(tid, start_x + dx, start_y, is_laizi=is_lz, special_text=txt, is_hidden=should_hide)
#             if is_human:
#                 rect = pygame.Rect(start_x + dx, start_y, TILE_WIDTH, TILE_HEIGHT)
#                 self.human_hand_rects.append((rect, tid))
#
#         if separate_tile is not None:
#             sep_x = start_x + base_width + 25
#             is_lz = (separate_tile in self.env.laizi_set)
#             self._draw_tile(separate_tile, sep_x, start_y, is_laizi=is_lz, is_hidden=should_hide, highlight=True)
#             if is_human:
#                 rect = pygame.Rect(sep_x, start_y - 20, TILE_WIDTH, TILE_HEIGHT)
#                 self.human_hand_rects.append((rect, separate_tile))
#
#         for i, fid in enumerate(player['flowers']):
#             r = i // 4;
#             c = i % 4
#             self._draw_tile(fid, flower_x + c * 35, flower_y + r * 45, scale=0.8)
#
#         self._draw_melds(player, meld_x, meld_y)
#         if self.env.current_player == pid:
#             ind_x = start_x - 20
#             ind_y = start_y + TILE_HEIGHT // 2
#             pygame.draw.circle(self.screen, (255, 0, 0), (ind_x, ind_y), 8)
#             pygame.draw.circle(self.screen, (255, 255, 255), (ind_x, ind_y), 10, 2)
#
#     def _draw_melds(self, player, start_x, start_y):
#         for i, (m_type, m_tile) in enumerate(player['melds']):
#             offset_x = i * (TILE_WIDTH * 2.8)
#             tiles = []
#             label = ""
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
#             lbl = self.font_small.render(label, True, (255, 200, 0))
#             self.screen.blit(lbl, (start_x + offset_x, start_y - 18))
#
#     def _draw_river(self, pid, cx, cy):
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
#             r = i // cols;
#             c = i % cols
#             self._draw_tile(tid, sx + c * (w + 2), sy + r * (h + 2), scale=RIVER_SCALE,
#                             is_laizi=(tid in self.env.laizi_set))
#
#     def _draw_hud(self):
#         panel = pygame.Surface((320, 180))
#         panel.set_alpha(180)
#         panel.fill((0, 0, 0))
#         self.screen.blit(panel, (20, 20))
#         lz_str = " ".join([TileUtils.to_string(l) for l in self.env.laizi_set])
#         last_str = "-"
#         if self.env.last_discard is not None:
#             last_str = f"A{self.env.last_discard_pid} 打 {TileUtils.to_string(self.env.last_discard)}"
#         texts = [
#             f"剩余牌数: {len(self.env.wall)}",
#             f"本局赖子: {lz_str}",
#             f"庄家: A{self.env.dealer} | 骰子: {self.env.dice}",
#             f"我的状态: {self.env.phase}",
#             f"上一动作: {last_str}",
#             "ESC退出 | R重开"
#         ]
#         for i, t in enumerate(texts):
#             color = (255, 255, 255)
#             if "我的状态" in t and self.env.current_player == self.human_pid:
#                 color = (255, 215, 0)
#             s = self.font.render(t, True, color)
#             self.screen.blit(s, (30, 30 + i * 28))
#
#     def _draw_interaction_panel(self):
#         if self.env.current_player != self.human_pid: return
#         if self.done: return
#         mask = self.obs['mask']
#         special_acts = {
#             Cfg.ACT_PASS: "过", Cfg.ACT_HU: "胡", Cfg.ACT_GANG: "杠", Cfg.ACT_PON: "碰",
#             Cfg.ACT_CHI_LEFT: "左吃", Cfg.ACT_CHI_MID: "中吃", Cfg.ACT_CHI_RIGHT: "右吃"
#         }
#         actions_available = []
#         for act_id, label in special_acts.items():
#             if mask[act_id] == 1.0: actions_available.append((act_id, label))
#         if not actions_available: return
#
#         self.active_buttons = []
#         btn_w, btn_h = 100, 50
#         gap = 20
#         total_w = len(actions_available) * (btn_w + gap)
#         start_x = (self.W - total_w) // 2
#         start_y = self.H - 220
#
#         for i, (act_id, label) in enumerate(actions_available):
#             bx = start_x + i * (btn_w + gap)
#             rect = pygame.Rect(bx, start_y, btn_w, btn_h)
#             mouse_pos = pygame.mouse.get_pos()
#             color = BTN_HOVER_COLOR if rect.collidepoint(mouse_pos) else BTN_COLOR
#             pygame.draw.rect(self.screen, color, rect, border_radius=8)
#             pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=8)
#             txt_surf = self.font_btn.render(label, True, BTN_TEXT_COLOR)
#             txt_rect = txt_surf.get_rect(center=rect.center)
#             self.screen.blit(txt_surf, txt_rect)
#             self.active_buttons.append((rect, act_id))
#
#     def handle_human_click(self, pos):
#         if self.done: return False
#         if self.env.current_player != self.human_pid: return False
#         mask = self.obs['mask']
#         for rect, act_id in self.active_buttons:
#             if rect.collidepoint(pos):
#                 self._execute_step(act_id)
#                 return True
#         if self.env.phase == 'DISCARD':
#             for rect, tile_id in self.human_hand_rects:
#                 if rect.collidepoint(pos):
#                     if mask[tile_id] == 1.0:
#                         self._execute_step(tile_id)
#                         return True
#                     else:
#                         print(f"不可出牌: {TileUtils.to_string(tile_id)}")
#         return False
#
#     def _execute_step(self, action):
#         prev_hand_count = self.env.players[self.human_pid]['hand'].copy()
#         self.obs, reward, self.done, info = self.env.step(action)
#         self.steps += 1
#         if not self.done and self.env.current_player == self.human_pid and self.env.phase == 'DISCARD':
#             curr_hand_count = self.env.players[self.human_pid]['hand']
#             diff = curr_hand_count - prev_hand_count
#             added_indices = np.where(diff > 0)[0]
#             if len(added_indices) > 0:
#                 self.last_drawn_tile = added_indices[0]
#             else:
#                 self.last_drawn_tile = None
#         elif self.env.current_player != self.human_pid:
#             self.last_drawn_tile = None
#         if self.done:
#             self.active_buttons = []
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
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False
#                 elif event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_ESCAPE:
#                         running = False
#                     elif event.key == pygame.K_r:
#                         self.reset_game()
#                 elif event.type == pygame.MOUSEBUTTONDOWN:
#                     if not self.done and event.button == 1: self.handle_human_click(event.pos)
#             if not self.done:
#                 if self.env.current_player != self.human_pid:
#                     pygame.time.wait(1000)
#                     action, _, _ = self.agent.select_action(self.obs, eval_mode=True)
#                     self._execute_step(action)
#             self.screen.fill(BG_COLOR)
#             cx, cy = self.W // 2, self.H // 2
#             for i in range(4): self._draw_river(i, cx, cy)
#             for i in range(4): self._draw_player_hand(i, cx, cy)
#             self._draw_hud()
#             self._draw_interaction_panel()
#             if self.env.current_player == self.human_pid:
#                 status = "轮到你了! 请出牌或选择操作"
#                 color = (255, 255, 0)
#             else:
#                 status = f"AI (A{self.env.current_player}) 思考中..."
#                 color = (200, 200, 200)
#             if self.done: status = self.info_text
#             tip = self.font.render(status, True, color)
#             self.screen.blit(tip, (20, self.H - 40))
#             pygame.display.flip()
#             self.clock.tick(30)
#         pygame.quit()
#
#
# if __name__ == "__main__":
#     try:
#         app = InteractiveMahjong()
#         app.run()
#     except Exception as e:
#         import traceback
#
#         traceback.print_exc()
#         input("Error! Press Enter to exit...")
#
#
