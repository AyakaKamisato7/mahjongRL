import numpy as np
import random
from config import MahjongConfig as Cfg


class TileUtils:
    """
    [工具类] 负责牌的ID转换、打印
    """

    @staticmethod
    def to_string(tile_id):
        if tile_id is None: return "None"
        if 0 <= tile_id <= 8: return f"{tile_id + 1}万"
        if 9 <= tile_id <= 17: return f"{tile_id - 9 + 1}筒"
        if 18 <= tile_id <= 26: return f"{tile_id - 18 + 1}索"
        winds = ["东", "南", "西", "北"]
        if 27 <= tile_id <= 30: return winds[tile_id - 27]
        dragons = ["中", "发", "白"]
        if 31 <= tile_id <= 33: return dragons[tile_id - 31]
        flowers = ["春", "夏", "秋", "冬", "梅", "兰", "竹", "菊"]
        if 34 <= tile_id <= 41: return flowers[tile_id - 34]
        return "未知"

    @staticmethod
    def get_tile_type(tile_id):
        if 0 <= tile_id <= 8: return 0
        if 9 <= tile_id <= 17: return 1
        if 18 <= tile_id <= 26: return 2
        if 27 <= tile_id <= 33: return 3
        return 4


class FengHuaRules:
    """
    [核心逻辑层] 奉化麻将数学公理
    """

    @staticmethod
    def calculate_laizi_set(indicator_id):
        """
        [逻辑] 根据指示牌返回赖子 ID 集合 (Set)
        """
        laizi_set = set()

        # --- 情况 A: 翻出花牌 (34-41) ---
        if indicator_id >= 34:
            if 34 <= indicator_id <= 37:  # 春夏秋冬
                group = {34, 35, 36, 37}
                group.remove(indicator_id)
                return group
            elif 38 <= indicator_id <= 41:  # 梅兰竹菊
                group = {38, 39, 40, 41}
                group.remove(indicator_id)
                return group

        # --- 情况 B: 普通牌 (本位牌即赖子) ---
        else:
            laizi_set.add(indicator_id)

        return laizi_set

    @staticmethod
    def is_winning(hand_counts, laizi_set, extra_laizi_cnt=0, check_special=True):
        """
        [胡牌判定] 强制 Standard 模式
        """
        # 1. 提取手牌中的赖子 (转换为万能牌计数)
        # 注意：这里 hand_counts 仅包含 Standing Hand，不含副露。
        # 副露中的牌不在此数组中，因此天然满足“赖子进副露失效”的规则。

        total_laizi_cnt = extra_laizi_cnt
        hand_pure = hand_counts.copy()

        for l_id in laizi_set:
            if l_id < 34:
                count = hand_pure[l_id]
                if count > 0:
                    total_laizi_cnt += count
                    hand_pure[l_id] = 0

        # [屏蔽] 特殊牌型 (乱风 / 十三烂 / 清风特判)
        # if check_special: ...

        # [激活] 标准胡牌
        if FengHuaRules._check_standard_win(hand_pure, total_laizi_cnt):
            return True, "Standard"

        return False, None

    @staticmethod
    def _check_standard_win(hand_pure, laizi_cnt):
        """[算法] 标准胡牌递归回溯"""
        current_tiles = sum(hand_pure) + laizi_cnt
        if current_tiles % 3 != 2: return False

        # 1. 尝试将牌
        for t in range(34):
            if hand_pure[t] >= 2:
                hand_pure[t] -= 2
                if FengHuaRules._check_melds(hand_pure, laizi_cnt, 0):
                    hand_pure[t] += 2
                    return True
                hand_pure[t] += 2
        # 2. 赖子将
        if laizi_cnt >= 1:
            for t in range(34):
                if hand_pure[t] >= 1:
                    hand_pure[t] -= 1
                    if FengHuaRules._check_melds(hand_pure, laizi_cnt - 1, 0):
                        hand_pure[t] += 1
                        return True
                    hand_pure[t] += 1
        # 3. 双赖子将
        if laizi_cnt >= 2:
            if FengHuaRules._check_melds(hand_pure, laizi_cnt - 2, 0):
                return True
        return False

    @staticmethod
    def _check_melds(hand, laizi, idx):
        if idx >= 34: return laizi % 3 == 0
        if hand[idx] == 0: return FengHuaRules._check_melds(hand, laizi, idx + 1)

        # 刻子
        if hand[idx] >= 3:
            hand[idx] -= 3
            if FengHuaRules._check_melds(hand, laizi, idx):
                hand[idx] += 3
                return True
            hand[idx] += 3
        if hand[idx] < 3 and laizi >= (3 - hand[idx]):
            needed = 3 - hand[idx]
            orig = hand[idx]
            hand[idx] = 0
            if FengHuaRules._check_melds(hand, laizi - needed, idx + 1):
                hand[idx] = orig
                return True
            hand[idx] = orig
        # 顺子
        if idx < 27 and (idx % 9) <= 6:
            if hand[idx + 1] > 0 and hand[idx + 2] > 0:
                hand[idx] -= 1
                hand[idx + 1] -= 1
                hand[idx + 2] -= 1
                if FengHuaRules._check_melds(hand, laizi, idx):
                    hand[idx] += 1
                    hand[idx + 1] += 1
                    hand[idx + 2] += 1
                    return True
                hand[idx] += 1
                hand[idx + 1] += 1
                hand[idx + 2] += 1
        return False

    @staticmethod
    def get_valid_actions(hand_counts, melds, game_state_info, extra_laizi_cnt=0):
        mask = np.zeros(Cfg.ACTION_DIM, dtype=np.float32)

        # --- 阶段 1: DISCARD ---
        if game_state_info['phase'] == 'DISCARD':
            for i in range(34):
                if hand_counts[i] > 0: mask[i] = 1.0

            # 暗杠 (可以暗杠赖子)
            for i in range(34):
                if hand_counts[i] == 4: mask[Cfg.ACT_GANG] = 1.0

            # 补杠
            for m_type, m_tile in melds:
                if m_type == 'PON' and hand_counts[m_tile] > 0: mask[Cfg.ACT_GANG] = 1.0

            is_win, _ = FengHuaRules.is_winning(hand_counts, game_state_info['laizi_set'], extra_laizi_cnt)
            if is_win: mask[Cfg.ACT_HU] = 1.0

        # --- 阶段 2: RESPONSE ---
        elif game_state_info['phase'] == 'RESPONSE':
            tile_in = game_state_info['incoming_tile']
            mask[Cfg.ACT_PASS] = 1.0

            # [关键] 只有当 tile_in 不在赖子集合中时，才允许 碰/明杠
            is_laizi_discard = (tile_in in game_state_info['laizi_set'])

            if not is_laizi_discard:
                if hand_counts[tile_in] >= 2: mask[Cfg.ACT_PON] = 1.0
                if hand_counts[tile_in] == 3: mask[Cfg.ACT_GANG] = 1.0

            # 吃 (天然限制了赖子不能当万能牌吃，因为手牌ID不匹配)
            if game_state_info['is_left_player']:
                if tile_in < 27:
                    rank = tile_in % 9
                    if rank <= 6:
                        if hand_counts[tile_in + 1] > 0 and hand_counts[tile_in + 2] > 0: mask[Cfg.ACT_CHI_LEFT] = 1.0
                    if 1 <= rank <= 7:
                        if hand_counts[tile_in - 1] > 0 and hand_counts[tile_in + 1] > 0: mask[Cfg.ACT_CHI_MID] = 1.0
                    if rank >= 2:
                        if hand_counts[tile_in - 2] > 0 and hand_counts[tile_in - 1] > 0: mask[Cfg.ACT_CHI_RIGHT] = 1.0

            # 点炮
            temp_hand = hand_counts.copy()
            temp_hand[tile_in] += 1
            is_win, _ = FengHuaRules.is_winning(temp_hand, game_state_info['laizi_set'], extra_laizi_cnt)
            if is_win: mask[Cfg.ACT_HU] = 1.0

        return mask


class MahjongEnv:
    def __init__(self, config=None):
        self.cfg = config if config else Cfg()
        self.rules = FengHuaRules()
        self.reset()

    def reset(self):
        self.wall = []
        for i in range(self.cfg.NUM_SUIT_TILES): self.wall.extend([i] * 4)
        for i in range(self.cfg.ID_FLOWER_START, self.cfg.ID_FLOWER_START + self.cfg.NUM_FLOWERS): self.wall.append(i)
        random.shuffle(self.wall)

        self.dice = [random.randint(1, 6), random.randint(1, 6)]
        dice_sum = sum(self.dice)
        indicator_idx = -1 * (dice_sum % len(self.wall))
        if indicator_idx == 0: indicator_idx = -1
        self.indicator_tile = self.wall.pop(indicator_idx)
        self.laizi_set = self.rules.calculate_laizi_set(self.indicator_tile)

        self.players = []
        for _ in range(4):
            self.players.append({
                'hand': np.zeros(34, dtype=np.int32),
                'flower_laizis': 0,
                'melds': [],
                'flowers': [],
                'score': 0
            })

        # [初始化] 必须在发牌前初始化 history
        self.action_history = []
        self.response_queue = []

        self.dealer = 0
        for pid in range(4): self._deal_initial_hand(pid, 13)
        self._player_draw(self.dealer)

        self.current_player = self.dealer
        self.phase = 'DISCARD'
        self.last_discard = None
        self.last_discard_pid = None
        self.incoming_tile = None

        self.gang_flag = False
        self.first_turn = True
        return self.get_observation(self.current_player)

    def _record_snapshot(self, pid, action):
        snapshot = []
        for p in self.players:
            snapshot.append({
                'hand': p['hand'].copy(),
                'melds': p['melds'].copy(),
                'flower_laizis': p['flower_laizis'],
                'flowers': p['flowers'].copy()
            })
        self.action_history.append({'pid': pid, 'action': action, 'snapshot': snapshot})

    def _player_draw(self, pid):
        while True:
            if len(self.wall) == 0: return False
            tile = self.wall.pop(0)

            if tile >= 34:
                if tile in self.laizi_set:
                    self.players[pid]['flower_laizis'] += 1
                    self._record_snapshot(pid, self.cfg.ACT_DRAW)
                    return True
                else:
                    self.players[pid]['flowers'].append(tile)
                    continue

            self.players[pid]['hand'][tile] += 1
            self._record_snapshot(pid, self.cfg.ACT_DRAW)
            return True

    def _deal_initial_hand(self, pid, count):
        while count > 0:
            tile = self.wall.pop(0)
            if tile >= 34:
                if tile in self.laizi_set:
                    self.players[pid]['flower_laizis'] += 1
                    count -= 1
                else:
                    self.players[pid]['flowers'].append(tile)
            else:
                self.players[pid]['hand'][tile] += 1
                count -= 1

    def _perform_gang(self, pid, incoming_tile=None):
        """
        [修复] 杠牌逻辑：扣除手牌 -> 记录 -> 补牌
        """
        player = self.players[pid]

        if incoming_tile is not None:
            # 明杠
            player['hand'][incoming_tile] -= 3
            player['melds'].append(('GANG', incoming_tile))
        else:
            # 暗杠或补杠
            gang_tile = -1
            # 暗杠
            for t in range(34):
                if player['hand'][t] == 4:
                    gang_tile = t
                    player['hand'][t] -= 4
                    player['melds'].append(('GANG', t))
                    break

            # 补杠
            if gang_tile == -1:
                for idx, (m_type, m_tile) in enumerate(player['melds']):
                    if m_type == 'PON' and player['hand'][m_tile] == 1:
                        gang_tile = m_tile
                        player['hand'][m_tile] -= 1
                        player['melds'][idx] = ('GANG', m_tile)
                        break

        self._record_snapshot(pid, self.cfg.ACT_GANG)

        # 杠后必补牌
        self.gang_flag = True
        return self._player_draw(pid)

    def calculate_final_score(self, pid, win_type, context=None):
        if context is None: context = {}
        score = self.cfg.R_WIN_BASE
        player = self.players[pid]
        multiplier = 1.0

        # 牌型倍率
        if win_type == 'ThirteenLan':
            multiplier += self.cfg.MULT_THIRTEEN_LAN
        elif win_type == 'LuanFeng':
            multiplier += self.cfg.MULT_LUAN_FENG
        elif win_type == 'QingFeng':
            multiplier += self.cfg.MULT_QING_FENG

        if context.get('is_hai_di', False): multiplier += self.cfg.TAI_HAI_DI
        if context.get('is_gang_kai', False): multiplier += self.cfg.TAI_GANG_KAI
        if context.get('is_tian_hu', False): multiplier += self.cfg.MULT_TIAN_HU
        if context.get('is_di_hu', False): multiplier += self.cfg.MULT_DI_HU
        if len(player['melds']) == 4: multiplier += self.cfg.TAI_QUAN_QIU

        if win_type in ['Standard', 'QingFeng']:
            suits = set()
            has_honor = False
            for t in range(34):
                if player['hand'][t] > 0 and t not in self.laizi_set:
                    if t >= 27:
                        has_honor = True
                    else:
                        suits.add(t // 9)
            for m_type, m_tile in player['melds']:
                if m_tile not in self.laizi_set:
                    if m_tile >= 27:
                        has_honor = True
                    else:
                        suits.add(m_tile // 9)
            if len(suits) == 1:
                if not has_honor:
                    multiplier += self.cfg.TAI_QING_YI_SE
                else:
                    multiplier += self.cfg.TAI_HUN_YI_SE

        f_counts = [0] * 42
        for f in player['flowers']: f_counts[f] = 1
        if sum(f_counts[34:38]) == 4: multiplier += self.cfg.TAI_FLOWER_SET
        if sum(f_counts[38:42]) == 4: multiplier += self.cfg.TAI_FLOWER_SET

        total_laizi = player['flower_laizis']
        hand_pure = player['hand']
        for l_id in self.laizi_set:
            if l_id < 34: total_laizi += hand_pure[l_id]
        if total_laizi >= 3: multiplier += self.cfg.MULT_LAIZI_ALL
        return score * multiplier

    def get_observation(self, pid):
        state_info = {
            'phase': self.phase,
            'laizi_set': self.laizi_set,
            'incoming_tile': self.incoming_tile,
            'is_left_player': False
        }
        if self.phase == 'RESPONSE' and self.last_discard_pid is not None:
            if (self.last_discard_pid + 1) % 4 == pid:
                state_info['is_left_player'] = True

        valid_mask = self.rules.get_valid_actions(
            self.players[pid]['hand'], self.players[pid]['melds'], state_info,
            extra_laizi_cnt=self.players[pid]['flower_laizis']
        )

        seq_len = self.cfg.HISTORY_LEN
        raw_hist = self.action_history[-seq_len:]
        formatted_hist = []
        for item in raw_hist:
            p_idx = item['pid']
            act_id = item['action']
            if act_id == self.cfg.ACT_DRAW: continue

            rel_p = (p_idx - pid + 4) % 4
            formatted_hist.append((rel_p, act_id))

        while len(formatted_hist) < seq_len:
            formatted_hist.insert(0, (-1, -1))

        obs = {
            "hand": self.players[pid]['hand'].copy(),
            "melds": self.players[pid]['melds'],
            "mask": valid_mask,
            "laizi_set": self.laizi_set,
            "flower_laizis": self.players[pid]['flower_laizis'],
            "flowers": self.players[pid]['flowers'],
            "history": formatted_hist
        }
        return obs

    def step(self, action):
        pid = self.current_player
        player = self.players[pid]
        reward = 0
        done = False
        info = {}

        is_hai_di = (len(self.wall) == 0)
        is_gang_kai = self.gang_flag
        is_first_turn = (len(self.action_history) < 4)

        state_info = {
            'phase': self.phase,
            'laizi_set': self.laizi_set,
            'incoming_tile': self.incoming_tile,
            'is_left_player': (self.last_discard_pid + 1) % 4 == pid if self.phase == 'RESPONSE' else False
        }
        mask = self.rules.get_valid_actions(
            player['hand'], player['melds'], state_info, extra_laizi_cnt=player['flower_laizis']
        )
        if mask[action] == 0:
            return self.get_observation(pid), self.cfg.R_INVALID, True, {"error": "Invalid Action"}

        # 记录 Action (除 Gang 外，Gang 内部记录)
        if action != self.cfg.ACT_GANG:
            self._record_snapshot(pid, action)

        # 2. 执行逻辑
        if self.phase == 'DISCARD':
            if action != self.cfg.ACT_GANG: self.gang_flag = False

            if action <= 33:  # 打牌
                player['hand'][action] -= 1
                self.last_discard = action
                self.last_discard_pid = pid
                self.incoming_tile = action

                # 响应队列生成
                self.phase = 'RESPONSE'
                self.response_queue = []
                for offset in [1, 2, 3]:
                    target_pid = (pid + offset) % 4
                    t_state = {
                        'phase': 'RESPONSE',
                        'laizi_set': self.laizi_set,
                        'incoming_tile': self.incoming_tile,
                        'is_left_player': (offset == 1)
                    }
                    t_mask = self.rules.get_valid_actions(
                        self.players[target_pid]['hand'], self.players[target_pid]['melds'], t_state,
                        extra_laizi_cnt=self.players[target_pid]['flower_laizis']
                    )
                    can_act = False
                    for act_i in range(Cfg.ACTION_DIM):
                        if act_i != Cfg.ACT_PASS and t_mask[act_i] > 0:
                            can_act = True
                            break
                    if can_act:
                        self.response_queue.append(target_pid)

                if not self.response_queue:
                    self.phase = 'DISCARD'
                    next_pid = (pid + 1) % 4
                    self.current_player = next_pid
                    if not self._player_draw(next_pid):
                        done = True
                        info['winner'] = None
                else:
                    self.current_player = self.response_queue[0]

            elif action == self.cfg.ACT_HU:  # 自摸
                is_win, w_type = self.rules.is_winning(player['hand'], self.laizi_set,
                                                       extra_laizi_cnt=player['flower_laizis'])
                if not is_win: return self.get_observation(pid), self.cfg.R_INVALID, True, {"error": "False HU"}

                ctx = {'is_hai_di': is_hai_di, 'is_gang_kai': is_gang_kai,
                       'is_tian_hu': (is_first_turn and pid == self.dealer)}
                final_score = self.calculate_final_score(pid, w_type, context=ctx)
                reward = final_score
                done = True
                info['winner'] = pid
                info['win_type'] = 'zimo'

            elif action == self.cfg.ACT_GANG:  # 暗杠/补杠
                if not self._perform_gang(pid):
                    done = True
                    info['winner'] = None

        elif self.phase == 'RESPONSE':
            self.gang_flag = False

            if action == self.cfg.ACT_PASS:
                if self.response_queue: self.response_queue.pop(0)
                if self.response_queue:
                    self.current_player = self.response_queue[0]
                else:
                    self.phase = 'DISCARD'
                    next_pid = (self.last_discard_pid + 1) % 4
                    self.current_player = next_pid
                    if not self._player_draw(next_pid):
                        done = True
                        info['winner'] = None

            elif action == self.cfg.ACT_PON:
                player['hand'][self.incoming_tile] -= 2
                player['melds'].append(('PON', self.incoming_tile))
                self.phase = 'DISCARD'
                self.response_queue = []

            elif action == self.cfg.ACT_HU:  # 点炮
                player['hand'][self.incoming_tile] += 1
                is_win, w_type = self.rules.is_winning(player['hand'], self.laizi_set,
                                                       extra_laizi_cnt=player['flower_laizis'])
                if not is_win: return self.get_observation(pid), self.cfg.R_INVALID, True, {"error": "False HU"}

                ctx = {'is_hai_di': is_hai_di, 'is_di_hu': (is_first_turn and pid != self.dealer)}
                final_score = self.calculate_final_score(pid, w_type, context=ctx)
                reward = final_score
                done = True
                info['winner'] = pid
                info['win_type'] = 'ron'
                info['loser'] = self.last_discard_pid

            elif action == self.cfg.ACT_CHI_LEFT:
                t = self.incoming_tile
                player['hand'][t + 1] -= 1;
                player['hand'][t + 2] -= 1
                # [修改] 标记为 CHI_L，方便GUI画图
                player['melds'].append(('CHI_L', t))
                self.phase = 'DISCARD'
                self.response_queue = []

            elif action == self.cfg.ACT_CHI_MID:
                t = self.incoming_tile
                player['hand'][t - 1] -= 1;
                player['hand'][t + 1] -= 1
                # [修改] 标记为 CHI_M
                player['melds'].append(('CHI_M', t))
                self.phase = 'DISCARD'
                self.response_queue = []

            elif action == self.cfg.ACT_CHI_RIGHT:
                t = self.incoming_tile
                player['hand'][t - 2] -= 1;
                player['hand'][t - 1] -= 1
                # [修改] 标记为 CHI_R
                player['melds'].append(('CHI_R', t))
                self.phase = 'DISCARD'
                self.response_queue = []

        return self.get_observation(self.current_player), reward, done, info