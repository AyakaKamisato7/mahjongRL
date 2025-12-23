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
        # [之前修复] 调整顺序以匹配图片资源
        dragons = ["白", "发", "中"]
        if 31 <= tile_id <= 33: return dragons[tile_id - 31]
        flowers = ["春", "夏", "秋", "冬", "梅", "兰", "竹", "菊"]
        if 34 <= tile_id <= 41: return flowers[tile_id - 34]
        return "未知"


class FengHuaRules:
    """
    [核心逻辑层] 奉化麻将数学公理
    """

    @staticmethod
    def calculate_laizi_set(indicator_id):
        laizi_set = set()
        if indicator_id >= 34:
            if 34 <= indicator_id <= 37:  # 春夏秋冬
                group = {34, 35, 36, 37}
                group.remove(indicator_id)
                return group
            elif 38 <= indicator_id <= 41:  # 梅兰竹菊
                group = {38, 39, 40, 41}
                group.remove(indicator_id)
                return group
        else:
            laizi_set.add(indicator_id)
        return laizi_set

    @staticmethod
    def is_winning(hand_counts, laizi_set, extra_laizi_cnt=0):
        total_laizi_cnt = extra_laizi_cnt
        hand_pure = hand_counts.copy()

        for l_id in laizi_set:
            if l_id < 34:
                count = hand_pure[l_id]
                if count > 0:
                    total_laizi_cnt += count
                    hand_pure[l_id] = 0

        if FengHuaRules._check_standard_win(hand_pure, total_laizi_cnt):
            return True, "Standard"
        return False, None

    @staticmethod
    def _check_standard_win(hand_pure, laizi_cnt):
        current_tiles = sum(hand_pure) + laizi_cnt
        if current_tiles % 3 != 2: return False

        # 1. 尝试自然将 (AA)
        for t in range(34):
            if hand_pure[t] >= 2:
                hand_pure[t] -= 2
                if FengHuaRules._check_melds(hand_pure, laizi_cnt, 0):
                    hand_pure[t] += 2
                    return True
                hand_pure[t] += 2

        # 2. 尝试 1张真牌 + 1赖子做将
        if laizi_cnt >= 1:
            for t in range(34):
                if hand_pure[t] >= 1:
                    hand_pure[t] -= 1
                    if FengHuaRules._check_melds(hand_pure, laizi_cnt - 1, 0):
                        hand_pure[t] += 1
                        return True
                    hand_pure[t] += 1

        # 3. 尝试 2赖子做将
        if laizi_cnt >= 2:
            if FengHuaRules._check_melds(hand_pure, laizi_cnt - 2, 0):
                return True
        return False

    @staticmethod
    def _check_melds(hand, laizi, idx):
        """
        [修复重点] 增加了赖子作为顺子中间和结尾的判定
        """
        # 跳过空牌
        while idx < 34 and hand[idx] == 0:
            idx += 1

        # 递归终止
        if idx >= 34:
            return laizi % 3 == 0

        # --- 分支 1: 组成刻子 (AAA) ---
        if hand[idx] + laizi >= 3:
            needed = max(0, 3 - hand[idx])
            orig_count = hand[idx]
            hand[idx] = max(0, hand[idx] - 3)

            if FengHuaRules._check_melds(hand, laizi - needed, idx):
                hand[idx] = orig_count
                return True
            hand[idx] = orig_count

        # --- 分支 2: 组成顺子 (ABC) ---
        # 我们需要考虑 hand[idx] 在顺子中的三个位置：开头、中间、结尾

        # 情况 A: hand[idx] 是顺子开头 (idx, idx+1, idx+2)
        if idx < 27 and (idx % 9) <= 6:
            c2 = 1 if hand[idx + 1] > 0 else 0
            c3 = 1 if hand[idx + 2] > 0 else 0
            needs = (1 - c2) + (1 - c3)

            if laizi >= needs:
                hand[idx] -= 1
                if c2: hand[idx + 1] -= 1
                if c3: hand[idx + 2] -= 1

                if FengHuaRules._check_melds(hand, laizi - needs, idx):
                    # 恢复并返回
                    hand[idx] += 1
                    if c2: hand[idx + 1] += 1
                    if c3: hand[idx + 2] += 1
                    return True

                # 回溯
                hand[idx] += 1
                if c2: hand[idx + 1] += 1
                if c3: hand[idx + 2] += 1

        # 情况 B: hand[idx] 是顺子中间 (idx-1, idx, idx+1)
        # 前提：idx-1 必须是合法的顺子开头。
        # 注意：因为循环是从小到大遍历且跳过了 0，能走到这里说明 hand[idx-1] 必定是 0。
        # 所以 idx-1 位置必须消耗 1 个赖子。
        if idx >= 1 and idx < 27 and ((idx - 1) % 9) <= 6:
            # 需要: 赖子(补idx-1) + 可能的(idx+1)
            c3 = 1 if hand[idx + 1] > 0 else 0  # 这里 c3 指的是顺子的第3张，即 idx+1
            needs = 1 + (1 - c3)  # 1是补idx-1

            if laizi >= needs:
                hand[idx] -= 1
                if c3: hand[idx + 1] -= 1

                if FengHuaRules._check_melds(hand, laizi - needs, idx):
                    hand[idx] += 1
                    if c3: hand[idx + 1] += 1
                    return True
                hand[idx] += 1
                if c3: hand[idx + 1] += 1

        # 情况 C: hand[idx] 是顺子结尾 (idx-2, idx-1, idx)
        # 前提：idx-2 必须是合法的顺子开头。
        # 注意：hand[idx-2] 和 hand[idx-1] 都必定是 0。
        # 所以必须消耗 2 个赖子。
        if idx >= 2 and idx < 27 and ((idx - 2) % 9) <= 6:
            needs = 2
            if laizi >= needs:
                hand[idx] -= 1
                if FengHuaRules._check_melds(hand, laizi - needs, idx):
                    hand[idx] += 1
                    return True
                hand[idx] += 1

        return False

    @staticmethod
    def get_valid_actions(hand_counts, melds, game_state_info, extra_laizi_cnt=0):
        mask = np.zeros(Cfg.ACTION_DIM, dtype=np.float32)

        if game_state_info['phase'] == 'DISCARD':
            for i in range(34):
                if hand_counts[i] > 0: mask[i] = 1.0
            for i in range(34):
                if hand_counts[i] == 4: mask[Cfg.ACT_GANG] = 1.0
            for m_type, m_tile in melds:
                if m_type == 'PON' and hand_counts[m_tile] > 0: mask[Cfg.ACT_GANG] = 1.0

            is_win, _ = FengHuaRules.is_winning(hand_counts, game_state_info['laizi_set'], extra_laizi_cnt)
            if is_win: mask[Cfg.ACT_HU] = 1.0

        elif game_state_info['phase'] == 'RESPONSE':
            tile_in = game_state_info['incoming_tile']
            mask[Cfg.ACT_PASS] = 1.0
            is_laizi_discard = (tile_in in game_state_info['laizi_set'])

            if not is_laizi_discard:
                if hand_counts[tile_in] >= 2: mask[Cfg.ACT_PON] = 1.0
                if hand_counts[tile_in] == 3: mask[Cfg.ACT_GANG] = 1.0
                if game_state_info['is_left_player']:
                    if tile_in < 27:
                        rank = tile_in % 9
                        if rank <= 6 and hand_counts[tile_in + 1] > 0 and hand_counts[tile_in + 2] > 0:
                            mask[Cfg.ACT_CHI_LEFT] = 1.0
                        if 1 <= rank <= 7 and hand_counts[tile_in - 1] > 0 and hand_counts[tile_in + 1] > 0:
                            mask[Cfg.ACT_CHI_MID] = 1.0
                        if rank >= 2 and hand_counts[tile_in - 2] > 0 and hand_counts[tile_in - 1] > 0:
                            mask[Cfg.ACT_CHI_RIGHT] = 1.0

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
                'flower_laizis': 0,  # 计数 (用于逻辑计算)
                'flower_laizi_ids': [],  # [新增] 存储具体ID (用于UI显示)
                'melds': [],
                'flowers': [],
                'score': 0
            })

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
        self.global_step_counter = 0

        return self.get_observation(self.current_player)

    def _record_snapshot(self, pid, action):
        snapshot = []
        for p in self.players:
            snapshot.append({
                'hand': p['hand'].copy(),
                'melds': p['melds'].copy(),
                'flower_laizis': p['flower_laizis'],
                'flower_laizi_ids': list(p['flower_laizi_ids']),  # 记录ID列表
                'flowers': p['flowers'].copy()
            })
        self.action_history.append({'pid': pid, 'action': action, 'snapshot': snapshot})

    def _player_draw(self, pid):
        if len(self.wall) == 0: return False
        tile = self.wall.pop(0)

        if tile >= 34:
            if tile in self.laizi_set:
                self.players[pid]['flower_laizis'] += 1
                self.players[pid]['flower_laizi_ids'].append(tile)  # [新增] 记录具体ID
                self._record_snapshot(pid, self.cfg.ACT_DRAW)
                return True
            else:
                self.players[pid]['flowers'].append(tile)
                return self._player_draw(pid)

        self.players[pid]['hand'][tile] += 1
        self._record_snapshot(pid, self.cfg.ACT_DRAW)
        return True

    def _deal_initial_hand(self, pid, count):
        while count > 0:
            tile = self.wall.pop(0)
            if tile >= 34:
                if tile in self.laizi_set:
                    self.players[pid]['flower_laizis'] += 1
                    self.players[pid]['flower_laizi_ids'].append(tile)  # [新增]
                    count -= 1
                else:
                    self.players[pid]['flowers'].append(tile)
            else:
                self.players[pid]['hand'][tile] += 1
                count -= 1

    def _perform_gang(self, pid, incoming_tile=None):
        player = self.players[pid]
        gang_tile = -1

        if incoming_tile is not None and self.phase == 'RESPONSE':
            if player['hand'][incoming_tile] == 3:
                player['hand'][incoming_tile] -= 3
                player['melds'].append(('GANG', incoming_tile))
                gang_tile = incoming_tile
        else:
            for t in range(34):
                if player['hand'][t] == 4:
                    gang_tile = t
                    player['hand'][t] -= 4
                    player['melds'].append(('GANG', t))
                    break
            if gang_tile == -1:
                for idx, (m_type, m_tile) in enumerate(player['melds']):
                    if m_type == 'PON' and player['hand'][m_tile] >= 1:
                        gang_tile = m_tile
                        player['hand'][m_tile] -= 1
                        player['melds'][idx] = ('GANG', m_tile)
                        break

        if gang_tile != -1:
            self._record_snapshot(pid, self.cfg.ACT_GANG)
            self.gang_flag = True
            return self._player_draw(pid)
        else:
            return True

    def calculate_final_score(self, pid, win_type, context=None):
        score = self.cfg.R_WIN_BASE
        player = self.players[pid]
        multiplier = 1.0
        total_laizi = player['flower_laizis']
        hand_pure = player['hand']
        for l_id in self.laizi_set:
            if l_id < 34: total_laizi += hand_pure[l_id]
        if total_laizi >= 3: multiplier += 2.0
        if context and context.get('is_gang_kai'): multiplier += 1.0
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
            act_id = item['action']
            if act_id == self.cfg.ACT_DRAW: continue
            rel_p = (item['pid'] - pid + 4) % 4
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

        self.global_step_counter += 1
        STEP_THRESHOLD = 40
        if self.global_step_counter > STEP_THRESHOLD:
            diff = self.global_step_counter - STEP_THRESHOLD
            delay_penalty = 0.005 * (diff ** 2)
            reward -= delay_penalty

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
            return self.get_observation(pid), -50.0, True, {"error": "Invalid Action"}

        if action != self.cfg.ACT_GANG:
            self._record_snapshot(pid, action)

        if self.phase == 'DISCARD' and action <= 33:
            if action in self.laizi_set:
                reward -= 100.0

        if self.phase == 'DISCARD':
            if action != self.cfg.ACT_GANG: self.gang_flag = False

            if action <= 33:
                player['hand'][action] -= 1
                self.last_discard = action
                self.last_discard_pid = pid
                self.incoming_tile = action

                self.phase = 'RESPONSE'
                possible_responses = []
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
                    can_hu = (t_mask[Cfg.ACT_HU] == 1.0)
                    can_pon_gang = (t_mask[Cfg.ACT_PON] == 1.0 or t_mask[Cfg.ACT_GANG] == 1.0)
                    can_chi = (t_mask[Cfg.ACT_CHI_LEFT] == 1.0 or t_mask[Cfg.ACT_CHI_MID] == 1.0 or t_mask[
                        Cfg.ACT_CHI_RIGHT] == 1.0)

                    if can_hu or can_pon_gang or can_chi:
                        possible_responses.append(
                            {'pid': target_pid, 'offset': offset, 'hu': can_hu, 'pon': can_pon_gang, 'chi': can_chi})

                self.response_queue = []
                hu_players = [p for p in possible_responses if p['hu']]
                if hu_players:
                    hu_players.sort(key=lambda x: x['offset'])
                    self.response_queue.append(hu_players[0]['pid'])
                else:
                    pon_players = [p for p in possible_responses if p['pon']]
                    if pon_players:
                        self.response_queue.append(pon_players[0]['pid'])
                    else:
                        chi_players = [p for p in possible_responses if p['chi']]
                        if chi_players:
                            self.response_queue.append(chi_players[0]['pid'])

                if not self.response_queue:
                    self.phase = 'DISCARD'
                    next_pid = (pid + 1) % 4
                    self.current_player = next_pid
                    if not self._player_draw(next_pid):
                        done = True
                        info['winner'] = None
                else:
                    self.current_player = self.response_queue[0]

            elif action == self.cfg.ACT_HU:
                is_win, w_type = self.rules.is_winning(player['hand'], self.laizi_set,
                                                       extra_laizi_cnt=player['flower_laizis'])
                if not is_win: return self.get_observation(pid), -50, True, {"error": "False HU"}
                final_score = self.calculate_final_score(pid, w_type)
                reward += final_score * 20
                done = True
                info['winner'] = pid
                info['win_type'] = 'zimo'

            elif action == self.cfg.ACT_GANG:
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
            elif action == self.cfg.ACT_HU:
                player['hand'][self.incoming_tile] += 1
                is_win, w_type = self.rules.is_winning(player['hand'], self.laizi_set,
                                                       extra_laizi_cnt=player['flower_laizis'])
                if not is_win: return self.get_observation(pid), -50, True, {"error": "False HU"}
                final_score = self.calculate_final_score(pid, w_type)
                reward += final_score * 20
                done = True
                info['winner'] = pid
                info['win_type'] = 'ron'
                info['loser'] = self.last_discard_pid
            elif action == self.cfg.ACT_GANG:
                if not self._perform_gang(pid, incoming_tile=self.incoming_tile):
                    done = True
                    info['winner'] = None
                else:
                    self.phase = 'DISCARD'
                    self.response_queue = []
            elif action in [self.cfg.ACT_CHI_LEFT, self.cfg.ACT_CHI_MID, self.cfg.ACT_CHI_RIGHT]:
                t = self.incoming_tile
                if action == self.cfg.ACT_CHI_LEFT:
                    player['hand'][t + 1] -= 1;
                    player['hand'][t + 2] -= 1
                    player['melds'].append(('CHI_L', t))
                elif action == self.cfg.ACT_CHI_MID:
                    player['hand'][t - 1] -= 1;
                    player['hand'][t + 1] -= 1
                    player['melds'].append(('CHI_M', t))
                elif action == self.cfg.ACT_CHI_RIGHT:
                    player['hand'][t - 2] -= 1;
                    player['hand'][t - 1] -= 1
                    player['melds'].append(('CHI_R', t))
                self.phase = 'DISCARD'
                self.response_queue = []

        if not done:
            hand_laizi_count = player['flower_laizis']
            for l_id in self.laizi_set:
                if l_id < 34: hand_laizi_count += player['hand'][l_id]
            if hand_laizi_count > 0: reward += (hand_laizi_count * 3)
            for h_id in range(27, 34):
                if player['hand'][h_id] == 1 and h_id not in self.laizi_set: reward -= 2

        return self.get_observation(self.current_player), reward, done, info

# import numpy as np
# import random
# from config import MahjongConfig as Cfg
#
#
# class TileUtils:
#     """
#     [工具类] 负责牌的ID转换、打印
#     """
#
#     @staticmethod
#     def to_string(tile_id):
#         if tile_id is None: return "None"
#         if 0 <= tile_id <= 8: return f"{tile_id + 1}万"
#         if 9 <= tile_id <= 17: return f"{tile_id - 9 + 1}筒"
#         if 18 <= tile_id <= 26: return f"{tile_id - 18 + 1}索"
#         winds = ["东", "南", "西", "北"]
#         if 27 <= tile_id <= 30: return winds[tile_id - 27]
#         dragons = ["中", "发", "白"]
#         if 31 <= tile_id <= 33: return dragons[tile_id - 31]
#         flowers = ["春", "夏", "秋", "冬", "梅", "兰", "竹", "菊"]
#         if 34 <= tile_id <= 41: return flowers[tile_id - 34]
#         return "未知"
#
#
# class FengHuaRules:
#     """
#     [核心逻辑层] 奉化麻将数学公理
#     包含：赖子定义、标准胡牌算法(完全回溯修复版)、动作合法性校验
#     """
#
#     @staticmethod
#     def calculate_laizi_set(indicator_id):
#         """根据指示牌计算赖子集合"""
#         laizi_set = set()
#         if indicator_id >= 34:
#             if 34 <= indicator_id <= 37:  # 春夏秋冬
#                 group = {34, 35, 36, 37}
#                 group.remove(indicator_id)
#                 return group
#             elif 38 <= indicator_id <= 41:  # 梅兰竹菊
#                 group = {38, 39, 40, 41}
#                 group.remove(indicator_id)
#                 return group
#         else:
#             laizi_set.add(indicator_id)  # 本位牌即赖子
#         return laizi_set
#
#     @staticmethod
#     def is_winning(hand_counts, laizi_set, extra_laizi_cnt=0):
#         """
#         [胡牌判定] 标准胡牌逻辑
#         """
#         total_laizi_cnt = extra_laizi_cnt
#         hand_pure = hand_counts.copy()
#
#         # 提取手牌中的赖子
#         for l_id in laizi_set:
#             if l_id < 34:
#                 count = hand_pure[l_id]
#                 if count > 0:
#                     total_laizi_cnt += count
#                     hand_pure[l_id] = 0
#
#         if FengHuaRules._check_standard_win(hand_pure, total_laizi_cnt):
#             return True, "Standard"
#         return False, None
#
#     @staticmethod
#     def _check_standard_win(hand_pure, laizi_cnt):
#         """[DFS入口] 尝试移除将牌"""
#         current_tiles = sum(hand_pure) + laizi_cnt
#         if current_tiles % 3 != 2: return False
#
#         # 1. 尝试自然将 (AA)
#         for t in range(34):
#             if hand_pure[t] >= 2:
#                 hand_pure[t] -= 2
#                 if FengHuaRules._check_melds(hand_pure, laizi_cnt, 0):
#                     hand_pure[t] += 2
#                     return True
#                 hand_pure[t] += 2
#
#         # 2. 尝试 1张真牌 + 1赖子做将
#         if laizi_cnt >= 1:
#             for t in range(34):
#                 if hand_pure[t] >= 1:
#                     hand_pure[t] -= 1
#                     if FengHuaRules._check_melds(hand_pure, laizi_cnt - 1, 0):
#                         hand_pure[t] += 1
#                         return True
#                     hand_pure[t] += 1
#
#         # 3. 尝试 2赖子做将
#         if laizi_cnt >= 2:
#             if FengHuaRules._check_melds(hand_pure, laizi_cnt - 2, 0):
#                 return True
#         return False
#
#     @staticmethod
#     def _check_melds(hand, laizi, idx):
#         """
#         [核心递归] 检测是否全为面子 (刻子/顺子)
#         [修复] 采用完全分支尝试，解决多赖子情况下的贪心错误
#         """
#         # 跳过空牌
#         while idx < 34 and hand[idx] == 0:
#             idx += 1
#
#         # 递归终止条件
#         if idx >= 34:
#             return laizi % 3 == 0
#
#         # --- 分支 1: 组成刻子 (AAA) ---
#         # 只要赖子够，就可以尝试做刻子
#         if hand[idx] + laizi >= 3:
#             needed = max(0, 3 - hand[idx])
#             # 扣除资源
#             orig_count = hand[idx]
#             hand[idx] = max(0, hand[idx] - 3)
#
#             if FengHuaRules._check_melds(hand, laizi - needed, idx):
#                 # 恢复资源并返回成功
#                 hand[idx] = orig_count
#                 return True
#             # 回溯
#             hand[idx] = orig_count
#
#         # --- 分支 2: 组成顺子 (ABC) ---
#         # 只有序数牌 (0-26) 且不是 8,9 (防止越界) 可以做顺子开头
#         # 字牌 (27-33) 绝对不能做顺子
#         if idx < 27 and (idx % 9) <= 6:
#             # hand[idx] 必须消耗掉 1 个 (因为是按顺序处理的，不做刻子就得做顺子)
#             c2 = 1 if hand[idx + 1] > 0 else 0
#             c3 = 1 if hand[idx + 2] > 0 else 0
#             needs_for_seq = (1 - c2) + (1 - c3)
#
#             if laizi >= needs_for_seq:
#                 hand[idx] -= 1
#                 if c2: hand[idx + 1] -= 1
#                 if c3: hand[idx + 2] -= 1
#
#                 if FengHuaRules._check_melds(hand, laizi - needs_for_seq, idx):
#                     # 恢复
#                     hand[idx] += 1
#                     if c2: hand[idx + 1] += 1
#                     if c3: hand[idx + 2] += 1
#                     return True
#
#                 # 回溯
#                 hand[idx] += 1
#                 if c2: hand[idx + 1] += 1
#                 if c3: hand[idx + 2] += 1
#
#         return False
#
#     @staticmethod
#     def get_valid_actions(hand_counts, melds, game_state_info, extra_laizi_cnt=0):
#         mask = np.zeros(Cfg.ACTION_DIM, dtype=np.float32)
#
#         if game_state_info['phase'] == 'DISCARD':
#             for i in range(34):
#                 if hand_counts[i] > 0: mask[i] = 1.0
#
#             # 暗杠 (自己有4张)
#             for i in range(34):
#                 if hand_counts[i] == 4: mask[Cfg.ACT_GANG] = 1.0
#
#             # 补杠 (碰过的牌 + 手里有1张)
#             for m_type, m_tile in melds:
#                 if m_type == 'PON' and hand_counts[m_tile] > 0: mask[Cfg.ACT_GANG] = 1.0
#
#             is_win, _ = FengHuaRules.is_winning(hand_counts, game_state_info['laizi_set'], extra_laizi_cnt)
#             if is_win: mask[Cfg.ACT_HU] = 1.0
#
#         elif game_state_info['phase'] == 'RESPONSE':
#             tile_in = game_state_info['incoming_tile']
#             mask[Cfg.ACT_PASS] = 1.0
#             is_laizi_discard = (tile_in in game_state_info['laizi_set'])
#
#             if not is_laizi_discard:
#                 if hand_counts[tile_in] >= 2: mask[Cfg.ACT_PON] = 1.0
#                 if hand_counts[tile_in] == 3: mask[Cfg.ACT_GANG] = 1.0  # 直杠/明杠
#
#                 # 吃牌逻辑
#                 if game_state_info['is_left_player']:
#                     if tile_in < 27:
#                         rank = tile_in % 9
#                         if rank <= 6 and hand_counts[tile_in + 1] > 0 and hand_counts[tile_in + 2] > 0:
#                             mask[Cfg.ACT_CHI_LEFT] = 1.0
#                         if 1 <= rank <= 7 and hand_counts[tile_in - 1] > 0 and hand_counts[tile_in + 1] > 0:
#                             mask[Cfg.ACT_CHI_MID] = 1.0
#                         if rank >= 2 and hand_counts[tile_in - 2] > 0 and hand_counts[tile_in - 1] > 0:
#                             mask[Cfg.ACT_CHI_RIGHT] = 1.0
#
#             # 点炮胡
#             temp_hand = hand_counts.copy()
#             temp_hand[tile_in] += 1
#             is_win, _ = FengHuaRules.is_winning(temp_hand, game_state_info['laizi_set'], extra_laizi_cnt)
#             if is_win: mask[Cfg.ACT_HU] = 1.0
#
#         return mask
#
#
# class MahjongEnv:
#     def __init__(self, config=None):
#         self.cfg = config if config else Cfg()
#         self.rules = FengHuaRules()
#         self.reset()
#
#     def reset(self):
#         self.wall = []
#         for i in range(self.cfg.NUM_SUIT_TILES): self.wall.extend([i] * 4)
#         for i in range(self.cfg.ID_FLOWER_START, self.cfg.ID_FLOWER_START + self.cfg.NUM_FLOWERS): self.wall.append(i)
#         random.shuffle(self.wall)
#
#         self.dice = [random.randint(1, 6), random.randint(1, 6)]
#         dice_sum = sum(self.dice)
#         indicator_idx = -1 * (dice_sum % len(self.wall))
#         if indicator_idx == 0: indicator_idx = -1
#         self.indicator_tile = self.wall.pop(indicator_idx)
#         self.laizi_set = self.rules.calculate_laizi_set(self.indicator_tile)
#
#         self.players = []
#         for _ in range(4):
#             self.players.append({
#                 'hand': np.zeros(34, dtype=np.int32),
#                 'flower_laizis': 0,
#                 'melds': [],
#                 'flowers': [],
#                 'score': 0
#             })
#
#         self.action_history = []
#         self.response_queue = []  # 存储 (pid, priority)
#         self.dealer = 0
#
#         for pid in range(4): self._deal_initial_hand(pid, 13)
#         self._player_draw(self.dealer)
#
#         self.current_player = self.dealer
#         self.phase = 'DISCARD'
#         self.last_discard = None
#         self.last_discard_pid = None
#         self.incoming_tile = None
#         self.gang_flag = False
#         self.first_turn = True
#         self.global_step_counter = 0
#
#         return self.get_observation(self.current_player)
#
#     def _record_snapshot(self, pid, action):
#         snapshot = []
#         for p in self.players:
#             snapshot.append({
#                 'hand': p['hand'].copy(),
#                 'melds': p['melds'].copy(),
#                 'flower_laizis': p['flower_laizis'],
#                 'flowers': p['flowers'].copy()
#             })
#         self.action_history.append({'pid': pid, 'action': action, 'snapshot': snapshot})
#
#     def _player_draw(self, pid):
#         if len(self.wall) == 0: return False
#         tile = self.wall.pop(0)
#
#         if tile >= 34:
#             if tile in self.laizi_set:
#                 self.players[pid]['flower_laizis'] += 1
#                 self._record_snapshot(pid, self.cfg.ACT_DRAW)
#                 return True
#             else:
#                 self.players[pid]['flowers'].append(tile)
#                 return self._player_draw(pid)
#
#         self.players[pid]['hand'][tile] += 1
#         self._record_snapshot(pid, self.cfg.ACT_DRAW)
#         return True
#
#     def _deal_initial_hand(self, pid, count):
#         while count > 0:
#             tile = self.wall.pop(0)
#             if tile >= 34:
#                 if tile in self.laizi_set:
#                     self.players[pid]['flower_laizis'] += 1
#                     count -= 1
#                 else:
#                     self.players[pid]['flowers'].append(tile)
#             else:
#                 self.players[pid]['hand'][tile] += 1
#                 count -= 1
#
#     def _perform_gang(self, pid, incoming_tile=None):
#         """
#         [修复] 杠牌逻辑优化
#         能够自动识别:
#         1. 直杠 (别人打出的 incoming_tile)
#         2. 暗杠 (自己手里有4张)
#         3. 补杠 (碰过 + 手里有1张)
#         """
#         player = self.players[pid]
#         gang_tile = -1
#
#         # 情况 A: 响应阶段 (直杠/明杠)
#         if incoming_tile is not None and self.phase == 'RESPONSE':
#             if player['hand'][incoming_tile] == 3:
#                 player['hand'][incoming_tile] -= 3
#                 player['melds'].append(('GANG', incoming_tile))
#                 gang_tile = incoming_tile
#
#         # 情况 B: 出牌阶段 (暗杠 或 补杠)
#         else:
#             # 优先检查暗杠 (通常分高)
#             for t in range(34):
#                 if player['hand'][t] == 4:
#                     gang_tile = t
#                     player['hand'][t] -= 4
#                     player['melds'].append(('GANG', t))
#                     break
#
#             # 如果没暗杠，检查补杠
#             if gang_tile == -1:
#                 for idx, (m_type, m_tile) in enumerate(player['melds']):
#                     if m_type == 'PON' and player['hand'][m_tile] >= 1:  # 实际上只能是1
#                         gang_tile = m_tile
#                         player['hand'][m_tile] -= 1
#                         player['melds'][idx] = ('GANG', m_tile)  # 升级为杠
#                         break
#
#         if gang_tile != -1:
#             self._record_snapshot(pid, self.cfg.ACT_GANG)
#             self.gang_flag = True
#             return self._player_draw(pid)
#         else:
#             # 理论上 UI/Mask 保证了不会进这里，但为了保险
#             return True
#
#     def calculate_final_score(self, pid, win_type, context=None):
#         score = self.cfg.R_WIN_BASE
#         player = self.players[pid]
#         multiplier = 1.0
#
#         total_laizi = player['flower_laizis']
#         hand_pure = player['hand']
#         for l_id in self.laizi_set:
#             if l_id < 34: total_laizi += hand_pure[l_id]
#         if total_laizi >= 3: multiplier += 2.0
#
#         if context and context.get('is_gang_kai'): multiplier += 1.0
#
#         return score * multiplier
#
#     def get_observation(self, pid):
#         state_info = {
#             'phase': self.phase,
#             'laizi_set': self.laizi_set,
#             'incoming_tile': self.incoming_tile,
#             'is_left_player': False
#         }
#         if self.phase == 'RESPONSE' and self.last_discard_pid is not None:
#             if (self.last_discard_pid + 1) % 4 == pid:
#                 state_info['is_left_player'] = True
#
#         valid_mask = self.rules.get_valid_actions(
#             self.players[pid]['hand'], self.players[pid]['melds'], state_info,
#             extra_laizi_cnt=self.players[pid]['flower_laizis']
#         )
#
#         seq_len = self.cfg.HISTORY_LEN
#         raw_hist = self.action_history[-seq_len:]
#         formatted_hist = []
#         for item in raw_hist:
#             act_id = item['action']
#             if act_id == self.cfg.ACT_DRAW: continue
#             rel_p = (item['pid'] - pid + 4) % 4
#             formatted_hist.append((rel_p, act_id))
#         while len(formatted_hist) < seq_len:
#             formatted_hist.insert(0, (-1, -1))
#
#         obs = {
#             "hand": self.players[pid]['hand'].copy(),
#             "melds": self.players[pid]['melds'],
#             "mask": valid_mask,
#             "laizi_set": self.laizi_set,
#             "flower_laizis": self.players[pid]['flower_laizis'],
#             "flowers": self.players[pid]['flowers'],
#             "history": formatted_hist
#         }
#         return obs
#
#     def step(self, action):
#         pid = self.current_player
#         player = self.players[pid]
#         reward = 0
#         done = False
#         info = {}
#
#         self.global_step_counter += 1
#
#         # 快攻惩罚
#         STEP_THRESHOLD = 48
#         if self.global_step_counter > STEP_THRESHOLD:
#             diff = self.global_step_counter - STEP_THRESHOLD
#             delay_penalty = 0.005 * (diff ** 2)
#             reward -= delay_penalty
#
#         # 动作检查
#         state_info = {
#             'phase': self.phase,
#             'laizi_set': self.laizi_set,
#             'incoming_tile': self.incoming_tile,
#             'is_left_player': (self.last_discard_pid + 1) % 4 == pid if self.phase == 'RESPONSE' else False
#         }
#         mask = self.rules.get_valid_actions(
#             player['hand'], player['melds'], state_info, extra_laizi_cnt=player['flower_laizis']
#         )
#         if mask[action] == 0:
#             return self.get_observation(pid), -50.0, True, {"error": "Invalid Action"}
#
#         if action != self.cfg.ACT_GANG:
#             self._record_snapshot(pid, action)
#
#         # 赖子惩罚
#         if self.phase == 'DISCARD' and action <= 33:
#             if action in self.laizi_set:
#                 reward -= 100.0
#
#         # --- 状态机流转 ---
#         if self.phase == 'DISCARD':
#             if action != self.cfg.ACT_GANG: self.gang_flag = False
#
#             if action <= 33:  # 打牌
#                 player['hand'][action] -= 1
#                 self.last_discard = action
#                 self.last_discard_pid = pid
#                 self.incoming_tile = action
#
#                 # -----------------------------------------------------
#                 # [修复] 优先级判断逻辑: 胡 > 碰/杠 > 吃
#                 # 且实现 "截胡" (距离点炮者最近的人胡)
#                 # -----------------------------------------------------
#                 self.phase = 'RESPONSE'
#                 possible_responses = []
#
#                 # 1. 扫描所有人的可用动作
#                 for offset in [1, 2, 3]:
#                     target_pid = (pid + offset) % 4
#                     t_state = {
#                         'phase': 'RESPONSE',
#                         'laizi_set': self.laizi_set,
#                         'incoming_tile': self.incoming_tile,
#                         'is_left_player': (offset == 1)  # 只有下家能吃
#                     }
#                     t_mask = self.rules.get_valid_actions(
#                         self.players[target_pid]['hand'], self.players[target_pid]['melds'], t_state,
#                         extra_laizi_cnt=self.players[target_pid]['flower_laizis']
#                     )
#
#                     # 分析该玩家能做什么
#                     can_hu = (t_mask[Cfg.ACT_HU] == 1.0)
#                     can_pon_gang = (t_mask[Cfg.ACT_PON] == 1.0 or t_mask[Cfg.ACT_GANG] == 1.0)
#                     can_chi = (t_mask[Cfg.ACT_CHI_LEFT] == 1.0 or t_mask[Cfg.ACT_CHI_MID] == 1.0 or t_mask[
#                         Cfg.ACT_CHI_RIGHT] == 1.0)
#
#                     if can_hu or can_pon_gang or can_chi:
#                         possible_responses.append({
#                             'pid': target_pid,
#                             'offset': offset,  # 距离 1=下家, 2=对家, 3=上家
#                             'hu': can_hu,
#                             'pon': can_pon_gang,
#                             'chi': can_chi
#                         })
#
#                 self.response_queue = []
#
#                 # 2. 判定优先级
#                 # A. 检查胡 (截胡规则: offset最小的人优先)
#                 hu_players = [p for p in possible_responses if p['hu']]
#                 if hu_players:
#                     hu_players.sort(key=lambda x: x['offset'])  # 按距离排序
#                     winner = hu_players[0]
#                     self.response_queue.append(winner['pid'])
#                     # 注意: 这里只放入了最高优先级的胡牌者
#                     # 如果他选择 PASS，理论上应该问下一个，但为简化 Env，若第一胡Pass，则视为都没胡
#
#                 # B. 如果没人胡，检查 碰/杠
#                 else:
#                     pon_players = [p for p in possible_responses if p['pon']]
#                     if pon_players:
#                         # 碰/杠 也是谁先碰到谁算 (其实只有一个能碰，因为牌一共4张，出了1张，最多手里2张，不可能两人同时碰)
#                         self.response_queue.append(pon_players[0]['pid'])
#
#                     # C. 如果没碰杠，检查 吃
#                     else:
#                         chi_players = [p for p in possible_responses if p['chi']]
#                         if chi_players:
#                             # 只有下家(offset=1)能吃，逻辑上已经在 mask 里限制了
#                             self.response_queue.append(chi_players[0]['pid'])
#
#                 # 3. 如果队列为空，直接流转到下家摸牌
#                 if not self.response_queue:
#                     self.phase = 'DISCARD'
#                     next_pid = (pid + 1) % 4
#                     self.current_player = next_pid
#                     if not self._player_draw(next_pid):
#                         done = True
#                         info['winner'] = None
#                 else:
#                     # 将控制权交给队列第一个人
#                     self.current_player = self.response_queue[0]
#
#             elif action == self.cfg.ACT_HU:  # 自摸
#                 is_win, w_type = self.rules.is_winning(player['hand'], self.laizi_set,
#                                                        extra_laizi_cnt=player['flower_laizis'])
#                 if not is_win: return self.get_observation(pid), -50, True, {"error": "False HU"}
#
#                 final_score = self.calculate_final_score(pid, w_type)
#                 reward += final_score * 20
#                 done = True
#                 info['winner'] = pid
#                 info['win_type'] = 'zimo'
#
#             elif action == self.cfg.ACT_GANG:  # 暗杠/补杠
#                 if not self._perform_gang(pid):
#                     done = True
#                     info['winner'] = None
#
#         elif self.phase == 'RESPONSE':
#             self.gang_flag = False
#
#             if action == self.cfg.ACT_PASS:
#                 # 既然我们只放了最高优先级的人进队列，如果他 PASS
#                 # 简单起见，我们认为该动作作废，直接进入下家摸牌
#                 # (不做复杂的 "玩家A不胡 -> 问玩家B胡不胡" 逻辑，这符合大部分 RL Env 的简化处理)
#                 if self.response_queue: self.response_queue.pop(0)
#
#                 if self.response_queue:
#                     self.current_player = self.response_queue[0]
#                 else:
#                     self.phase = 'DISCARD'
#                     next_pid = (self.last_discard_pid + 1) % 4
#                     self.current_player = next_pid
#                     if not self._player_draw(next_pid):
#                         done = True
#                         info['winner'] = None
#
#             elif action == self.cfg.ACT_PON:
#                 player['hand'][self.incoming_tile] -= 2
#                 player['melds'].append(('PON', self.incoming_tile))
#                 self.phase = 'DISCARD'
#                 self.response_queue = []
#
#             elif action == self.cfg.ACT_HU:  # 点炮
#                 player['hand'][self.incoming_tile] += 1
#                 is_win, w_type = self.rules.is_winning(player['hand'], self.laizi_set,
#                                                        extra_laizi_cnt=player['flower_laizis'])
#                 if not is_win: return self.get_observation(pid), -50, True, {"error": "False HU"}
#
#                 final_score = self.calculate_final_score(pid, w_type)
#                 reward += final_score * 20
#                 done = True
#                 info['winner'] = pid
#                 info['win_type'] = 'ron'
#                 info['loser'] = self.last_discard_pid
#
#             elif action == self.cfg.ACT_GANG:  # 明杠/直杠
#                 if not self._perform_gang(pid, incoming_tile=self.incoming_tile):
#                     done = True
#                     info['winner'] = None
#                 else:
#                     # 杠完摸牌，phase 变回 discard，由该玩家继续
#                     self.phase = 'DISCARD'
#                     self.response_queue = []
#
#             elif action in [self.cfg.ACT_CHI_LEFT, self.cfg.ACT_CHI_MID, self.cfg.ACT_CHI_RIGHT]:
#                 t = self.incoming_tile
#                 if action == self.cfg.ACT_CHI_LEFT:
#                     player['hand'][t + 1] -= 1;
#                     player['hand'][t + 2] -= 1
#                     player['melds'].append(('CHI_L', t))
#                 elif action == self.cfg.ACT_CHI_MID:
#                     player['hand'][t - 1] -= 1;
#                     player['hand'][t + 1] -= 1
#                     player['melds'].append(('CHI_M', t))
#                 elif action == self.cfg.ACT_CHI_RIGHT:
#                     player['hand'][t - 2] -= 1;
#                     player['hand'][t - 1] -= 1
#                     player['melds'].append(('CHI_R', t))
#
#                 self.phase = 'DISCARD'
#                 self.response_queue = []
#
#         if not done:
#             hand_laizi_count = player['flower_laizis']
#             for l_id in self.laizi_set:
#                 if l_id < 34:
#                     hand_laizi_count += player['hand'][l_id]
#
#             if hand_laizi_count > 0:
#                 reward += (hand_laizi_count * 3)
#
#             # 孤张字牌惩罚
#             for h_id in range(27, 34):
#                 if player['hand'][h_id] == 1 and h_id not in self.laizi_set:
#                     reward -= 2
#
#         return self.get_observation(self.current_player), reward, done, info
#
#
