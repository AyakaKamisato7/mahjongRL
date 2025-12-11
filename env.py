import numpy as np
import random
from config import MahjongConfig as Cfg


class TileUtils:
    """
    [工具类] 负责牌的ID转换、打印
    仅用于日志输出和调试，不涉及核心逻辑。
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
        # 花牌显示逻辑
        flowers = ["春", "夏", "秋", "冬", "梅", "兰", "竹", "菊"]
        if 34 <= tile_id <= 41: return flowers[tile_id - 34]
        return "未知"

    @staticmethod
    def get_tile_type(tile_id):
        """返回牌的大类：0=万, 1=筒, 2=索, 3=字, 4=花"""
        if 0 <= tile_id <= 8: return 0
        if 9 <= tile_id <= 17: return 1
        if 18 <= tile_id <= 26: return 2
        if 27 <= tile_id <= 33: return 3
        return 4


class FengHuaRules:
    """
    [核心逻辑层] 奉化麻将数学公理 (The Axioms)
    包含：赖子集合计算、胡牌算法(十三烂/清风/乱风/标准)、合法动作掩码生成。
    此类是无状态的(Stateless)，所有状态数据由 Env 传入。
    """

    @staticmethod
    def calculate_laizi_set(indicator_id):
        """
        [逻辑] 根据指示牌返回赖子 ID 集合 (Set)
        """
        laizi_set = set()

        # --- 情况 A: 翻出花牌 (34-41) ---
        if indicator_id >= 34:
            # 春夏秋冬 (34-37)
            if 34 <= indicator_id <= 37:
                group = {34, 35, 36, 37}
                group.remove(indicator_id)
                return group
            # 梅兰竹菊 (38-41)
            elif 38 <= indicator_id <= 41:
                group = {38, 39, 40, 41}
                group.remove(indicator_id)
                return group

        # --- 情况 B: 普通牌 (0-33) ---
        type_idx = TileUtils.get_tile_type(indicator_id)
        target = -1

        # 万筒索 (9变1)
        if type_idx < 3:
            base = (indicator_id // 9) * 9
            rank = indicator_id % 9
            target = base + ((rank + 1) % 9)
        # 风牌 (北变东)
        elif 27 <= indicator_id <= 30:
            target = 27 + ((indicator_id - 27 + 1) % 4)
        # 箭牌 (白变中)
        else:
            target = 31 + ((indicator_id - 31 + 1) % 3)

        laizi_set.add(target)
        return laizi_set

    @staticmethod
    def is_winning(hand_counts, laizi_set, extra_laizi_cnt=0, check_special=True):
        """
        [胡牌判定主入口]
        目前的配置：【屏蔽】乱风和十三烂，强迫 Agent 学习 Standard。
        """
        # 1. 计算总赖子数 & 提取纯手牌
        # 初始赖子数为手里的花牌赖子
        total_laizi_cnt = extra_laizi_cnt

        hand_pure = hand_counts.copy()

        # 遍历赖子集合，如果赖子是普通牌(0-33)，从手牌里扣除并加到计数器
        for l_id in laizi_set:
            if l_id < 34:
                count = hand_pure[l_id]
                if count > 0:
                    total_laizi_cnt += count
                    hand_pure[l_id] = 0  # 赖子离手，变身万能牌

        # --- A. 检查【全字牌】(乱风 / 清风) ---
        if check_special:
            is_all_winds = True
            for i in range(27):  # 扫描万筒索
                if hand_pure[i] > 0:
                    is_all_winds = False
                    break

            if is_all_winds:
                # 已经是全字牌了，进一步区分结构
                # 如果能组成标准胡牌公式 -> 清风 (8倍)
                if FengHuaRules._check_standard_win(hand_pure, total_laizi_cnt):
                    return True, "QingFeng"
                else:
                    # [修改] 暂时注释掉乱风，防止 Agent 走捷径
                    # return True, "LuanFeng"
                    pass

        # --- B. 检查【十三烂】(Thirteen Unrelated) ---
        if check_special:
            # [修改] 暂时注释掉十三烂
            # if FengHuaRules._check_thirteen_lan(hand_pure, total_laizi_cnt):
            #    return True, "ThirteenLan"
            pass

        # --- C. 检查【标准胡牌】(Standard) ---
        if FengHuaRules._check_standard_win(hand_pure, total_laizi_cnt):
            return True, "Standard"

        return False, None

    @staticmethod
    def _check_thirteen_lan(hand_pure, laizi_cnt):
        """
        [算法] 十三烂判定
        """
        # 1. 绝对禁止硬牌对子
        if np.any(hand_pure > 1):
            return False

        # 2. 检查序数牌间隔
        tiles = [t for t in range(34) if hand_pure[t] > 0]
        suits = [[], [], []]  # Man, Pin, Sou
        for t in tiles:
            if t < 27:
                suits[t // 9].append(t % 9 + 1)  # 存数值 1-9

        for suit_tiles in suits:
            if len(suit_tiles) < 2: continue
            suit_tiles.sort()
            for k in range(len(suit_tiles) - 1):
                if suit_tiles[k + 1] - suit_tiles[k] < 3:
                    return False

        # 3. 只要硬牌满足上述条件，剩下的空位由赖子填充即可
        return True

    @staticmethod
    def _check_standard_win(hand_pure, laizi_cnt):
        """
        [算法] 标准胡牌递归回溯 (DFS)
        目标：4面子 + 1将
        """
        # 剪枝：张数校验 (3n + 2)
        current_tiles = sum(hand_pure) + laizi_cnt
        if current_tiles % 3 != 2:
            return False

        # 1. 尝试用两张硬牌做将
        for t in range(34):
            if hand_pure[t] >= 2:
                hand_pure[t] -= 2
                if FengHuaRules._check_melds(hand_pure, laizi_cnt, 0):
                    hand_pure[t] += 2
                    return True
                hand_pure[t] += 2

        # 2. 尝试用一张硬牌 + 一张赖子做将
        if laizi_cnt >= 1:
            for t in range(34):
                if hand_pure[t] >= 1:
                    hand_pure[t] -= 1
                    if FengHuaRules._check_melds(hand_pure, laizi_cnt - 1, 0):
                        hand_pure[t] += 1
                        return True
                    hand_pure[t] += 1

        # 3. 尝试用两张赖子做将
        if laizi_cnt >= 2:
            if FengHuaRules._check_melds(hand_pure, laizi_cnt - 2, 0):
                return True

        return False

    @staticmethod
    def _check_melds(hand, laizi, idx):
        """
        [算法] 递归消消乐：检查剩余牌能否组成面子
        idx: 当前遍历到的牌ID (0-33)
        """
        # 终止条件：所有牌ID遍历完毕
        if idx >= 34:
            return laizi % 3 == 0  # 剩余赖子必须能组成刻子 (AAA)

        # 跳过空牌
        if hand[idx] == 0:
            return FengHuaRules._check_melds(hand, laizi, idx + 1)

        # --- 策略 A: 组成刻子 (AAA) ---
        # 1. 自己有3张
        if hand[idx] >= 3:
            hand[idx] -= 3
            if FengHuaRules._check_melds(hand, laizi, idx):  # 递归同一idx
                hand[idx] += 3
                return True
            hand[idx] += 3

        # 2. 赖子补刻子 (我有1张补2赖，我有2张补1赖)
        if hand[idx] < 3 and laizi >= (3 - hand[idx]):
            needed = 3 - hand[idx]
            orig = hand[idx]
            hand[idx] = 0  # 消耗光当前牌
            if FengHuaRules._check_melds(hand, laizi - needed, idx + 1):
                hand[idx] = orig
                return True
            hand[idx] = orig

        # --- 策略 B: 组成顺子 (ABC) ---
        # 仅限序数牌，且不能是8或9开头
        if idx < 27 and (idx % 9) <= 6:
            # 1. 纯硬牌顺子 (X, X+1, X+2)
            if hand[idx + 1] > 0 and hand[idx + 2] > 0:
                hand[idx] -= 1
                hand[idx + 1] -= 1
                hand[idx + 2] -= 1
                if FengHuaRules._check_melds(hand, laizi, idx):  # 再次检查当前位置
                    hand[idx] += 1
                    hand[idx + 1] += 1
                    hand[idx + 2] += 1
                    return True
                hand[idx] += 1
                hand[idx + 1] += 1
                hand[idx + 2] += 1

            # 2. 赖子补顺子 (简化版: 不做深度搜索)

        return False

    @staticmethod
    def get_valid_actions(hand_counts, melds, game_state_info, extra_laizi_cnt=0):
        """
        [Mask生成] 获取当前状态下的合法动作掩码
        :param extra_laizi_cnt: 传入花牌赖子数量，用于判断胡牌
        """
        mask = np.zeros(Cfg.ACTION_DIM, dtype=np.float32)

        # --- 阶段 1: 必须出牌 (DISCARD) ---
        if game_state_info['phase'] == 'DISCARD':
            # 1. 出牌
            for i in range(34):
                if hand_counts[i] > 0:
                    mask[i] = 1.0

            # 2. 暗杠
            for i in range(34):
                if hand_counts[i] == 4:
                    mask[Cfg.ACT_GANG] = 1.0

            # 3. 补杠
            for m_type, m_tile in melds:
                if m_type == 'PON' and hand_counts[m_tile] > 0:
                    mask[Cfg.ACT_GANG] = 1.0

            # 4. 自摸胡
            is_win, _ = FengHuaRules.is_winning(
                hand_counts,
                game_state_info['laizi_set'],
                extra_laizi_cnt
            )
            if is_win:
                mask[Cfg.ACT_HU] = 1.0

        # --- 阶段 2: 响应别人打牌 (RESPONSE) ---
        elif game_state_info['phase'] == 'RESPONSE':
            tile_in = game_state_info['incoming_tile']
            mask[Cfg.ACT_PASS] = 1.0  # 永远可以选择“过”

            # 1. 碰
            if hand_counts[tile_in] >= 2:
                mask[Cfg.ACT_PON] = 1.0

            # 2. 明杠
            if hand_counts[tile_in] == 3:
                mask[Cfg.ACT_GANG] = 1.0

            # 3. 吃
            if game_state_info['is_left_player']:
                if tile_in < 27:  # 序数牌才能吃
                    rank = tile_in % 9
                    # 左吃 (56 吃 4) -> 需有 5, 6
                    if rank <= 6:
                        if hand_counts[tile_in + 1] > 0 and hand_counts[tile_in + 2] > 0:
                            mask[Cfg.ACT_CHI_LEFT] = 1.0
                    # 中吃 (35 吃 4) -> 需有 3, 5
                    if 1 <= rank <= 7:
                        if hand_counts[tile_in - 1] > 0 and hand_counts[tile_in + 1] > 0:
                            mask[Cfg.ACT_CHI_MID] = 1.0
                    # 右吃 (23 吃 4) -> 需有 2, 3
                    if rank >= 2:
                        if hand_counts[tile_in - 2] > 0 and hand_counts[tile_in - 1] > 0:
                            mask[Cfg.ACT_CHI_RIGHT] = 1.0

            # 4. 点炮胡
            temp_hand = hand_counts.copy()
            temp_hand[tile_in] += 1
            is_win, _ = FengHuaRules.is_winning(
                temp_hand,
                game_state_info['laizi_set'],
                extra_laizi_cnt
            )
            if is_win:
                mask[Cfg.ACT_HU] = 1.0

        return mask


class MahjongEnv:
    """
    [环境层] 奉化麻将 Gym 环境
    管理：牌墙、玩家状态、回合流转、奖励计算。
    """

    def __init__(self, config=None):
        self.cfg = config if config else Cfg()
        self.rules = FengHuaRules()
        self.reset()

    def reset(self):
        """
        [初始化] 新的一局
        """
        # 1. 初始化牌墙 (144张)
        self.wall = []
        for i in range(self.cfg.NUM_SUIT_TILES):
            self.wall.extend([i] * 4)
        for i in range(self.cfg.ID_FLOWER_START, self.cfg.ID_FLOWER_START + self.cfg.NUM_FLOWERS):
            self.wall.append(i)
        random.shuffle(self.wall)

        # 2. 定赖子
        self.dice = [random.randint(1, 6), random.randint(1, 6)]
        dice_sum = sum(self.dice)
        indicator_idx = -1 * (dice_sum % len(self.wall))
        if indicator_idx == 0: indicator_idx = -1
        self.indicator_tile = self.wall.pop(indicator_idx)
        self.laizi_set = self.rules.calculate_laizi_set(self.indicator_tile)

        # 3. 初始化玩家
        self.players = []
        for _ in range(4):
            self.players.append({
                'hand': np.zeros(34, dtype=np.int32),
                'flower_laizis': 0,
                'melds': [],
                'flowers': [],
                'score': 0
            })

        # 4. 发牌
        self.dealer = 0
        for pid in range(4):
            self._deal_initial_hand(pid, 13)
        self._player_draw(self.dealer)

        # 5. 设置初始状态
        self.current_player = self.dealer
        self.phase = 'DISCARD'
        self.last_discard = None
        self.last_discard_pid = None
        self.incoming_tile = None

        # 6. 局况追踪
        self.action_history = []
        self.gang_flag = False  # 是否刚刚杠过 (用于杠上开花)
        self.first_turn = True  # 是否首轮 (用于天胡地胡)

        return self.get_observation(self.current_player)

    def _player_draw(self, pid):
        while True:
            if len(self.wall) == 0:
                return False

            tile = self.wall.pop(0)

            # --- 花牌处理逻辑 ---
            if tile >= 34:
                if tile in self.laizi_set:
                    self.players[pid]['flower_laizis'] += 1
                    return True
                else:
                    self.players[pid]['flowers'].append(tile)
                    continue

                    # --- 普通牌处理 ---
            self.players[pid]['hand'][tile] += 1
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

    def calculate_final_score(self, pid, win_type, context=None):
        """
        [计分系统] 计算最终倍率
        """
        if context is None: context = {}

        score = self.cfg.R_WIN_BASE
        player = self.players[pid]

        # --- 1. 牌型基础倍率 ---
        multiplier = 1.0
        if win_type == 'ThirteenLan':
            multiplier += self.cfg.MULT_THIRTEEN_LAN
        elif win_type == 'LuanFeng':
            multiplier += self.cfg.MULT_LUAN_FENG
        elif win_type == 'QingFeng':
            multiplier += self.cfg.MULT_QING_FENG

        # --- 2. 检查 清一色/混一色/对对胡 ---
        if win_type in ['Standard', 'QingFeng']:
            suits = set()
            has_honor = False

            # 检查手牌
            for t in range(34):
                if player['hand'][t] > 0 and t not in self.laizi_set:
                    if t >= 27:
                        has_honor = True
                    else:
                        suits.add(t // 9)

            # 检查副露
            for m_type, m_tile in player['melds']:
                if m_tile not in self.laizi_set:
                    if m_tile >= 27:
                        has_honor = True
                    else:
                        suits.add(m_tile // 9)

            # 判定
            if len(suits) == 1:
                if not has_honor:
                    multiplier += self.cfg.TAI_QING_YI_SE  # 清一色
                else:
                    multiplier += self.cfg.TAI_HUN_YI_SE  # 混一色

            # 对对胡判定 (简单判定: 碰杠数>=3)
            # 完整逻辑需要回溯，这里做近似
            pass

        # --- 3. 局况倍率 ---
        if context.get('is_hai_di', False):
            multiplier += self.cfg.TAI_HAI_DI
        if context.get('is_gang_kai', False):
            multiplier += self.cfg.TAI_GANG_KAI
        if context.get('is_tian_hu', False):
            multiplier += self.cfg.MULT_TIAN_HU
        if context.get('is_di_hu', False):
            multiplier += self.cfg.MULT_DI_HU

        # 全求人 (Melds=4)
        if len(player['melds']) == 4:
            multiplier += self.cfg.TAI_QUAN_QIU

        # --- 4. 花牌 & 赖子加成 ---
        f_counts = [0] * 42
        for f in player['flowers']: f_counts[f] = 1

        if sum(f_counts[34:38]) == 4:  # 集齐春夏秋冬
            multiplier += self.cfg.TAI_FLOWER_SET
        if sum(f_counts[38:42]) == 4:  # 集齐梅兰竹菊
            multiplier += self.cfg.TAI_FLOWER_SET

        total_laizi = player['flower_laizis']
        hand_pure = player['hand']
        for l_id in self.laizi_set:
            if l_id < 34:
                total_laizi += hand_pure[l_id]

        if total_laizi >= 3:
            multiplier += self.cfg.MULT_LAIZI_ALL

        return score * multiplier

    def get_observation(self, pid):
        """
        [Obs生成] 适配 Dictionary 格式的历史记录
        """
        # 1. 基础信息
        state_info = {
            'phase': self.phase,
            'laizi_set': self.laizi_set,
            'incoming_tile': self.incoming_tile,
            'is_left_player': False
        }

        if self.phase == 'RESPONSE' and self.last_discard_pid is not None:
            if (self.last_discard_pid + 1) % 4 == pid:
                state_info['is_left_player'] = True

        # 2. Mask
        valid_mask = self.rules.get_valid_actions(
            self.players[pid]['hand'],
            self.players[pid]['melds'],
            state_info,
            extra_laizi_cnt=self.players[pid]['flower_laizis']
        )

        # 3. History Parsing [关键修改点]
        seq_len = self.cfg.HISTORY_LEN
        raw_hist = self.action_history[-seq_len:]

        formatted_hist = []
        for item in raw_hist:
            # 现在 item 是一个字典 {'pid':..., 'action':..., 'snapshot':...}
            p_idx = item['pid']
            act_id = item['action']

            rel_p = (p_idx - pid + 4) % 4
            formatted_hist.append((rel_p, act_id))

        # Padding
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
        """
        [Step核心]
        修改点：记录 Action 之前，先拍摄全场快照 (Snapshot)
        """
        pid = self.current_player
        player = self.players[pid]
        reward = 0
        done = False
        info = {}

        # 0. 局况快照
        is_hai_di = (len(self.wall) == 0)
        is_gang_kai = self.gang_flag
        is_first_turn = (len(self.action_history) < 4)

        # 1. Mask 校验
        state_info = {
            'phase': self.phase,
            'laizi_set': self.laizi_set,
            'incoming_tile': self.incoming_tile,
            'is_left_player': (self.last_discard_pid + 1) % 4 == pid if self.phase == 'RESPONSE' else False
        }
        mask = self.rules.get_valid_actions(
            player['hand'],
            player['melds'],
            state_info,
            extra_laizi_cnt=player['flower_laizis']
        )

        if mask[action] == 0:
            return self.get_observation(pid), self.cfg.R_INVALID, True, {"error": "Invalid Action"}

        # --- [关键修改] 记录动作前的 全场手牌快照 ---
        # 我们需要深拷贝每位玩家的状态，否则引用变了历史记录也会变
        snapshot = []
        for p in self.players:
            snapshot.append({
                'hand': p['hand'].copy(),  # Numpy array copy
                'melds': p['melds'].copy(),  # List shallow copy
                'flower_laizis': p['flower_laizis'],
                'flowers': p['flowers'].copy()
            })

        # 存入字典结构
        self.action_history.append({
            'pid': pid,
            'action': action,
            'snapshot': snapshot
        })

        # 2. 执行动作
        if self.phase == 'DISCARD':
            if action != self.cfg.ACT_GANG:
                self.gang_flag = False

            if action <= 33:  # 打牌
                player['hand'][action] -= 1
                self.last_discard = action
                self.last_discard_pid = pid
                self.incoming_tile = action
                self.phase = 'RESPONSE'
                self.current_player = (pid + 1) % 4

            elif action == self.cfg.ACT_HU:  # 自摸胡
                is_win, w_type = self.rules.is_winning(
                    player['hand'],
                    self.laizi_set,
                    extra_laizi_cnt=player['flower_laizis']
                )
                ctx = {
                    'is_hai_di': is_hai_di,
                    'is_gang_kai': is_gang_kai,
                    'is_tian_hu': (is_first_turn and pid == self.dealer)
                }
                final_score = self.calculate_final_score(pid, w_type, context=ctx)
                reward = final_score
                done = True
                info['winner'] = pid

            elif action == self.cfg.ACT_GANG:  # 暗杠/补杠
                self.gang_flag = True  # 标记杠状态
                self._player_draw(pid)

        elif self.phase == 'RESPONSE':
            self.gang_flag = False

            if action == self.cfg.ACT_PASS:
                self.phase = 'DISCARD'
                self._player_draw(self.current_player)

            elif action == self.cfg.ACT_PON:
                player['hand'][self.incoming_tile] -= 2
                player['melds'].append(('PON', self.incoming_tile))
                self.phase = 'DISCARD'

            elif action == self.cfg.ACT_HU:  # 点炮胡
                player['hand'][self.incoming_tile] += 1
                is_win, w_type = self.rules.is_winning(
                    player['hand'],
                    self.laizi_set,
                    extra_laizi_cnt=player['flower_laizis']
                )
                ctx = {
                    'is_hai_di': is_hai_di,
                    'is_di_hu': (is_first_turn and pid != self.dealer)
                }
                final_score = self.calculate_final_score(pid, w_type, context=ctx)
                reward = final_score
                done = True
                info['winner'] = pid

            elif action == self.cfg.ACT_CHI_LEFT:
                t = self.incoming_tile
                player['hand'][t + 1] -= 1
                player['hand'][t + 2] -= 1
                player['melds'].append(('CHI', t))
                self.phase = 'DISCARD'

            elif action == self.cfg.ACT_CHI_MID:
                t = self.incoming_tile
                player['hand'][t - 1] -= 1
                player['hand'][t + 1] -= 1
                player['melds'].append(('CHI', t))
                self.phase = 'DISCARD'

            elif action == self.cfg.ACT_CHI_RIGHT:
                t = self.incoming_tile
                player['hand'][t - 2] -= 1
                player['hand'][t - 1] -= 1
                player['melds'].append(('CHI', t))
                self.phase = 'DISCARD'

        return self.get_observation(self.current_player), reward, done, info
