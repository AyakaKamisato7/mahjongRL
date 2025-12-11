class MahjongConfig:
    """
    奉化麻将全局配置类
    包含：游戏常量、动作空间定义、奖励函数参数
    """

    # --- 基础定义 ---
    NUM_TILES_TOTAL = 144
    NUM_SUIT_TILES = 34  # 序数牌+字牌种类 (0-33)
    NUM_FLOWERS = 8  # 花牌数量 (34-41)
    HAND_SIZE = 13  # 标准手牌数 (胡牌时14)

    # --- 动作空间映射 (Action Space: 41) ---
    # 0-33: 打出对应的牌 (Discard)
    ACT_DISCARD_START = 0
    ACT_DISCARD_END = 33

    # 特殊动作 (34-40)
    ACT_PASS = 34  # 过 (放弃吃碰杠胡)
    ACT_HU = 35  # 胡 (自摸或点炮)
    ACT_PON = 36  # 碰 (Pon)
    ACT_GANG = 37  # 杠 (含明杠、暗杠、补杠，逻辑层自动区分)
    ACT_CHI_LEFT = 38  # 左吃 (如手里56吃4)
    ACT_CHI_MID = 39  # 中吃 (如手里35吃4)
    ACT_CHI_RIGHT = 40  # 右吃 (如手里23吃4)

    ACTION_DIM = 41

    # --- 牌 ID 映射 (方便理解) ---
    ID_MAN_START = 0  # 万 (0-8)
    ID_PIN_START = 9  # 筒 (9-17)
    ID_SOU_START = 18  # 索 (18-26)
    ID_WIND_START = 27  # 东南西北 (27-30)
    ID_DRAGON_START = 31  # 中发白 (31-33)
    ID_FLOWER_START = 34  # 花牌开始

    # ==============================
    # 2. 奖励系统 (Reward Shaping)
    # ==============================
    R_WIN_BASE = 1.0  # 基础胡牌
    R_LOSE_BASE = -1.0  # 基础点炮/被自摸
    R_INVALID = -10.0  # 非法动作兜底

    # 引导性奖励 (根据你的代码保留)
    R_CHIPON = 0.1  # 鼓励吃碰
    R_GANG = 0.5  # 鼓励杠
    # R_STEP = -0.01  # 鼓励快攻

    # --- 台数倍率 (Multipliers) ---
    # 规则确认：
    # 1. 花牌仅整套加分 (TAI_FLOWER_SET)
    # 2. 赖子仅三张齐加分 (MULT_LAIZI_ALL)

    # 特殊大牌
    MULT_THIRTEEN_LAN = 0.2  # 十三烂
    MULT_LUAN_FENG = 0.2  # 乱风
    MULT_QING_FENG = 8.0  # 清风
    MULT_TIAN_HU = 10.0  # 天胡
    MULT_DI_HU = 5.0  # 地胡
    MULT_LAIZI_ALL = 3.0  # 三赖子集齐

    # 常见规则
    TAI_PENG_PENG = 2.0  # 对对胡
    TAI_QING_YI_SE = 6.0  # 清一色
    TAI_HUN_YI_SE = 3.0  # 混一色
    TAI_FLOWER_SET = 2.0  # 一套花 (春夏秋冬 OR 梅兰竹菊)
    TAI_GANG_KAI = 2.0  # 杠上开花
    TAI_HAI_DI = 2.0  # 海底捞月
    TAI_QUAN_QIU = 2.0  # 全求人
    TAI_QIANG_GANG = 2.0  # 抢杠胡

    # ==============================
    # 3. 训练超参数 (Training Hyperparams)
    # ==============================
    # 基础设置
    SEED = 42
    MAX_TIMESTEPS = 1_000_000  # 总训练步数
    CHECKPOINT_FREQ = 100  # 每多少次更新保存一次模型

    # PPO 算法核心参数
    BATCH_SIZE = 4096  # 收集多少步数据后进行一次更新 (建议大一些，因为麻将方差大)
    MINIBATCH_SIZE = 128  # 显卡每次计算的小批次
    EPOCHS = 10  # 每次收集的数据重复训练几轮

    # 学习率与优化器
    LR_START = 3e-4  # 初始学习率 (3e-4 是 PPO 的黄金点)
    LR_MIN = 1e-6  # 余弦退火的最低点
    MAX_GRAD_NORM = 0.5  # 梯度裁剪 (防止梯度爆炸)

    # 强化学习机制
    GAMMA = 0.99  # 折扣因子 (0.99 关注长期，麻将一局很长，建议0.99)
    GAE_LAMBDA = 0.95  # GAE 平滑系数
    EPS_CLIP = 0.2  # PPO 截断范围 (限制策略突变)
    ENTROPY_COEF = 0.01  # 熵系数 (替代 Epsilon-Greedy，鼓励探索)
    VF_COEF = 0.5  # 价值函数损失权重

    # 早停与评估
    USE_EARLY_STOPPING = True
    PATIENCE = 50  # 容忍多少轮 Reward 不上升
    MIN_DELTA = 0.001  # 提升阈值

    # ==============================
    # 4. 神经网络架构参数 (Network Arch)
    # ==============================
    # 特征提取
    CNN_CHANNEL_IN = 5  # 输入通道数 (手牌, 赖子, 碰杠, 弃牌, 花牌)
    CNN_CHANNEL_OUT = 64  # CNN 输出特征图深度

    # LSTM 时序参数
    USE_LSTM = True
    HISTORY_LEN = 30  # LSTM 回看过去多少步动作 (Time Steps)
    LSTM_INPUT_DIM = 41 + 4  # 动作OneHot(41) + 玩家相对位置OneHot(4)
    LSTM_HIDDEN_DIM = 128  # LSTM 隐藏层维度
    LSTM_LAYERS = 2

    # 全连接层
    FC_HIDDEN_DIM = 512  # 融合后的全连接层宽度

    # 自博弈参数
    OPPONENT_POOL_SIZE = 10  # 保存多少个旧模型用于陪练
    UPDATE_OPPONENT_FREQ = 500  # 每训练多少局更新一次对手池


    # # ==============================
    # # 3. 训练超参数 (Training Hyperparams)
    # # ==============================
    # # 基础训练设置
    # SEED = 42
    # MAX_TIMESTEPS = 1_000_000  # 总训练步数
    # BATCH_SIZE = 2048  # PPO 每次更新的数据量
    # MINIBATCH_SIZE = 64  # 小批次更新
    # EPOCHS = 10  # 每次收集数据后更新网络的次数
    #
    # # PPO 特定参数
    # GAMMA = 0.99  # 折扣因子 (关注长期利益)
    # GAE_LAMBDA = 0.95  # 优势函数平滑系数
    # EPS_CLIP = 0.2  # PPO 截断范围 (0.1 - 0.2)
    # ENTROPY_COEF = 0.01  # 熵系数 (替代 Epsilon-Greedy，控制探索)
    # VF_COEF = 0.5  # 价值损失权重
    # MAX_GRAD_NORM = 0.5  # 梯度裁剪防爆炸
    #
    # # --- 学习率调度 (Cosine Annealing) ---
    # LR_START = 3e-4  # 初始学习率
    # LR_MIN = 1e-6  # 最低学习率
    # T_MAX = MAX_TIMESTEPS  # 周期 (通常设为总步数)
    #
    # # --- 早停机制 (Early Stopping) ---
    # USE_EARLY_STOPPING = True
    # PATIENCE = 50  # 如果 50 次评估没有提升则停止
    # MIN_DELTA = 0.001  # 提升阈值