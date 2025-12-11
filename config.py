class MahjongConfig:
    """
    奉化麻将全局配置类 (v4.0 Final)
    """
    # --- 基础定义 ---
    NUM_TILES_TOTAL = 144
    NUM_SUIT_TILES = 34
    NUM_FLOWERS = 8
    HAND_SIZE = 13

    # --- 动作空间 ---
    ACT_DISCARD_START = 0
    ACT_DISCARD_END = 33
    ACT_PASS = 34
    ACT_HU = 35
    ACT_PON = 36
    ACT_GANG = 37
    ACT_CHI_LEFT = 38
    ACT_CHI_MID = 39
    ACT_CHI_RIGHT = 40

    # 辅助动作 (不输入网络)
    ACT_DRAW = -1

    ACTION_DIM = 41

    # --- 牌 ID 映射 ---
    ID_MAN_START = 0
    ID_PIN_START = 9
    ID_SOU_START = 18
    ID_WIND_START = 27
    ID_DRAGON_START = 31
    ID_FLOWER_START = 34

    # ==============================
    # 奖励系统
    # ==============================
    R_WIN_BASE = 1.0
    R_LOSE_BASE = -1.0  # 实际上我们在 env 里动态计算
    R_INVALID = -10.0

    # 稠密奖励 (可选)
    R_CHIPON = 0.1
    R_GANG = 0.5

    # --- 台数倍率 ---
    # 特殊大牌 (逻辑已屏蔽，参数保留)
    MULT_THIRTEEN_LAN = 0.2
    MULT_LUAN_FENG = 0.2
    MULT_QING_FENG = 8.0
    MULT_TIAN_HU = 10.0
    MULT_DI_HU = 5.0
    MULT_LAIZI_ALL = 3.0

    # 常见规则
    TAI_PENG_PENG = 2.0
    TAI_QING_YI_SE = 6.0
    TAI_HUN_YI_SE = 3.0
    TAI_FLOWER_SET = 2.0
    TAI_GANG_KAI = 2.0
    TAI_HAI_DI = 2.0
    TAI_QUAN_QIU = 2.0
    TAI_QIANG_GANG = 2.0

    # ==============================
    # 训练参数
    # ==============================
    SEED = 42
    MAX_TIMESTEPS = 1_000_000
    CHECKPOINT_FREQ = 100
    BATCH_SIZE = 4096
    MINIBATCH_SIZE = 128
    EPOCHS = 10
    LR_START = 3e-4
    LR_MIN = 1e-6
    MAX_GRAD_NORM = 0.5
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    EPS_CLIP = 0.2
    ENTROPY_COEF = 0.01
    VF_COEF = 0.5
    USE_EARLY_STOPPING = True
    PATIENCE = 50
    MIN_DELTA = 0.001

    # ==============================
    # 网络参数
    # ==============================
    CNN_CHANNEL_IN = 5
    CNN_CHANNEL_OUT = 64
    USE_LSTM = True
    HISTORY_LEN = 30
    LSTM_INPUT_DIM = 41 + 4
    LSTM_HIDDEN_DIM = 128
    LSTM_LAYERS = 2
    FC_HIDDEN_DIM = 512
    OPPONENT_POOL_SIZE = 10
    UPDATE_OPPONENT_FREQ = 500