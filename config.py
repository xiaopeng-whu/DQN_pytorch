class EnvConfig:
    ENV = "CartPole-v0"
    def get_env_config(self, args):
        self.ENV = args['env'] if args['env'] else self.ENV

class AgentConfig:
    EXPERIMENT_NO = 99  # 实验序号

    START_EPISODE = 0   # 从第几个episode开始
    NUM_EPISODES = 500  # episodes总数量
    MEMORY_CAPA = 50000  # the capacity of ReplayMemory
    MAX_EPS = 1.0   # the maximum of epsilon in eps-greedy
    MIN_EPS = 0.01  # the minimum of epsilon in eps-greedy
    UPDATE_FREQ = 10    # 每10episodes更新一次target network
    DEMO_NUM = 100  # 测试执行的episodes总数量

    LR = 5e-4  # learning rate
    LR_STEP_SIZE = 9999  # learning rate step size
    DECAY_RATE = 0.99  # decay rate
    BATCH_SIZE = 32  # batch size
    GAMMA = 0.99  # gamma

    RES_PATH = './experiments/'  # parent folder storing the experiments' result

    def get_agent_config(self, args):
        self.EXPERIMENT_NO = args['experiment_num'] if args['experiment_num'] else self.EXPERIMENT_NO

        self.LR = args['learning_rate'] if args['learning_rate'] else self.LR
        self.DECAY_RATE = args['decay_rate'] if args['decay_rate'] else self.DECAY_RATE
        self.BATCH_SIZE = args['batch_size'] if args['batch_size'] else self.BATCH_SIZE
        self.NUM_EPISODES = args['num_episodes'] if args['num_episodes'] else self.NUM_EPISODES
        self.GAMMA = args['gamma'] if args['gamma'] else self.GAMMA

        self.LR_STEP_SIZE = args['lr_step_size'] if args['lr_step_size'] else self.LR_STEP_SIZE