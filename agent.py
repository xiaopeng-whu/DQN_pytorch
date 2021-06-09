import torch
import gym
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from memory import ReplayMemory, Transition
from networks import DQN
from config import AgentConfig, EnvConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(AgentConfig, EnvConfig):
    def __init__(self, args):
        self.get_env_config(args)
        self.get_agent_config(args)
        self.build()

    def build(self):    # build the environment and agent
        self.env = gym.make(self.ENV)
        self.memory = ReplayMemory(capacity = self.MEMORY_CAPA)
        self.num_actions = self.env.action_space.n
        self.policy_net = DQN(self.num_actions, input_size=4, hidden_size=32).to(device) # 将模型加载到cpu上
        self.target_net = DQN(self.num_actions, input_size=4, hidden_size=32).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())   # 初始化两个网络参数相同
        self.epsilon = self.MAX_EPS

    def eps_decay(self):    # epsilon随时间衰减至MIN_EPS
        self.epsilon = max(self.epsilon * self.DECAY_RATE, self.MIN_EPS)

    def greedy_action(self, state, eps):
        # 起始完全随机抽样选择动作，慢慢提高按照max选取动作的概率
        if torch.rand(1).item() > eps:  # item()将tensor值返回为python number
            with torch.no_grad():   # 在使用pytorch时，并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）
                q_values = self.policy_net(state.unsqueeze(0).float())  # unsqueeze在0位置加一维，留给batchsize更好地批处理
                action = q_values.max(1)[1].view(1) # q_values.max返回Tensor对象每行的最大列值结果，结果的第二列是该最大值元素的indices，view就是reshape
        else:
            action = torch.tensor([self.env.action_space.sample()], device=device, dtype=torch.long)
        return action

    def policy_action(self, state): # 训练到一定程度进行测试时使用训练好的network的policy
        with torch.no_grad():
            # input a state_batch -> state_action_values(q_value)
            q_values = self.policy_net(state.unsqueeze(0).float())
            action = q_values.max(1)[1].view(1)
        return action

    def train(self):
        # define the optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.LR)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.LR_STEP_SIZE, gamma=0.5)  # 每step_size个epoch，学习率进行一次更新调整
        # define the records
        self.episode_durations = []     # 每个episode的持续时间，该环境设置下持续时间越长说明表现越好。这个记录的是训练时greedy-policy的结果。
        self.policy_net_scores = []     # 这个和上一项貌似是一样的效果。这个记录的是policy network的结果。
        self.eps_list = []  # 记录当前epsilon
        self.lr_list = []   # 记录当前学习率

        # 训练agent
        for i_episode in range(self.START_EPISODE, self.NUM_EPISODES):
            state = self.env.reset()
            state = torch.from_numpy(state)

            # decay the epsilon
            self.eps_decay()

            for t in range(501):
                action = self.greedy_action(state, self.epsilon)
                obs, reward, done, _ = self.env.step(action.item()) # item()方法将一个标量Tensor转化为一个python number
                if done:
                    reward = -1.
                reward = torch.tensor([reward], device=device)
                done = torch.tensor([done], device=device)  # 转化为tensor，后面传入ReplayMemory中

                if not done:
                    next_state = torch.from_numpy(obs)
                else:
                    next_state = torch.zeros_like(state)

                # push transition to the memory
                self.memory.push(state, action, next_state, reward, done)

                state = next_state

                self.optimize_model()

                if done:
                    self.episode_durations.append(t + 1)
                    break
            # optimizer.param_groups[0]：长度为6的字典，包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数
            cur_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            print("Episode {} finished after {} timesteps -- EPS: {:.4f} -- LR: {:.6f}".format(i_episode, t + 1, self.epsilon, cur_lr))
            self.lr_list.append(cur_lr)
            self.policy_net_scores.append(self.execute())
            self.eps_list.append(self.epsilon)
            # 一定时间周期后更新target网络
            if (i_episode + 1) % self.UPDATE_FREQ == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print("Target network updated!")
            self.scheduler.step()   # optimizer更新后，scheduler再更新optimizer的学习率
        print("Train progess completes!")

    def optimize_model(self):   # 算法更新关键模块
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitons = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitons))   # 转置批样本

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        done_batch = torch.stack(batch.done)
        next_state_batch = torch.stack(batch.next_state)

        not_done_mask = [k for k, v in enumerate(done_batch) if v == 0] # 非终止状态的掩码（enumerate将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标）
        not_done_next_states = next_state_batch[not_done_mask]

        state_action_values = self.policy_net(state_batch.float()).gather(1, action_batch)  # 将网络输出的二维tensor值按照采样得到的action选择对应的值作为q值
        # 下一个状态的V值是通过旧的target网络计算得到的，通过掩码判断状态终止与否，获得期望状态值或0
        next_state_values = torch.zeros_like(state_action_values)
        next_state_values[not_done_mask] = self.target_net(not_done_next_states.float()).max(1)[0].view(-1, 1).detach() # 和前面的action选取操作一样，detach()切断反向传播，不需要计算该tensor的梯度
        target_values = reward_batch + (self.GAMMA * next_state_values) # 计算期望Q值

        # 计算Huber loss  # SooothL1Loss其实是L2Loss和L1Loss的结合，它同时拥有L2 Loss和L1 Loss的部分优点。
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        t = torch.abs(state_action_values - target_values)
        losses = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)  # 手动实现SooothL1Loss
        # 计算最终平均loss
        loss = torch.mean(losses).to(device)
        # Optimize the model
        self.optimizer.zero_grad()  # 每次运算都要将上一次所有参数的梯度清空
        loss.backward() # 反向传播
        for key, param in self.policy_net.named_parameters():
            param.grad.data.clamp_(-1, 1)   # 梯度截断，将所有梯度限制在-1至1之间
        self.optimizer.step()   # 更新模型参数

    def execute(self):
        scores = []
        for i_episode in range(self.DEMO_NUM):
            state = self.env.reset()
            state = torch.from_numpy(state)

            for t in range(501):    # 假设最多500个timesteps内终止一个episode
                action = self.policy_action(state)  # 从train一段时间后的policy_net来获取动作执行测试
                obs, _, done, _ = self.env.step(action.item())
                state = torch.from_numpy(obs)

                if done:
                    scores.append(t + 1)    # 该环境设置为持续时间越长则total reward越高
                    break
        net_score = np.array(scores, dtype = float)

        return net_score.mean()

    def save_results(self):
        # plot and save figure
        plt.figure(0)
        policy_net_scores = torch.tensor(self.policy_net_scores, dtype=torch.float)
        plt.title("DQN Experiment %d" % self.EXPERIMENT_NO)
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(policy_net_scores.numpy())     # 当前episode对应的policy network的表现
        plt.plot(np.array(self.episode_durations, dtype=np.float))  # 采用greedy policy的表现
        # Take 10 episode policy net score averages and plot them too
        if len(policy_net_scores) >= 10:
            means = policy_net_scores.unfold(0, 10, 1).mean(1).view(-1)     # unfold是手动实现的滑动窗口作用
            means = torch.cat((torch.zeros(9), means))  # 前9次episode的平均结果置零
            plt.plot(means.numpy())     # 最近10次episode对应的policy network的平均表现
        plt.savefig(self.RES_PATH + "%d-result.png" % self.EXPERIMENT_NO)
        results_dict = {
            'policy_net_scores': policy_net_scores.numpy(),
            'episode_durations': np.array(self.episode_durations, dtype=np.float),
            'means': means.numpy()
        }
        torch.save(results_dict, self.RES_PATH + "%d-ret.dict" % self.EXPERIMENT_NO)
        # plt.show()

        self.write_results(self.RES_PATH)

    def write_results(self, PATH):
        attr_dict = {
            "EXPERIMENT_NO": self.EXPERIMENT_NO,
            "START_EPISODE": self.START_EPISODE,
            "NUM_EPISODES": self.NUM_EPISODES,
            "MEMORY_CAPA": self.MEMORY_CAPA,
            "MAX_EPS": self.MAX_EPS,
            "MIN_EPS": self.MIN_EPS,
            "UPDATE_FREQ": self.UPDATE_FREQ,
            "DEMO_NUM": self.DEMO_NUM,

            "LR": self.LR,
            "LR_STEP_SIZE": self.LR_STEP_SIZE,
            "DECAY_RATE": self.DECAY_RATE,
            "BATCH_SIZE": self.BATCH_SIZE,
            "GAMMA": self.GAMMA,

            "RES_PATH": self.RES_PATH
        }
        with open(PATH + "%d-log.txt" % self.EXPERIMENT_NO, 'w') as f:
            for k, v in attr_dict.items():
                f.write("{} = {}\n".format(k, v))
            f.write("------------------\n")
            for i in range(len(self.episode_durations)):
                f.write("Ep %d finished after %d steps -- EPS: %.4f -- LR: %.6f -- policy net score: %.2f\n"
                    % (i + 1, self.episode_durations[i], self.eps_list[i], self.lr_list[i], self.policy_net_scores[i]))

    def env_close(self):
        self.env.close()
