import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import random
import time
from collections import deque

# 用队列存放记忆
class ReplayMemory:
    def __init__(self, memory_size):
        self.buffer = deque()
        self.memory_size = memory_size  # 记忆库的大小
        self.memory_counter = 0  # 记忆库的数量

    # 存储记忆
    def append(self, state, action, reward, next_state, terminal):
        state = np.asarray(state).astype(np.float64)
        next_state = np.asarray(next_state).astype(np.float64)
        self.buffer.append((state, action, reward, next_state, terminal))
        if len(self.buffer) >= self.memory_size:
            self.buffer.popleft()
        self.memory_counter += 1  # 记忆库的数量+1

    # 从记忆中选取batch
    def sample(self, size):
        minibatch = random.sample(self.buffer, size)
        states = np.array([data[0] for data in minibatch])
        actions = np.array([data[1] for data in minibatch])
        rewards = np.array([data[2] for data in minibatch])
        next_states = np.array([data[3] for data in minibatch])
        terminals = np.array([data[4] for data in minibatch])
        return states, actions, rewards, next_states, terminals

# DQN算法
class DQN(object):
    def __init__(self, a_dim, s_dim, learning_rate, gamma, epsilon):
        self.a_dim = a_dim  # 动作空间的维度
        self.s_dim = s_dim  # 状态空间的维度
        self.learning_rate = learning_rate  # 梯度下降的学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-greedy策略进行探索

        # 各种占位符
        self.s = tf.placeholder(tf.float32, [None, self.s_dim])
        self.s_ = tf.placeholder(tf.float32, [None, self.s_dim])
        self.a = tf.placeholder(tf.int32, [None], name="action")
        self.r = tf.placeholder(tf.float32, [None], name="reward")
        self.d = tf.placeholder(tf.float32, [None], name="done")

        self.memory = ReplayMemory(8000)  # 初始化记忆库
        self.learn_step_counter = 0  # 学习的步数
        self.replace_target_iter = 100  # 替换target网络的频率

        # 初始化eval网络
        self.q_eval = self.build_network(self.s, "eval_net")
        # 初始化target网络
        self.q_next = self.build_network(self.s_, "target_net")

        # 把动作转为one_hot编码
        self.action_onehot = tf.one_hot(self.a, self.a_dim, dtype=tf.float32)
        # 选择对应one_hot编码动作的q_eval值
        self.q_eval_wrt_a = tf.reduce_sum(tf.multiply(self.q_eval, self.action_onehot), axis=1)
        # 根据target网络的maxq(s',a)计算q_target
        q_target = self.r + (1. - self.d) * self.gamma * tf.reduce_max(self.q_next, axis=1)
        self.q_target = tf.stop_gradient(q_target)
        # 构建损失函数
        self.loss = tf.reduce_mean(tf.square(self.q_eval_wrt_a - self.q_target))
        # 更新参数的操作
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # 收集q_eval网络的参数
        self.q_eval_params = tf.global_variables("eval_net")
        # 收集q_target网络的参数
        self.q_target_params = tf.global_variables("target_net")
        # 替换q_target网络的参数
        self.target_updates = [tf.assign(tq, q) for tq, q in zip(self.q_target_params, self.q_eval_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_updates) # 替换q_target网络的参数操作

    # 选择动作
    def choose_action(self, s):
        if s.ndim < 2: s = s[np.newaxis, :] # 扩展一个batch上的维度
        action = self.sess.run(self.q_eval, feed_dict={self.s: s})  # 用eval网络计算q值
        if np.random.rand(1) < self.epsilon:  # 如果小于epsilon，动作随机
            action = np.random.randint(0, 2)
        else:
            action = np.argmax(action, axis=1)[0] # 如果大于等于epsilon，动作为网络中Q值最大的
        return action

    # 更新参数
    def learn(self):
        # 替换target网络参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_updates)
        # 从记忆库中选择batch
        s, a, r, s_, d = self.memory.sample(128)
        # 更新参数
        self.sess.run(self.train_op,feed_dict={self.s: s, self.a: a,
                                                       self.s_: s_, self.r: r, self.d: np.float32(d)})
        self.learn_step_counter += 1 # 学习的步数+1

    # 构建神经网络
    def build_network(self, s, scope):
        with tf.variable_scope(scope):
            # 权重初始化
            w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
            # 第一层
            h1 = tf.layers.dense(s, 10, tf.nn.tanh,
                                 kernel_initializer=w_initializer, bias_initializer=b_initializer)
            # 输出Q值
            value = tf.layers.dense(h1, self.a_dim,
                                    kernel_initializer=w_initializer, bias_initializer=b_initializer)
            return value

if __name__ == '__main__':
    # 初始化环境
    env = gym.make('CartPole-v0')
    env.seed(1)
    env = env.unwrapped
    # 初始化agent
    agent = DQN(a_dim=env.action_space.n, s_dim=env.observation_space.shape[0],
                learning_rate=0.01, gamma=0.999, epsilon=0.2)
    # 初始化超参数
    nepisode = 300  # episode数
    episode_step = 5000  # 每个episode的最大步数
    iteration = 0  # 总步数
    epsilon_step = 10  # epsilon衰减的步数
    epsilon_decay = 0.99  # epsilon衰减率
    epsilon_min = 0.001  # epsilon的最小值
    # 记录时间
    start_time = time.time()
    ep_reward_list = []  # 存放每回合的reward
    mean_ep_reward_list = []  # 整个训练过程的平均reward
    # 循环nepisode个episode
    for e in range(nepisode):
        s = env.reset()  # 初始化state
        ep_reward = 0  # 初始化每回合的reward
        # 每个回合循环episode_step步
        # for t in range(episode_step):
        while True:
            env.render()  # 显示图形
            a = agent.choose_action(s)  # 选择动作
            s_, r, done, _ = env.step(a)  # 与环境交互得到下一个状态 奖励和done
            agent.memory.append(s, a, r, s_, done)  # 储存记忆
            s = s_  # 更新状态
            ep_reward += r  # 更新当前回合reward
            # learn
            if iteration >= 128 * 3:
                agent.learn()
                # epsilon衰减
                if iteration % epsilon_step == 0:
                    agent.epsilon = max([agent.epsilon * 0.99, 0.001])
            # 到达终止状态，显示信息，跳出循环
            if done:
                # 计算运行时间
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                # 输出该回合的累计回报等信息
                print('Ep: %d ep_reward: %.2f iteration: %d epsilon: %.3f time: %d:%02d:%02d' % (e, ep_reward,
                                                                                               iteration,
                                                                                               agent.epsilon, h, m, s))
                ep_reward_list.append(ep_reward)
                average = np.mean(np.array(ep_reward_list))
                mean_ep_reward_list.append(average)
                break
            iteration += 1
    # 画图
    plt.plot(range(len(ep_reward_list)), ep_reward_list, color="red", label="ep_reward", linewidth=1.5, linestyle='--')
    plt.plot(range(len(mean_ep_reward_list)), mean_ep_reward_list, color="green", label="mean_reward", linewidth=1.5)
    size = 12
    plt.xticks(fontsize=size)  # 默认字体大小为10
    plt.yticks(fontsize=size)
    plt.ylabel('reward')
    plt.xlabel('episodes')
    plt.title('CartPole-v0', fontsize=size)
    plt.legend()
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=size)  # 设置图例字体的大小和粗细
    plt.savefig('reward.png')
    plt.show()