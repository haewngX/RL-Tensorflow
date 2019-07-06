import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import random
import time

# 存放一个episode的轨迹
class Memory(object):
    def __init__(self):
        # 轨迹中的状态、动作、回报
        self.ep_s, self.ep_a, self.ep_r = [], [], []

    # 存到轨迹中
    def append(self, s, a, r):
        self.ep_s.append(s)
        self.ep_a.append(a)
        self.ep_r.append(r)

    # 重置
    def reset(self):
        self.ep_s, self.ep_a, self.ep_r = [], [], []

# Policy
class Policy(object):
    def __init__(self, act_dim,  scope):
        self.act_dim = act_dim  # 动作空间的维度
        self.scope = scope  # 网络的名字(eval网络还是target网络）

    # 构建网络 输出动作的概率和Q值
    def build_network(self, s, reuse):
        with tf.variable_scope(self.scope, reuse=reuse):
            h1 = tf.layers.dense(s, 10, tf.nn.tanh,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            all_act = tf.layers.dense(h1, self.act_dim, None,
                                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            all_act_prob = tf.nn.softmax(all_act)  # 根据网络的输出用softmax转为动作的概率
        return all_act_prob, all_act

    # 计算实际交互中选择动作的概率的-log_p(a|s)
    def get_neglogp(self, s, a, reuse=True):
        all_act_prob, _ = self.build_network(s, reuse)
        neg_log_prob = tf.reduce_sum(-tf.log(all_act_prob) * tf.one_hot(a, self.act_dim), axis=1)
        return neg_log_prob

# PG
class PG(object):
    def __init__(self, a_dim, s_dim, lr,  gamma):
        self.a_dim = a_dim  # 动作空间维度
        self.s_dim = s_dim  # 状态空间维度
        self.lr = lr  # actor学习率
        self.gamma = gamma  # 折扣因子

        # 各种占位符
        self.s = tf.placeholder(tf.float32, [None, self.s_dim])
        self.a = tf.placeholder(tf.int32, [None, ])
        self.discount_r = tf.placeholder(tf.float32, [None, ])

        # 初始化Policy策略网络
        policy = Policy(self.a_dim, 'policy')

        # 初始化经验池
        self.memory = Memory()

        # 计算状态s时动作的概率
        self.all_act_prob, _ = policy.build_network(self.s, False)
        # 计算实际交互中选择动作的概率的-log_p(a|s)
        neg_log_prob = policy.get_neglogp(self.s, self.a)
        # 损失函数 最小化-(log_p * R(tau))
        loss = tf.reduce_mean(neg_log_prob * self.discount_r)
        # 梯度下降更新
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    # 选择动作
    def choose_action(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        # 计算状态s时动作的概率
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.s: s})
        # 根据动作概率选择实际的动作
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    # 更新参数
    def learn(self):
        # 计算归一化的轨迹上的折扣回报
        discounted_ep_rs_norm = self.discount_and_norm_rewards()

        # 训练
        self.sess.run(self.train_op, feed_dict={
            self.s: np.vstack(self.memory.ep_s),  # shape=[None, n_obs]
            self.a: np.array(self.memory.ep_a),  # shape=[None, ]
            self.discount_r: discounted_ep_rs_norm,  # shape=[None, ]
        })
        self.memory.reset()  # 重置轨迹

    # 逆向计算每一个step的折扣回报 R(t)=gamma*R(t+1)+r(t+1)
    def discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.memory.ep_r)
        running_add = 0
        for t in reversed(range(0, len(self.memory.ep_r))):
            running_add = running_add * self.gamma + self.memory.ep_r[t]
            discounted_ep_rs[t] = running_add
        # 归一化
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

if __name__ == '__main__':
    # 初始化环境
    env = gym.make('CartPole-v0')
    env.seed(1)
    env = env.unwrapped
    # 初始化agent
    agent = PG(a_dim=env.action_space.n, s_dim=env.observation_space.shape[0], lr=0.02, gamma=0.99)
    # 初始化超参数
    episode = 120  # episode数
    episode_step = 200  # 每个episode的最大步数
    # 记录时间
    start_time = time.time()
    ep_reward_list = []  # 存放每回合的reward
    mean_ep_reward_list = []  # 整个训练过程的平均reward
    # 循环n个episode
    for e in range(episode):
        s = env.reset()  # 初始化state
        ep_reward = 0  # 初始化每回合的reward
        # 每个回合循环episode_step步
        while True:
        # for t in range(episode_step):
            env.render()  # 显示图形
            a = agent.choose_action(s)  # 选择动作
            s_, r, done, _ = env.step(a)  # 与环境交互得到下一个状态 奖励和done
            agent.memory.append(s, a, r)  # 储存记忆
            s = s_  # 更新状态
            ep_reward += r  # 更新当前回合reward
            # 到达终止状态，显示信息，跳出循环 learn
            if done:
                agent.learn()
                # 计算运行时间
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                # 输出该回合的累计回报等信息
                print('Ep: %d ep_reward: %.2f time: %d:%02d:%02d' % (e, ep_reward,h, m, s))
                ep_reward_list.append(ep_reward)
                average = np.mean(np.array(ep_reward_list))
                mean_ep_reward_list.append(average)
                break
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