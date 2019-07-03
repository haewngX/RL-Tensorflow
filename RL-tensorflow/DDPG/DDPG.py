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
        return states, actions, rewards[:, np.newaxis], next_states, terminals[:, np.newaxis]

# Actor
class Actor(object):
    def __init__(self, act_dim,  scope):
        self.act_dim = act_dim  # 动作空间的维度
        self.scope = scope  # 网络的名字(eval网络还是target网络）

    # 构建网络
    def build_network(self, s, reuse):
        with tf.variable_scope(self.scope, reuse=reuse):
            h1 = tf.layers.dense(s, 64, tf.nn.relu,
                                 kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2)))
            h2 = tf.layers.dense(h1, 64, tf.nn.relu,
                                 kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2)))
            action = tf.layers.dense(h2, self.act_dim, tf.nn.tanh,
                                 kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2)))
        return action

    # 根据网络的输出得到动作
    def get_action(self, obs, reuse=False):
        action = self.build_network(obs, reuse)
        return action

# Critic
class Critic(object):
    def __init__(self, scope):
        self.scope = scope  # 网络的名字(eval网络还是target网络）

    # 构建网络
    def build_network(self, s, a, reuse):
        with tf.variable_scope(self.scope, reuse=reuse):
            h1 = tf.layers.dense(s, 64, tf.nn.relu,
                                 kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2)))
            h1 = tf.concat([h1, a], axis=-1)
            h2 = tf.layers.dense(h1, 64, tf.nn.relu,
                                 kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2)))
            value = tf.layers.dense(h2, 1,
                                    kernel_initializer=tf.orthogonal_initializer(gain=0.01))
            return value

    # 根据网络的输出得到Q值
    def get_q(self, s, a, reuse=False):
        q_value = self.build_network(s, a, reuse)
        return q_value


# DDPG
class DDPG(object):
    def __init__(self, a_dim, s_dim, lr_actor, lr_critic, gamma,
                 tau, action_noise_std, action_bound):
        self.a_dim = a_dim  # 动作空间维度
        self.s_dim = s_dim  # 状态空间维度
        self.lr_actor = lr_actor  # actor学习率
        self.lr_critic = lr_critic  # critic学习率
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # target网络soft更新的参数
        self.action_noise_std = action_noise_std  # 实际动作执行时的噪声
        self.action_bound = action_bound  # 连续动作的区间[-self.action_bound,self.action_bound]

        # 各种占位符
        self.s = tf.placeholder(tf.float32, [None, self.s_dim])
        self.s_ = tf.placeholder(tf.float32, [None, self.s_dim])
        self.a = tf.placeholder(tf.float32, [None, self.a_dim])
        self.r = tf.placeholder(tf.float32, [None, 1])
        self.target_q = tf.placeholder(tf.float32, [None, 1])
        self.d = tf.placeholder(tf.float32, [None, 1])

        # 初始化Actor网络和Critic网络
        actor = Actor(self.a_dim, 'actor')
        critic = Critic('critic')

        # 初始化target网络
        target_actor = Actor(self.a_dim, 'target_actor')
        target_critic = Critic('target_critic')

        # 初始化经验池
        self.memory = ReplayMemory(8000)

        # 得到神经网络对当前状态选择的动作及Q值 Q(s,u(s))
        self.action = actor.get_action(self.s)
        self.q_value_with_actor = critic.get_q(self.s, self.action)
        # 得到实际交互中当前状态动作的Q值 Q(s,a)
        q_value = critic.get_q(self.s, self.a, reuse=True)
        # 用target网络计算s_的动作和target_q_value（即公式中的yi)
        target_action = target_actor.get_action(self.s_)
        self.target_q_value = self.r + (1. - self.d) * self.gamma * target_critic.get_q(self.s_, target_action)
        # critic的loss为实际Q值和yi的均方差
        critic_loss = tf.reduce_mean(tf.square(q_value - self.target_q))
        # 梯度下降更新critic
        self.critic_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_critic).minimize(critic_loss)
        # actor的loss可以看作使Q(s,u(s))越大越好
        actor_loss = -tf.reduce_mean(self.q_value_with_actor)
        # 梯度下降更新actor
        self.actor_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_actor).minimize(actor_loss)

        # 收集各个网络参数
        self.actor_params = tf.global_variables('actor')
        self.target_actor_params = tf.global_variables('target_actor')
        self.q_value_params = tf.global_variables('critic')
        self.target_q_value_params = tf.global_variables('target_critic')

        # 初始化网络参数 target和eval参数一致
        self.target_init_updates = \
            [[tf.assign(ta, a), tf.assign(tc, c)]
             for ta, a, tc, c in zip(self.target_actor_params, self.actor_params,
                                     self.target_q_value_params, self.q_value_params)]

        # soft更新target网络参数
        self.target_soft_updates = \
            [[tf.assign(ta, (1 - tau) * ta + tau * a), tf.assign(tc, (1 - tau) * tc + tau * c)]
             for ta, a, tc, c in zip(self.target_actor_params, self.actor_params,
                                     self.target_q_value_params, self.q_value_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init_updates)

    # 选择动作
    def choose_action(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        action = self.sess.run(self.action, feed_dict={self.s: s})
        action = action + np.random.normal(0, self.action_noise_std)
        action = np.clip(action, -self.action_bound, self.action_bound).squeeze(axis=1)
        return action

    # 更新参数
    def learn(self):
        # 从记忆库中选择batch
        s, a, r, s_, d = self.memory.sample(128)
        # 计算yi
        target_q_value = self.sess.run(self.target_q_value, feed_dict={self.s_: s_, self.r: r, self.d: np.float32(d)})
        # 更新critic
        self.sess.run(self.critic_train_op,feed_dict={self.s: s, self.a: a, self.target_q: target_q_value})
        # 更新actor
        self.sess.run(self.actor_train_op, feed_dict={self.s: s})
        # 替换target网络参数
        self.sess.run(self.target_soft_updates)
        # 噪声衰减
        self.action_noise_std = max([self.action_noise_std * noise_decay, noise_min])

if __name__ == '__main__':
    # 初始化环境
    env = gym.make('Pendulum-v0')
    env.seed(1)
    env = env.unwrapped
    # 初始化agent
    agent = DDPG(a_dim=env.action_space.shape[0], s_dim=env.observation_space.shape[0],
                        lr_actor=0.0001, lr_critic=0.001, gamma=0.99, tau=0.01, action_noise_std=1, action_bound=2)
    # 初始化超参数
    episode = 300  # episode数
    episode_step = 200  # 每个episode的最大步数
    iteration = 0  # 总步数
    noise_decay = 0.9999  # 噪声的std的衰减率
    noise_min = 0.001  # 噪声的std的最小值
    # 记录时间
    start_time = time.time()
    ep_reward_list = []  # 存放每回合的reward
    mean_ep_reward_list = []  # 整个训练过程的平均reward
    # 循环n个episode
    for e in range(episode):
        s = env.reset()  # 初始化state
        ep_reward = 0  # 初始化每回合的reward
        # 每个回合循环episode_step步
        for t in range(episode_step):
            env.render()  # 显示图形
            a = agent.choose_action(s)  # 选择动作
            s_, r, done, _ = env.step(a)  # 与环境交互得到下一个状态 奖励和done
            agent.memory.append(s, a, r/10, s_, done)  # 储存记忆
            s = s_  # 更新状态
            ep_reward += r  # 更新当前回合reward
            # learn
            if iteration >= 128 * 3:
                agent.learn()
                # 到达终止状态，显示信息，跳出循环
                if done or t == episode_step-1:
                    # 计算运行时间
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    # 输出该回合的累计回报等信息
                    print('Ep: %d ep_reward: %.2f iteration: %d  time: %d:%02d:%02d' % (e, ep_reward,
                                                                                        iteration,
                                                                                        h, m, s))
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
    plt.title('Pendulum-v0', fontsize=size)
    plt.legend()
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=size)  # 设置图例字体的大小和粗细
    plt.savefig('reward.png')
    plt.show()
