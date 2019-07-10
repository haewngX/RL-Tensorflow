import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import time
import tensorflow_probability as tfp


# 存放一个episode的轨迹
class Memory(object):
    def __init__(self):
        # 轨迹中的状态、动作、回报、下一个状态和done
        self.ep_s, self.ep_a, self.ep_r, self.ep_d, self.ep_s_= [], [], [], [], []

    # 存到轨迹中
    def append(self, s, a, r, d, s_):
        self.ep_s.append(s)
        self.ep_a.append(a)
        self.ep_r.append(r)
        self.ep_d.append(d)
        self.ep_s_.append(s_)

    # 重置
    def reset(self):
        self.ep_s, self.ep_a, self.ep_r, self.ep_d, self.ep_s_= [], [], [], [], []



# Actor
class Actor(object):
    def __init__(self, act_dim,  scope):
        self.act_dim = act_dim  # 动作空间的维度
        self.scope = scope  # 网络的名字(eval网络还是target网络）

    # 构建网络 输出策略 一个正态分布
    def build_network(self, s, reuse):
        with tf.variable_scope(self.scope, reuse=reuse):
            w_init = tf.random_normal_initializer(0., .1)
            l_a = tf.layers.dense(s, 200, tf.nn.relu, kernel_initializer=w_init)
            mu = tf.layers.dense(l_a, self.act_dim, tf.nn.tanh, kernel_initializer=w_init)
            sigma = tf.layers.dense(l_a, self.act_dim, tf.nn.softplus, kernel_initializer=w_init)
        return mu, sigma

    # 计算实际交互中选择动作的概率的log_p(a|s)
    def get_logp(self, s, a, reuse=True):
        mu, sigma = self.build_network(s, reuse)
        mu, sigma = mu * 2, sigma + 1e-4
        normal_dist = tfp.distributions.Normal(mu, sigma)
        log_prob = normal_dist.log_prob(a)
        return log_prob


# Critic
class Critic(object):
    def __init__(self, scope):
        self.scope = scope  # 网络的名字(eval网络还是target网络）

    # 构建网络
    def build_network(self, s, reuse):
        with tf.variable_scope(self.scope, reuse=reuse):
            # 权重初始化
            w_init = tf.random_normal_initializer(0., .1)
            l_c = tf.layers.dense(s, 100, tf.nn.relu, kernel_initializer=w_init)
            value = tf.layers.dense(l_c, 1, kernel_initializer=w_init)
            return value

    # 根据网络的输出得到V值
    def get_v(self, s, reuse=True):
        value = self.build_network(s, reuse)
        return value


# A3C
class A3C(object):
    def __init__(self, scope, globalAC, a_dim, s_dim, lr_a, lr_c, gamma):  # 这个scope指的是worker类型 local或者global
        self.a_dim = a_dim  # 动作空间维度
        self.s_dim = s_dim  # 状态空间维度
        self.lr_a = lr_a  # actor学习率
        self.lr_c = lr_c  # critic学习率
        self.gamma = gamma  # 折扣因子

        # 初始化经验池
        self.memory = Memory()

        if scope == global_net_scope:  # global network 只收集参数
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.s_dim])
                # 初始化Actor和Critic
                actor = Actor(self.a_dim, 'actor')
                critic = Critic('critic')
                # 初始化网络
                mu, sigma = actor.build_network(self.s, tf.AUTO_REUSE)
                v = critic.build_network(self.s, tf.AUTO_REUSE)
                # Global参数
                self.actor_params = tf.global_variables(scope + '/actor')
                self.critic_params = tf.global_variables(scope + '/critic')
        else:  # local net计算loss
            with tf.variable_scope(scope):
                # 各种占位符
                self.s = tf.placeholder(tf.float32, [None, self.s_dim])
                self.a = tf.placeholder(tf.float32, [None, self.a_dim])
                self.Q = tf.placeholder(tf.float32, [None, ])

                # 初始化Actor和Critic
                actor = Actor(self.a_dim, 'actor')
                critic = Critic('critic')

                # 计算策略的正态分布
                self.mu, self.sigma = actor.build_network(self.s, tf.AUTO_REUSE)
                self.mu, self.sigma = self.mu * 2, self.sigma + 1e-4
                normal_dist = tfp.distributions.Normal(self.mu, self.sigma)
                self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), -2, 2)

                # 计算状态值
                self.v = critic.build_network(self.s, tf.AUTO_REUSE)

                # local参数
                self.actor_params = tf.global_variables(scope + '/actor')
                self.critic_params = tf.global_variables(scope + '/critic')

                # critic损失函数  最小化优势函数
                self.advantage = self.Q - self.v
                self.c_loss = tf.square(self.advantage)

                # actor损失函数 最小化-(log_p * td_error)
                log_prob = actor.get_logp(self.s, self.a, tf.AUTO_REUSE)
                exp_v = log_prob * tf.stop_gradient(self.advantage)
                entropy = normal_dist.entropy()  # 实际中加了一项entropy项，作为一个regularizer，让policy更随机
                self.exp_v = entropy_beta * entropy + exp_v
                self.a_loss = tf.reduce_mean(-self.exp_v)

                # 计算当前worker的梯度
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.actor_params)
                    self.c_grads = tf.gradients(self.c_loss, self.critic_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):  # 获取global参数,复制到local—net
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in
                                             zip(self.actor_params, globalAC.actor_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in
                                             zip(self.critic_params, globalAC.critic_params)]
                with tf.name_scope('push'):  # 将参数传送到global中去
                    # 其中传送的是local—net的actor和critic的参数梯度grads,具体计算在上面定义
                    # apply_gradients是tf.train.Optimizer中自带的功能函数，将求得的梯度参数更新到global中
                    self.update_a_op = tf.train.RMSPropOptimizer(self.lr_a).apply_gradients(zip(self.a_grads, globalAC.actor_params))
                    self.update_c_op = tf.train.RMSPropOptimizer(self.lr_c).apply_gradients(zip(self.c_grads, globalAC.critic_params))

    # 用local更新global参数
    def update_global(self):
        s, a, done = np.vstack(self.memory.ep_s), np.array(self.memory.ep_a), self.memory.ep_d[-1]
        q = self.compute_q(done)
        SESS.run([self.update_a_op, self.update_c_op], {self.s: s, self.Q: q, self.a: a})
        self.memory.reset()  # 重置轨迹

    # 获取global参数,复制到local—net
    def pull_global(self):
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    # 选择动作
    def choose_action(self, s):
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})

    # 计算Q(s,a)，使用r+V(s')来近似，这样会增加一定的方差,但可以忽略不计 V(s)=gamma*V(s')+r(s')
    def compute_q(self, done):
        q = np.zeros_like(self.memory.ep_r)
        if done:
            value = 0  # terminal状态为0
        else:
            value = SESS.run(self.v, {self.s: self.memory.ep_s_[-1][np.newaxis, :]})[0, 0]
        for t in reversed(range(0, len(self.memory.ep_r))):
            value = value * self.gamma + self.memory.ep_r[t]
            q[t] = value
        return q

# 每个CPU上的worker
class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make('Pendulum-v0')
        self.env.seed(1)
        self.env = self.env.unwrapped
        self.name = name
        self.AC = A3C(name, globalAC,a_dim=self.env.action_space.shape[0], s_dim=self.env.observation_space.shape[0], lr_a=0.0001, lr_c=0.001, gamma=0.99)

    def work(self):
        global ep_global_reward_list, global_episode, global_e
        iteration = 1  # 总步数
        # 记录时间
        start_time = time.time()
        while not COORD.should_stop() and global_e < global_episode:
            s = self.env.reset()  # 初始化state
            ep_reward = 0  # 初始化每回合的reward
            # 每个回合循环episode_step步
            for t in range(episode_step):
                if self.name == 'W_0':
                    self.env.render()
                a = self.AC.choose_action(s)  # 选择动作
                # print(a.shape)
                s_, r, done, _ = self.env.step(a)  # 与环境交互得到下一个状态 奖励和done
                self.AC.memory.append(s, a, (r+8)/8,done,s_)  # 储存记忆
                ep_reward += r  # 更新当前回合reward
                # 更新
                if iteration % update_global_iter == 0 or done or t == episode_step - 1:
                    self.AC.update_global()
                    self.AC.pull_global()
                s = s_  # 更新状态
                iteration+=1
                if done or t == episode_step - 1:
                    # 计算运行时间
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    # 输出该回合的累计回报等信息
                    if len(ep_global_reward_list) == 0:
                        ep_global_reward_list.append(ep_reward)
                    else:
                        ep_global_reward_list.append(0.9 * ep_global_reward_list[-1] + 0.1 * ep_reward)
                    average = np.mean(np.array(ep_global_reward_list))
                    mean_global_ep_reward_list.append(average)
                    print(self.name, 'Ep: %d ep_reward: %.2f time: %d:%02d:%02d' % (global_e, ep_global_reward_list[-1], h, m, s))
                    global_e += 1
                    break


if __name__ == "__main__":
    n_workers = multiprocessing.cpu_count()  # worker的数量
    episode_step = 200  # 每个episode中最大step数
    global_episode = 5000  # 最大global episode数
    global_net_scope = 'Global_Net'  # 全局网络的名字
    update_global_iter = 10  # 全局网络更新的频率
    entropy_beta = 0.01  # entropy项的系数beta
    ep_global_reward_list = []  # 存放的reward
    mean_global_ep_reward_list = []  # 整个训练过程的平均reward
    global_e = 0  # 当前episode代数
    SESS = tf.Session()
    env = gym.make('Pendulum-v0')
    with tf.device("/cpu:0"):
        GLOBAL_AC = A3C(global_net_scope,None,a_dim=env.action_space.shape[0], s_dim=env.observation_space.shape[0], lr_a=0.0001, lr_c=0.001, gamma=0.99)  # we only need its params
        workers = []
        # Create worker
        for i in range(n_workers):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()  # 使用 tf.train.Coordinator()来创建一个线程管理器（协调器）对象。
    SESS.run(tf.global_variables_initializer())
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)  # job为该线程需要执行的方法
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)  # 把开启的线程加入

    # 画图
    plt.plot(range(len(ep_global_reward_list)), ep_global_reward_list, color="red", label="ep_reward", linewidth=1.5, linestyle='--')
    plt.plot(range(len(mean_global_ep_reward_list)), mean_global_ep_reward_list, color="green", label="mean_reward", linewidth=1.5)
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
