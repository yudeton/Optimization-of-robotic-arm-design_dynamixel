import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import TensorBoard
import gym
from gym import spaces
import numpy as np
import os
import datetime


class MotorEnv:
    def __init__(self):
        super(MotorEnv, self).__init__()  # 调用父类的构造函数

        # self.state_dim = 58     # 狀態空間維度
        # self.action_dim = 6     # 離散動作空間維度
        # self.param_dim = 2      # 每個離散動作的連續參數維度
        # self.max_steps = 100    # 每個 episode 的最大步數

        # 定义状态空间的边界值
        self.low_torque_over = -1.0
        self.high_torque_over = 1.0
        self.low_power_consumption = -1.0
        self.high_power_consumption = 1.0
        self.low_reach_eva = -1.0
        self.high_reach_eva = 1.0
        self.low_manipulability = -1.0
        self.high_manipulability = 1.0
        self.low_std_L2 = -1.0
        self.high_std_L2 = 1.0
        self.low_std_L3 = -1.0
        self.high_std_L3 = 1.0
        self.low_torque_cost = -1.0
        self.high_torque_cost = 1.0

        # 定义状态空间
        self.observation_space = spaces.Box(
            low=np.array([self.low_torque_over, self.low_power_consumption, self.low_reach_eva, self.low_manipulability, self.low_std_L2, self.low_std_L3, self.low_torque_cost]),
            high=np.array([self.high_torque_over, self.high_power_consumption, self.high_reach_eva, self.high_manipulability, self.high_std_L2, self.high_std_L3, self.high_torque_cost]),
            dtype=np.float64
        )

        # 定义动作空间：6个离散动作，每个动作有2个连续参数
        self.action_space = spaces.Tuple((
            spaces.Discrete(6),
            spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
        ))

        # 定义额外的属性
        self.state_dim = self.observation_space.shape[0]  # 状态空间维度
        self.action_dim = self.action_space.spaces[0].n  # 离散动作空间维度
        self.param_dim = self.action_space.spaces[1].shape[0]  # 每个离散动作的连续参数维度
        self.max_steps = 100  # 每个 episode 的最大步数


    def reset(self):
        """
        重製環境狀態，返回一個新的初始狀態
        """
        self.steps = 0
        self.state = np.random.randn(self.state_dim) # 生成隨機初始狀態
        return self.state

    def step(self, action, params):
        """
        執行給定的動作及其參數，返回新的狀態,獎勵和是否結束
        """
        self.steps += 1
        next_state = np.random.randn(self.state_dim)    # 生成隨機的新狀態
        reward = self.calculate_reward(action, params)  # 计算奖励
        done = self.steps >= self.max_steps             # 判斷是否達到最大步數
        return next_state, reward, done

    def calculate_reward(self, action, params):
        """
        根据动作和参数计算奖励。
        """
        





        # 简化的奖励计算示例，实际情况根据具体任务定义
        reward = -np.sum(params**2)  # 假设连续参数的平方和越小奖励越高
        return reward

class QNet(Model):
    def __init__(self, input_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.fc3 = layers.Dense(64, activation='relu')
        self.out = layers.Dense(action_dim)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.out(x)

class ParamNet(Model):
    def __init__(self, input_dim, param_dim):
        super(ParamNet, self).__init__()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.fc3 = layers.Dense(64, activation='relu')
        self.out = layers.Dense(param_dim, activation='tanh')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.out(x)

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, n_params):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.param_memory = np.zeros((self.mem_size, n_params), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, params, reward, state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.param_memory[index] = params
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 0

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        params = self.param_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, params, rewards, states_

class PDQNAgent:
    def __init__(self, state_dim, action_dim, param_dim, buffer_size=50000, batch_size=64, gamma=0.99, lr_q=0.001, lr_p=0.001, log_dir='./logs'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.param_dim = param_dim
        self.buffer = ReplayBuffer(buffer_size, (state_dim,), action_dim, param_dim)
        self.batch_size = batch_size
        self.gamma = gamma

        self.q_net = QNet(state_dim + param_dim, action_dim)
        self.target_q_net = QNet(state_dim + param_dim, action_dim)
        self.param_net = ParamNet(state_dim, param_dim)

        self.optimizer_q = tf.keras.optimizers.Adam(learning_rate=lr_q)
        self.optimizer_p = tf.keras.optimizers.Adam(learning_rate=lr_p)
        
        self.update_target_network()
        
        self.log_dir = log_dir
        self.summary_writer = tf.summary.create_file_writer(log_dir)

       # Set up checkpointing
        self.checkpoint_dir = './checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer_q=self.optimizer_q,
                                              optimizer_p=self.optimizer_p,
                                              q_net=self.q_net,
                                              param_net=self.param_net)

    def update_target_network(self):
        self.target_q_net.set_weights(self.q_net.get_weights())

    def choose_action(self, state, epsilon):
        if np.random.random() < epsilon:
            action = np.random.choice(self.action_dim)
            params = np.random.uniform(-1, 1, self.param_dim)
        else:
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            params = self.param_net(state)
            q_input = tf.concat([state, params], axis=1)
            q_values = self.q_net(q_input)
            action = tf.argmax(q_values[0]).numpy()
            params = params.numpy()[0]
        return action, params

    def store_transition(self, state, action, params, reward, state_):
        self.buffer.store_transition(state, action, params, reward, state_)

    def learn(self,episode):
        if self.buffer.mem_cntr < self.batch_size:
            return None, None

        states, actions, params, rewards, states_ = self.buffer.sample_buffer(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        with tf.GradientTape() as tape:
            params_ = self.param_net(states_)
            q_next = self.target_q_net(tf.concat([states_, params_], axis=1))
            q_next = tf.reduce_max(q_next, axis=1)
            q_target = rewards + self.gamma * q_next
            q_target = tf.reshape(q_target, (self.batch_size, 1))

            actions_onehot = tf.one_hot(actions, self.action_dim)
            q_eval = self.q_net(tf.concat([states, params], axis=1))
            q_eval = tf.reduce_sum(q_eval * actions_onehot, axis=1, keepdims=True)

            loss_q = tf.keras.losses.MSE(q_target, q_eval)

        grads = tape.gradient(loss_q, self.q_net.trainable_variables)
        self.optimizer_q.apply_gradients(zip(grads, self.q_net.trainable_variables))

        with tf.GradientTape() as tape:
            x_all = self.param_net(states)
            q_all = self.q_net(tf.concat([states, x_all], axis=1))
            loss_p = -tf.reduce_mean(tf.reduce_sum(q_all, axis=1))

        grads = tape.gradient(loss_p, self.param_net.trainable_variables)
        self.optimizer_p.apply_gradients(zip(grads, self.param_net.trainable_variables))

        self.update_target_network()
        
        loss_q_mean = tf.reduce_mean(loss_q)
        loss_p_mean = tf.reduce_mean(loss_p)

        # 紀錄loss到TensorBoard
        with self.summary_writer.as_default():
            tf.summary.scalar('Loss/Q_loss', loss_q_mean.numpy(), step=episode)
            tf.summary.scalar('Loss/Param_loss', loss_p_mean.numpy(), step=episode)
            self.summary_writer.flush()
        
        return loss_q_mean.numpy(), loss_p_mean.numpy()

    def save_model(self, episode):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        # Save model in SavedModel format
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        q_net_path = f'saved_models/q_net_{episode}'
        param_net_path = f'saved_models/param_net_{episode}'
        tf.saved_model.save(self.q_net, q_net_path)
        tf.saved_model.save(self.param_net, param_net_path)

    def load_model(self, checkpoint_path):
        self.checkpoint.restore(checkpoint_path)

        # Restore SavedModel
        latest_q_net = tf.train.latest_checkpoint('saved_models')
        latest_param_net = tf.train.latest_checkpoint('saved_models')
        self.q_net = tf.saved_model.load(latest_q_net)
        self.param_net = tf.saved_model.load(latest_param_net)


def train_pdqn(agent, env, episodes=20000, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, save_interval=10000):
    epsilon = epsilon_start
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action, params = agent.choose_action(state, epsilon)
            next_state, reward, done = env.step(action, params)
            agent.store_transition(state, action, params, reward, next_state)
            loss_q, loss_p = agent.learn(episode)
            state = next_state
            total_reward += reward
            step_count += 1

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        print(f'Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f},Q_loss: {loss_q},Param_loss: {loss_p}')

        with agent.summary_writer.as_default():
            tf.summary.scalar('Episode_Return', total_reward, step=episode)
            agent.summary_writer.flush()

        # 定期保存模型
        if episode % save_interval == 0:
            agent.save_model(episode)
        # if episode % save_interval == 0:
        #     if not os.path.exists('models'):
        #         os.makedirs('models')
        #     agent.q_net.save_weights(f'models/q_net_{episode}.h5')
        #     agent.param_net.save_weights(f'models/param_net_{episode}.h5')

# 初始化环境和代理
env = SimpleEnv()
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
agent = PDQNAgent(env.state_dim, env.action_dim, env.param_dim, log_dir=log_dir)

# 训练 PDQN 代理
train_pdqn(agent, env, episodes=20000, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, save_interval=100)

