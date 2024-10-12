import gym
import random
import numpy as np
# import os
# os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/partha/anaconda/lib"
# os.environ["CUDA_HOME"] = "/usr"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
tf.disable_v2_behavior()
# tf.logging.set_verbosity(tf.logging.ERROR)
# tf.config.optimizer.set_jit(True)
# config = tf.ConfigProto()
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
# tf.config.threading.set_intra_op_parallelism_threads(24)
# sess = tf.Session(config=config)
from collections import deque

SAMPLE_NUMBER = 0
REPLAY_MEMORY_SIZE = 10000  # number of tuples in experience replay
total_steps = 0
EPSILON = 1.0  # epsilon of epsilon-greedy exploration
MIN_EPSILON = 0.1
MAX_EPSILON = 1.0
EPSILON_DECAY = 0.0001  # exponential decay multiplier for epsilon
episode_length = 0
Maximum_global_steps = 60000
# target_Weights = dict.fromkeys(['W1', 'b1', 'W2', 'b2', 'Wo', 'b0'])


class DQN:
    HIDDEN1_SIZE = 128  # size of hidden layer 1
    HIDDEN2_SIZE = 128  # size of hidden layer 2

    EPISODES_NUM = 2000  # number of episodes to train on. Ideally shouldn't take longer than 2000
    MAX_STEPS = 200  # maximum number of steps in an episode
    LEARNING_RATE = 0.0001  # learning rate and other parameters for SGD/RMSProp/Adam
    MINIBATCH_SIZE = 25  # size of mini-batch sampled from the experience replay
    DISCOUNT_FACTOR = 0.9  # MDP's gamma
    TARGET_UPDATE_FREQ = 10  # number of steps (not episodes) after which to update the target networks

    LOG_DIR = './LOGS'  # directory wherein logging takes place

    def __init__(self, env):
        self.env = gym.make(env)
        assert len(self.env.observation_space.shape) == 1
        self.input_size = self.env.observation_space.shape[0]  # In case of cartpole, 4 state features
        self.output_size = self.env.action_space.n  # In case of cartpole, 2 actions (right/left)
        self.REPLAY_MEMORY = deque(maxlen=REPLAY_MEMORY_SIZE)

    # Create the Q-network

    def initialize_network(self):
        self.State = tf.placeholder(tf.float32, [None, self.input_size])
        self.Action = tf.placeholder(tf.int32, [None, ])
        OHot_action = tf.one_hot(self.Action, self.output_size)
        self.Reward = tf.placeholder(tf.float32, [None, ])
        self.Target = tf.placeholder(tf.float32, [None, ])
        self.Next_state = tf.placeholder(tf.float32, [None, self.input_size])

        with tf.variable_scope('Q'):
            W_1 = tf.Variable(
                tf.truncated_normal([self.input_size, self.HIDDEN1_SIZE],
                                    stddev=0.01), name='W_1')
            b_1 = tf.Variable(tf.zeros(self.HIDDEN1_SIZE), name='b_1')

            W_2 = tf.Variable(
                tf.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE],
                                    stddev=0.01), name='W_2')
            b_2 = tf.Variable(tf.zeros(self.HIDDEN2_SIZE), name='b_2')

            W_n = tf.Variable(
                tf.truncated_normal([self.HIDDEN2_SIZE, self.output_size],
                                    stddev=0.01), name='W_n')
            b_n = tf.Variable(tf.zeros(self.output_size), name='b_n')

            self.Q = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(self.State, W_1) + b_1), W_2) + b_2),
                               W_n) + b_n
            self.Weights = {"W1": W_1, "b1": b_1, "W2": W_2, "b2": b_2, "Wo": W_n, "bo": b_n}

            self.Target_Weights = self.Weights
            self.Q_next = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(self.Next_state,
                                                                              self.Target_Weights["W1"])
                                                                    + self.Target_Weights["b1"]),
                                                         self.Target_Weights["W2"]) + self.Target_Weights["b2"]),
                                                         self.Target_Weights["Wo"]) + self.Target_Weights["bo"]

            self.Q_a = tf.reduce_sum(tf.multiply(self.Q, OHot_action), axis=1)
            # self.loss = tf.reduce_mean((self.Q_a - self.Target) ** 2)

            self.loss = tf.nn.l2_loss((self.Q_a - self.Target))
            optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


    def train(self, episodes_num=EPISODES_NUM):

        # Initialize summary for TensorBoard
        summary_writer = tf.summary.FileWriter(self.LOG_DIR)
        summary_writer.flush()
        # summary_writer.close()
        summary = tf.Summary()
        # Alternatively, you could use animated real-time plots from matplotlib
        # (https://stackoverflow.com/a/24228275/3284912)

        # Initialize the TF session
        self.session = tf.Session()
        # self.session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.session.run(tf.global_variables_initializer())

        global SAMPLE_NUMBER
        global REPLAY_MEMORY_SIZE
        global total_steps
        global EPSILON
        global MINIMUM_EPSILON
        global EPSILON_DECAY
        global episode_length
        global Maximum_global_steps
        state = self.env.reset()
        while SAMPLE_NUMBER < REPLAY_MEMORY_SIZE:
            action = np.random.randint(0, self.output_size)
            next_state, reward, done, _ = self.env.step(action)
            if done:
                next_state = np.zeros(self.input_size)
                self.REPLAY_MEMORY.append((state, action, reward, next_state))
                state = self.env.reset()
            else:
                self.REPLAY_MEMORY.append((state, action, reward, next_state))
                state = next_state
                SAMPLE_NUMBER += 1

        for episode in range(episodes_num):
            state = self.env.reset()
            tf.dtypes.cast(state, tf.float32)
            episode_reward = 0
            episode_length = 0
            loss_per_episode = 0
            # Random_action_count = 0
            # Greedy_action_count = 0
            while episode_length <= self.MAX_STEPS:

                if np.random.uniform() > EPSILON:
                    # forward feed the observation and get q value for every actions
                    actions_value = self.session.run(self.Q,
                                                     feed_dict={self.State: state.reshape((1, *state.shape))})
                    action = np.argmax(actions_value)
                    # Greedy_action_count += 1
                else:
                    action = np.random.randint(0, self.output_size)
                    # Random_action_count += 1
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                episode_length += 1
                if done:
                    next_state = np.zeros(self.input_size)
                    self.REPLAY_MEMORY.append((state, action, reward, next_state))
                    episode_length = self.MAX_STEPS + 1
                self.REPLAY_MEMORY.append((state, action, reward, next_state))
                state = next_state
                sample_indices = np.random.choice(np.arange(len(self.REPLAY_MEMORY)),
                                                  size=self.MINIBATCH_SIZE, replace=False)

                batch = [self.REPLAY_MEMORY[ind] for ind in sample_indices]
                batch_state = np.array([each[0] for each in batch])
                batch_action = np.array([each[1] for each in batch])
                batch_reward = np.array([each[2] for each in batch])
                batch_next_state = np.array([each[3] for each in batch])

                if total_steps % self.TARGET_UPDATE_FREQ == 0:
                    self.Target_Weights = self.session.run(self.Weights)

                total_steps = self.session.run(self.global_step)
                Q_t = self.session.run(self.Q_next, feed_dict={self.Next_state: batch_next_state,
                                                                    self.Reward: batch_reward})
                episode_ends = (batch_next_state == np.zeros(batch_state[0].shape)).all(axis=1)
                Q_t[episode_ends] = (0, 0)
                Q_target = batch_reward + self.DISCOUNT_FACTOR * np.max(Q_t, axis=1)

                Loss, _ = self.session.run([self.loss, self.train_op],
                                           feed_dict={self.State: batch_state, self.Action: batch_action,
                                                      self.Target: Q_target})

                loss_per_episode += Loss

            loss_per_episode = loss_per_episode / episode_reward
            # print(EPSILON)
            # print("Epsilon = %f, Random action count = %d, Greedy action count= %d" % (
            #     EPSILON, Random_action_count, episode_length-Random_action_count))
            EPSILON = round(MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-EPSILON_DECAY * total_steps), 5)
            print("Training: Episode = %d, Reward = %d, Global step = %d, Average loss  = %f, Epsilon = %f" % (
             episode, episode_reward, total_steps, loss_per_episode, EPSILON))
            print("####################################******************###################################")
            summary.value.add(tag="episode reward", simple_value=episode_reward)
            summary.value.add(tag="Average loss per episode", simple_value=loss_per_episode)
            summary_writer.add_summary(summary, episode)
            if total_steps > Maximum_global_steps:
                break
        summary_writer.close()

    # Simple function to visually 'test' a policy
    def playPolicy(self):

        done = False
        steps = 0
        state = self.env.reset()

        # we assume the CartPole task to be solved if the pole remains upright for 200 steps
        while not done and steps < 200:
            # self.env.render()
            q_values = self.session.run(self.Q, feed_dict={self.State: state.reshape((1, *state.shape))})
            action = q_values.argmax()
            state, _, done, _ = self.env.step(action)
            steps += 1

        return steps


if __name__ == '__main__':

    # Create and initialize the model
    dqn = DQN('CartPole-v0')
    dqn.initialize_network()

    print("\nStarting training...\n")
    dqn.train()
    print("\nFinished training...\nCheck out some demonstrations\n")

    # Visualize the learned behaviour for a few episodes
    results = []
    for i in range(100):
        episode_length = dqn.playPolicy()
        print("Episode = %d, Test steps = %d" % (i+1, episode_length))
        results.append(episode_length)
    print("Mean steps = ", sum(results) / len(results))

    print("\nFinished.")
    print("\nCiao, and hasta la vista...\n")
