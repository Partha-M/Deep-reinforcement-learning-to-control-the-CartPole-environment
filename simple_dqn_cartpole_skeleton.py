import gym
import random

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

SAMPLE_NUMBER = 0
target_W1 = None
target_b1 = None
target_W2 = None
target_b2 = None
target_Wn = None
target_bn = None
total_steps = 0


class DQN:
    REPLAY_MEMORY_SIZE = 10000  # number of tuples in experience replay
    EPSILON = 0.5  # epsilon of epsilon-greedy exploration
    EPSILON_DECAY = 0.99  # exponential decay multiplier for epsilon
    HIDDEN1_SIZE = 128  # size of hidden layer 1
    HIDDEN2_SIZE = 128  # size of hidden layer 2

    EPISODES_NUM = 2000  # number of episodes to train on. Ideally shouldn't take longer than 2000
    MAX_STEPS = 200  # maximum number of steps in an episode
    LEARNING_RATE = 0.0001  # learning rate and other parameters for SGD/RMSProp/Adam
    MINIBATCH_SIZE = 10  # size of mini-batch sampled from the experience replay
    DISCOUNT_FACTOR = 0.9  # MDP's gamma
    TARGET_UPDATE_FREQ = 100  # number of steps (not episodes) after which to update the target networks

    LOG_DIR = './logs'  # directory wherein logging takes place

    # Create and initialize the environment
    def __init__(self, env):
        self.env = gym.make(env)
        assert len(self.env.observation_space.shape) == 1
        self.input_size = self.env.observation_space.shape[0]  # In case of cartpole, 4 state features
        self.output_size = self.env.action_space.n  # In case of cartpole, 2 actions (right/left)

    # Create the Q-network
    def initialize_network(self):

        # placeholder for the state-space input to the q-network

        self.State = tf.placeholder(tf.float32, [None, self.input_size], name='s')
        self.Action = tf.placeholder(tf.int32, [None, ], name='a')
        self.Reward = tf.placeholder(tf.float32, [None, ], name='r')
        self.Next_State = tf.placeholder(tf.float32, [None, self.input_size], name='n_s')

        ############################################################
        # Design your q-network here.
        #
        # Add hidden layers and the output layer. For instance:
        #
        # with tf.name_scope('output'):
        # W_n = tf.Variable(
        # 			 tf.truncated_normal([self.HIDDEN_n-1_SIZE, self.output_size],
        # 			 stddev=0.01), name='W_n')
        # 	b_n = tf.Variable(tf.zeros(self.output_size), name='b_n')
        # 	self.Q = tf.matmul(h_n-1, W_n) + b_n
        #
        #############################################################

        with tf.name_scope('hidden_1'):
            W_1 = tf.Variable(
                tf.truncated_normal([self.input_size, self.HIDDEN1_SIZE],
                                    stddev=0.01), name='W_1')
            b_1 = tf.Variable(tf.zeros(self.HIDDEN1_SIZE), name='b_1')
            h_1 = tf.nn.relu(tf.matmul(self.State, W_1) + b_1)

        with tf.name_scope('hidden_2'):
            W_2 = tf.Variable(
                tf.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE],
                                    stddev=0.01), name='W_2')
            b_2 = tf.Variable(tf.zeros(self.HIDDEN2_SIZE), name='b_2')
            h_2 = tf.nn.relu(tf.matmul(h_1, W_2) + b_2)

        with tf.name_scope('output'):
            W_n = tf.Variable(
                tf.truncated_normal([self.HIDDEN2_SIZE, self.output_size],
                                    stddev=0.01), name='W_n')
            b_n = tf.Variable(tf.zeros(self.output_size), name='b_n')
            self.Q = tf.matmul(h_2, W_n) + b_n

        ############################################################
        # Next, compute the loss.
        #
        # First, compute the q-values. Note that you need to calculate these
        # for the actions in the (s,a,s',r) tuples from the experience replay's minibatch
        #
        # Next, compute the l2 loss between these estimated q-values and
        # the target (which is computed using the frozen target network)
        #
        ############################################################
        v = tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(self.Next_State, target_W1) + target_b1), target_W2) + target_b2)
        self.target_value = self.Reward + gamma * np.max((tf.matmul(v, target_Wn) + target_bn), axis=1)

        if self.Action:
            self.Q_estimate = self.Q[1]
        else:
            self.Q_estimate = self.Q[0]

        self.loss = tf.reduce_mean((self.Q_estimate - self.target_value) ** 2)

        ############################################################
        # Finally, choose a gradient descent algorithm : SGD/RMSProp/Adam.
        #
        # For instance:
        # optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        # self.train_op = optimizer.minimize(self.loss, global_step=global_step)
        #
        ############################################################

        optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)

    ############################################################

    def train(self, episodes_num=EPISODES_NUM):

        # Initialize summary for TensorBoard
        summary_writer = tf.summary.FileWriter(self.LOG_DIR)
        summary = tf.Summary()
        # Alternatively, you could use animated real-time plots from matplotlib
        # (https://stackoverflow.com/a/24228275/3284912)

        # Initialize the TF session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        ############################################################
        # Initialize other variables (like the replay memory)
        ############################################################

        global SAMPLE_NUMBER
        global target_W1
        global target_b1
        global target_W2
        global target_b2
        global target_Wn
        global target_bn
        global total_steps

        ############################################################
        # Main training loop
        #
        # In each episode,
        # pick the action for the given state,
        # perform a 'step' in the environment to get the reward and next state,
        # update the replay buffer,
        # sample a random minibatch from the replay buffer,
        # perform Q-learning,
        # update the target network, if required.
        #
        #
        #
        # You'll need to write code in various places in the following skeleton
        #
        ############################################################

        for episode in range(episodes_num):

            state = self.env.reset()

            ############################################################
            # Episode-specific initializations go here.
            ############################################################
            episode_reward = 0
            ############################################################

            while True:

                ############################################################
                # Pick the next action using epsilon greedy and and execute it
                ############################################################

                if np.random.uniform() > EPSILON:
                    # forward feed the observation and get q value for every actions
                    actions_value = sess.run(self.Q, feed_dict={s: state})
                    action = np.argmax(actions_value)
                else:
                    action = np.random.randint(0, self.output_size)

                ############################################################
                # Step in the environment. Something like:
                # next_state, reward, done, _ = self.env.step(action)
                ############################################################

                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                ############################################################
                # Update the (limited) replay buffer.
                #
                # Note : when the replay buffer is full, you'll need to
                # remove an entry to accommodate a new one.
                ############################################################
                new_transition = np.hstack((state, [action, reward], next_state))
                # replace the old transition with new transition
                index = SAMPLE_NUMBER % REPLAY_MEMORY_SIZE
                REPLAY_MEMORY[index, :] = new_transition
                SAMPLE_NUMBER += 1

                ############################################################
                # Sample a random minibatch and perform Q-learning (fetch max Q at s')
                #
                # Remember, the target (r + gamma * max Q) is computed
                # with the help of the target network.
                # Compute this target and pass it to the network for computing
                # and minimizing the loss with the current estimates
                #
                ############################################################

                # Your code here

                ############################################################
                # Update target weights.
                #
                # Something along the lines of:
                # if total_steps % self.TARGET_UPDATE_FREQ == 0:
                # 	target_weights = self.session.run(self.weights)
                ############################################################
                if total_steps % self.TARGET_UPDATE_FREQ == 0:
                    target_W1 = self.session.run(W_1)
                    target_b1 = self.session.run(b_1)
                    target_W2 = self.session.run(W_2)
                    target_b2 = self.session.run(b_2)
                    target_Wn = self.session.run(W_n)
                    target_bn = self.session.run(b_n)

                sample_indices = np.random.choice(REPLAY_MEMORY_SIZE, MINIBATCH_SIZE)
                batch_memory = REPLAY_MEMORY[sample_indices, :]
                batch_state = batch_memory[:, :self.input_size]
                batch_action = batch_memory[:, self.input_size].astype(int)
                batch_reward = batch_memory[:, self.input_size + 1]
                batch_next_state = batch_memory[:, -self.input_size:]
                sess.run(self.train_op, {s: batch_state, a: batch_action, r: batch_reward, n_s: batch_next_state})
                total_steps += 1
                if EPSILON > MINIMUM_EPSILON:
                    EPSILON = EPSILON * EPSILON_DECAY

            ############################################################
            # Break out of the loop if the episode ends
            #
            # Something like:
            # if done or (episode_length == self.MAX_STEPS):
            # 	break
            #
            ############################################################

                if done:
                    break
                state = next_state

            ############################################################
            # Logging.
            #
            # Very important. This is what gives an idea of how good the current
            # experiment is, and if one should terminate and re-run with new parameters
            # The earlier you learn how to read and visualize experiment logs quickly,
            # the faster you'll be able to prototype and learn.
            #
            # Use any debugging information you think you need.
            # For instance :

            print("Training: Episode = %d, Length = %d, Global step = %d" % (episode, episode_length, total_steps))
            summary.value.add(tag="episode length", simple_value=episode_length)
            summary_writer.add_summary(summary, episode)

    # Simple function to visually 'test' a policy
    def playPolicy(self):

        done = False
        steps = 0
        state = self.env.reset()

        # we assume the CartPole task to be solved if the pole remains upright for 200 steps
        while not done and steps < 200:
            self.env.render()
            q_vals = self.session.run(self.Q, feed_dict={self.x: [state]})
            action = q_vals.argmax()
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
    for i in range(50):
        episode_length = dqn.playPolicy()
        print("Test steps = ", episode_length)
        results.append(episode_length)
    print("Mean steps = ", sum(results) / len(results))

    print("\nFinished.")
    print("\nCiao, and hasta la vista...\n")
