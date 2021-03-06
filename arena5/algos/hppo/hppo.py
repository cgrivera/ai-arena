# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import random
import os
import numpy as np
from arena5.algos.hppo.utils import ned_to_ripCoords_tf
from arena5.algos.hppo.GAE import GAE
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common import tf_util, zipsame
from stable_baselines.common.distributions import DiagGaussianProbabilityDistribution
from tensorflow.keras.layers import Lambda, Input, LSTM, Dense, Reshape, Flatten, multiply, RepeatVector, Permute
from tensorflow.keras.initializers import Orthogonal
from stable_baselines.common import Dataset

from tensorflow.keras import backend as K
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common import tf_util, zipsame
from tensorflow.keras.backend import set_session
from stable_baselines import logger

ORTHO_01 = Orthogonal(0.01)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class HPPOPolicy():

    def __init__(self, env, policy_comm, **kwargs):

        # Pull params out of kwargs
        self.params = kwargs['params']

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras

        self.env = env
        self.comm = policy_comm

        # Setup tensorboard logger
        logger.configure(self.params.logDir, format_strs=['tensorboard'])

        state_size = env.observation_space.shape
        action_size_behavior = env.action_space.shape
        self.b_agent_attack = BehaviorModel(state_size, action_size_behavior,  self.comm, label='attack')
        self.b_agent_evade = BehaviorModel(state_size, action_size_behavior, self.comm, label='evade')
        self.b_agent_transit = BehaviorModel(state_size, action_size_behavior,  self.comm, label='transit')

        # Define meta agent
        self.m_agent = MetaAgent(state_size, [self.b_agent_attack, self.b_agent_evade, self.b_agent_transit], self.comm)

        # constants
        self.discount_factor = 0.99

    def run(self, num_steps, data_dir, policy_record=None):
        local_steps = int(num_steps / self.comm.Get_size())

        steps = 0

        # TODO: Add alpha annealing over num_steps

        while True:

            # sync weights
            self.b_agent_attack.sync_weights()
            self.b_agent_evade.sync_weights()
            self.b_agent_transit.sync_weights()
            self.m_agent.sync_weights()

            # create placeholders to store experience that we gather
            training_state = {"meta": [], "attack": [], "evade": [], "transit": []}
            training_action = {"meta": [], "attack": [], "evade": [], "transit": []}
            training_reward = {"meta": [], "attack": [], "evade": [], "transit": []}
            training_next_state = {"meta": [], "attack": [], "evade": [], "transit": []}
            training_done = {"meta": [], "attack": [], "evade": [], "transit": []}
            training_reward_sum_combined = 0  # keep track of combined reward over all episodes between training

            state = self.env.reset()
            reward_sum = {}
            done = False
            while not done:
                complete_action, distribution, beh_actions, label = self.m_agent.get_action(state)

                next_state, reward, done, info = self.env.step(complete_action, dst=distribution, label=label)

                # Aggregate reward throughout the episode
                if not reward_sum:
                    reward_sum = reward
                else:
                    reward_sum = {k: reward_sum[k] + reward[k] for (k, v) in reward.items()}
                training_reward_sum_combined += reward["combined"]

                training_state["meta"].append(state)
                training_action["meta"].append(distribution)
                training_reward["meta"].append(reward["combined"])
                training_next_state["meta"].append(next_state)
                training_done["meta"].append(done)

                for idx, label in enumerate(['attack', 'evade', 'transit']):
                    training_state[label].append(state)
                    training_action[label].append(beh_actions[idx])
                    training_reward[label].append(reward[label])
                    training_next_state[label].append(next_state)
                    training_done[label].append(done)

                state = next_state

            # #now we have batches of data: compute the values and advantages
            # training_value = {"meta": None, "attack": None, "evade": None, "transit": None}
            # training_advantages = {"meta": None, "attack": None, "evade": None, "transit": None}

            # log tensorboard
            {logger.logkv(k, v) for (k, v) in reward_sum.items()}
            logger.dumpkvs()

            # vcompute advantages and values
            models = [self.b_agent_attack, self.b_agent_evade, self.b_agent_transit, self.m_agent]
            for model in models:

                network = model.label

                states = training_state[network]
                actions = training_action[network]
                reward = training_reward[network]
                next_states = training_next_state[network]
                done = training_done[network]

                # Convert done bools to ints and invert
                done_int = np.invert(done).astype(np.int)

                # Generalized advantage estimation (gets advantages to train on and value estimates)
                target, advantages = GAE(states, actions, reward, next_states, done_int, model.sample_value, T=128, y=0.99, lam=0.95, use_Q=False)

                # train this model
                dataset = Dataset(dict(ob=np.asarray(states), ac=np.asarray(actions), atarg=np.asarray(advantages), vtarg=np.asarray(target)), shuffle=True)

                for k in range(4):
                    for i, batch in enumerate(dataset.iterate_once(len(states))):
                        model.train(batch["ob"], batch["ac"], batch["vtarg"], batch["atarg"], 1.0)

            print('FINISHED TRAINING EPISODE')


def general_actor_critic(input_shape_vec, act_output_shape, comm, learn_rate=[0.001, 0.001], trainable=True, label=""):

    sess = K.get_session()
    np.random.seed(0)
    tf.set_random_seed(0)

    # network 1 (new policy)
    with tf.variable_scope(label+"_pi_new", reuse=False):
        inp = Input(shape=input_shape_vec)  # [5,6,3]
        # rc_lyr = Lambda(lambda x:  ned_to_ripCoords_tf(x, 4000))(inp)
        trunk_x = Reshape([input_shape_vec[0], input_shape_vec[1] * 3])(inp)
        trunk_x = LSTM(128)(trunk_x)
        dist, sample_action_op, action_ph, value_output = ppo_continuous(3, trunk_x)

    # network 2 (old policy)
    with tf.variable_scope(label+"_pi_old", reuse=False):
        inp_old = Input(shape=input_shape_vec)  # [5,6,3]
        # rc_lyr = Lambda(lambda x:  ned_to_ripCoords_tf(x, 4000))(inp_old)
        trunk_x = Reshape([input_shape_vec[0], input_shape_vec[1] * 3])(inp_old)
        trunk_x = LSTM(128)(trunk_x)
        dist_old, sample_action_op_old, action_ph_old, value_output_old = ppo_continuous(3, trunk_x)

    # additional placeholders
    adv_ph = tf.placeholder(tf.float32, [None], name="advantages_ph")
    alpha_ph = tf.placeholder(tf.float32, shape=(), name="alpha_ph")
    vtarg = tf.placeholder(tf.float32, [None])  # target value placeholder

    # loss
    loss = ppo_continuous_loss(dist, dist_old, value_output, action_ph, alpha_ph, adv_ph, vtarg)

    # gradient
    with tf.variable_scope("grad", reuse=False):
        gradient = tf_util.flatgrad(loss, tf_util.get_trainable_vars(label+"_pi_new"))
        adam = MpiAdam(tf_util.get_trainable_vars(label+"_pi_new"), epsilon=0.00001, sess=sess, comm=comm)

    # method for sync'ing the two policies
    assign_old_eq_new = tf_util.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                                          zipsame(tf_util.get_globals_vars(label+"_pi_old"), tf_util.get_globals_vars(label+"_pi_new"))])

    # initialize all the things
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # methods for interacting with this model

    def sync_weights():
        assign_old_eq_new(sess=sess)

    def sample_action(states, logstd_override=None):
        a = sess.run(sample_action_op, feed_dict={inp: states})
        return a

    def sample_value(states):
        v = sess.run(value_output, feed_dict={inp: states})
        return v

    def train(states, actions, vtarget, advs, alpha):
        alpha = max(alpha, 0.0)
        adam_lr = learn_rate[0]

        g = sess.run([gradient], feed_dict={
                inp: states,
                inp_old: states,
                action_ph: actions,
                adv_ph: advs,
                alpha_ph: alpha,
                vtarg: vtarget
            })

        adam.update(g[0], adam_lr * alpha)

    # initial sync
    adam.sync()
    sync_weights()

    return sync_weights, sample_action, sample_value, train


def ppo_continuous(num_actions, previous_layer):

    # act distribution
    action_ph = tf.placeholder(tf.float32, [None, num_actions], name="actions_ph")
    means = Dense(num_actions, activation="linear", kernel_initializer=ORTHO_01)(previous_layer)
    vlogstd = tf.get_variable(name='pi/vlogstd', shape=[1, num_actions], initializer=tf.zeros_initializer())
    means_and_logstd = tf.concat([means, means*0.0 + vlogstd], 1)
    distribution = DiagGaussianProbabilityDistribution(means_and_logstd)

    # sample op
    sample_action_op = distribution.sample()

    # value
    value_output = Dense(1)(previous_layer)

    return distribution, sample_action_op, action_ph, value_output


def ppo_continuous_loss(new_dist, old_dist, value_output, actions_ph, alpha_ph, adv_ph, val_ph, clipping_epsilon=0.2):

    ratio = tf.exp(new_dist.logp(actions_ph) - old_dist.logp(actions_ph))
    epsilon = clipping_epsilon * alpha_ph
    surr1 = ratio * adv_ph
    surr2 = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv_ph
    ploss = - tf.reduce_mean(tf.minimum(surr1, surr2))
    vloss = tf.reduce_mean(tf.square(value_output - val_ph))
    loss = ploss + vloss

    return loss


class BehaviorModel:
    def __init__(self, input_shape, output_shape, comm, label='behavior'):
        self.label = label
        self.input_shape = input_shape
        self.output_shape = output_shape

        # These are all methods!  Use them to interact with the ppo models.
        self.sync_weights, self.sample_action, self.sample_value, self.train = general_actor_critic(self.input_shape, self.output_shape, comm, label=self.label)

    def get_action(self, state):
        action = self.sample_action(np.asarray([state]))[0]  # batch dimension 1
        return action


class MetaAgent:
    def __init__(self, input_shape, behavior_primitive_mdls, comm):
        self.label = "meta"
        self.behavior_primitive_mdls = behavior_primitive_mdls
        self.input_shape = input_shape
        self.output_shape = len(behavior_primitive_mdls)
        self.sync_weights, self.sample_action, self.sample_value, self.train = general_actor_critic(self.input_shape, self.output_shape, comm, label=self.label)

    def get_action(self, state):
        meta_action = self.sample_action(np.asarray([state]))[0]
        beh_actions = np.array([bm.get_action(state) for bm in self.behavior_primitive_mdls])

        meta_action_softmax = np.exp(meta_action)/sum(np.exp(meta_action))

        # Get the argmax of the softmax
        meta_action_argmax = np.argmax(meta_action_softmax, axis=-1)

        # Get label from argmax
        label = self.behavior_primitive_mdls[meta_action_argmax].label

        # Should be doing a vectorized dot product
        complete_action = np.tensordot(beh_actions, np.expand_dims(meta_action_softmax, axis=0), axes=[0, 1])
        # complete_action = np.tensordot(meta_action_softmax, beh_actions, axes=[0, 1])

        return complete_action, meta_action_softmax, beh_actions, label
