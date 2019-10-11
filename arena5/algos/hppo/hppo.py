import random, os

from arena5.algos.hppo.ppo1_mod import PPO1
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common import tf_util, zipsame
from stable_baselines.common.distributions import DiagGaussianProbabilityDistribution
from tensorflow.keras.layers import Lambda, Input, LSTM, Dense, Reshape, Flatten, multiply, RepeatVector, Permute
from tensorflow.keras.initializers import Orthogonal

ORTHO_01 = Orthogonal(0.01)

class HPPOPolicy():

	def __init__(self, env, policy_comm):
		self.env = env
		self.comm = policy_comm

		state_size = env.observation_space
		action_size_behavior = env.action_space
		self.b_agent_attack = BehaviorModel(state_size, action_size_behavior,  label='attack')
    	self.b_agent_evade = BehaviorModel(state_size, action_size_behavior, label='evade')
    	self.b_agent_transit = BehaviorModel(state_size, action_size_behavior,  label='transit')

    	# Define meta agent
    	self.m_agent = MetaAgent(state_size,[self.b_agent_attack, self.b_agent_evade, self.b_agent_transit])




	def run(self, num_steps, data_dir, policy_record=None):
		local_steps = int(num_steps / self.comm.Get_size())
		steps =0



		while True:

			training_state =       {"meta": [], "attack": [], "evade": [], "transit": []}
	        training_action =      {"meta": [], "attack": [], "evade": [], "transit": []}
	        training_reward =      {"meta": [], "attack": [], "evade": [], "transit": []}
	        training_next_state =  {"meta": [], "attack": [], "evade": [], "transit": []}
	        training_done =        {"meta": [], "attack": [], "evade": [], "transit": []}
	        training_reward_sum_combined = 0 # keep track of combined reward over all episodes between training


			done = False
			while not done:
				action, distribution, label, attn, selected = self.m_agent.get_action(state)
                next_state, reward, done, info = self.env.step(action)

				training_state["meta"].append(state)
                training_action["meta"].append(selected)
                training_reward["meta"].append(reward["combined"])
                training_next_state["meta"].append(next_state)
                training_done["meta"].append(done)
                # attack/evade/transit

                training_state[label].append(state)
                training_action[label].append(action)
                training_reward[label].append(reward[label])
                training_next_state[label].append(next_state)
                training_done[label].append(done)

				state = next_state




def behavior_actor_critic(input_shape_vec, act_output_shape, learn_rate=[0.001, 0.001], trainable=True,label=""):

    inp = Input(shape=input_shape_vec)  # [5,6,3]
    trunk_x = Reshape([input_shape_vec[0], input_shape_vec[1] * 4])(inp)
    trunk_x = LSTM(128)(trunk_x)

    # TODO concatenate on the static vector input and pass to dense layer
    act_x = Dense(act_output_shape, activation='tanh')(trunk_x)

    crt_x = Dense(1,name=label)(trunk_x)

    beh_model = Model(inp, [act_x, crt_x])

    for layer in beh_model.layers:
        layer.trainable = trainable

    beh_model.summary()

    beh_model.compile(optimizer=AdamOptimizer(learning_rate=learn_rate[0]), loss='mse')

    return beh_model


def single_model_meta_actor_critic(input_shape_vec, behavior_primitive_mdls, learn_rate=[0.001, 0.001], trainable=True):

    inp = Input(shape=input_shape_vec)  # [5,6,3]
    meta_lyr = Lambda(lambda x:  ned_to_ripCoords_tf(x, 4000))(inp)

    trunk_x = Reshape([input_shape_vec[0], input_shape_vec[1] * 4])(trunk_x)
    trunk_x = LSTM(128)(trunk_x)

    # Collect action predictions from all of the primitives
    #behavior predictions
    behavior_predictions = [p.model(meta_lyr)[0] for p in behavior_primitive_mdls]
    pred = tf.keras.layers.Lambda(lambda x: tf.keras.backend.stack(x, axis=-2))(behavior_predictions)

    # TODO concatenate on the static vector input and pass to dense layer
    act_x = Dense(len(behavior_primitive_mdls), activation='softmax')(trunk_x)

    combined_action = tf.keras.layers.Dot(axes=0,name="ahat")([act_x, pred])

    crt_x = Dense(1,name="value")(trunk_x)
    argmax_label = tf.keras.layers.Lambda(lambda x: tf.keras.backend.argmax(x, axis=-1),name="argmax")(act_x)

	#####
	# Adding mean and log standard deviation to the Model
	# from this point forward we are using tensorflow
	#####




    #act_model = Model(inp, [soft,crt_x,argmax_label]+behavior_predictions)


    #for layer in act_model.layers:
    #    layer.trainable = trainable


    #act_model.summary()




    #loss_dict = {'ahat': 'mse'}.update({b.label:'mse' for b in behavior_primitive_mdls})
    #act_model.compile(optimizer=AdamOptimizer(learning_rate=learn_rate[0]), loss=loss_dict)


    return act_model


def tf_continuous(num_actions,):
	    action_ph = tf.placeholder(tf.float32,[None, num_actions], name="actions_ph")
	    means = Dense(num_actions, activation="linear", kernel_initializer=ORTHO_01)(x)
	    vlogstd = tf.get_variable(name='pi/vlogstd', shape=[1, num_actions], initializer=tf.zeros_initializer())
	    means_and_logstd = tf.concat([means, means*0.0 + vlogstd], 1)
	    distribution = DiagGaussianProbabilityDistribution(means_and_logstd)


class BehaviorModel:
    def __init__(self,input_shape,output_shape,label='behavior'):
        self.label = label
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = behavior_actor_critic(self.input_shape,self.output_shape,label=self.label)


class MetaAgent:
    def __init__(self,input_shape,behavior_primitive_mdls):
        self.behavior_primitive_mdls = behavior_primitive_mdls
        self.input_shape = input_shape
        self.model = single_model_meta_actor_critic(self.input_shape,self.behavior_primitive_mdls)

    def get_action(self,state):
        s = np.expand_dims(state,axis=0)
        act =  self.model.predict(s)
        act = np.squeeze(act)
        return act
