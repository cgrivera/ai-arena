from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import os
import arena5.algos.masac.network_utils as core

from arena5.core.mpi_pytorch_utils import sync_weights, sync_grads


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, N, obs_dim, act_dim, size):
        self.obs_buf = []
        self.obs2_buf = []
        self.act_buf = []
        self.N = N
        for i in range(N):
            self.obs_buf.append( np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32) )
            self.obs2_buf.append( np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32) )
            self.act_buf.append( np.zeros(core.combined_shape(size, act_dim), dtype=np.float32) )
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        for i in range(self.N):
            self.obs_buf[i][self.ptr] = obs[i]
            self.obs2_buf[i][self.ptr] = next_obs[i]
            self.act_buf[i][self.ptr] = act[i]
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
                    obs=[torch.as_tensor(self.obs_buf[i][idxs], dtype=torch.float32) for i in range(self.N)],
                    obs2=[torch.as_tensor(self.obs2_buf[i][idxs], dtype=torch.float32) for i in range(self.N)],
                    act=[torch.as_tensor(self.act_buf[i][idxs], dtype=torch.float32) for i in range(self.N)],
                    rew=torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32),
                    done=torch.as_tensor(self.done_buf[idxs], dtype=torch.float32)
                )

        return batch #{k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



def masac(env_fn, comm, data_dir, policy_record=None, eval_mode=False, 
        actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):


    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    N = len(env.observation_spaces)
    obs_dim = env.observation_spaces[0].shape
    act_dim = env.action_spaces[0].shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_spaces[0].high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_spaces, env.action_spaces, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # if records exist, load weights
    if policy_record is not None:
        if os.path.exists(data_dir+"ac.pt"):
            ac.load_state_dict(torch.load(data_dir+"ac.pt"))
        if os.path.exists(data_dir+"ac_targ.pt"):
            ac_targ.load_state_dict(torch.load(data_dir+"ac_targ.pt"))

    # initial weight sync
    sync_weights(comm, ac.parameters())
    sync_weights(comm, ac_targ.parameters())

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(N, obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.current_pi(o2)
            total_action_logprob = sum(logp_a2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * total_action_logprob)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_losses_pi(data):
        o = data['obs']
        pi, logp_pi = ac.current_pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        losses_pi = [(alpha * logp_pi[i] - q_pi).mean() for i in range(N)]
        loss_pi = sum(losses_pi)

        return loss_pi

    def compute_losses_pi_ddpg(data):
        o = data['obs']
        pi, logp_pi = ac.current_pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        losses_pi = (-q_pi).mean()
        return losses_pi

    # Set up optimizers for policy and q-function
    pi_optimizers = [Adam(ac.pis[i].parameters(), lr=lr) for i in range(N)]
    q_optimizer = Adam(q_params, lr=lr)


    def update(data):

        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        sync_grads(comm, q_params)
        q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for each pi.
        [opt.zero_grad() for opt in pi_optimizers]
        loss_pi = compute_losses_pi(data)
        loss_pi.backward()
        [sync_grads(comm, ac.pis[i].parameters()) for i in range(N)]
        [opt.step() for opt in pi_optimizers]

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True


        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)


        # sync weights
        sync_weights(comm, q_params)
        [sync_weights(comm, pi.parameters()) for pi in ac.pis]
        sync_weights(comm, ac_targ.parameters())

        # save weights
        if policy_record is not None:
            torch.save(ac.state_dict(), data_dir+"ac.pt")
            torch.save(ac_targ.state_dict(), data_dir+"ac_targ.pt")


    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

    # def test_agent():
    #     for j in range(num_test_episodes):
    #         o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
    #         while not(d or (ep_len == max_ep_len)):
    #             # Take deterministic actions at test time 
    #             o, r, d, _ = test_env.step(get_action(o, True))
    #             ep_ret += r
    #             ep_len += 1

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if eval_mode:
            a = get_action(o, True)
        else:
            if t > start_steps:
                a = get_action(o)
            else:
                a = [asp.sample() for asp in env.action_spaces]

        # Step the env
        o2, rs, d, _ = env.step(a)
        r = sum(rs)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d:
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling (training only)
        if not eval_mode:
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            # test_agent()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)