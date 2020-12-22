# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import os
import arena5.algos.maddpg.network_utils as core
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



def maddpg(env_fn, comm, data_dir, policy_record=None, eval_mode=False,
        common_actor=False,
        actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, 
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """
    Deep Deterministic Policy Gradient (DDPG)
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:
            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to DDPG.
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
        pi_lr (float): Learning rate for policy.
        q_lr (float): Learning rate for Q-networks.
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
        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)
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
    ac = actor_critic(env.observation_spaces, env.action_spaces, common_actor, **ac_kwargs)
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

    # for param in ac.named_parameters():
    #     print(param)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(N, obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, [ac_targ.pis[i](o2[i]) for i in range(N)])
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, [ac.pis[i](o[i]) for i in range(N)])
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizers = [Adam(p.parameters(), lr=pi_lr) for p in ac.unique_pis]
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)


    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        sync_grads(comm, ac.q.parameters())
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        [opt.zero_grad() for opt in pi_optimizers]
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        [sync_grads(comm, p.parameters()) for p in ac.unique_pis]
        [opt.step() for opt in pi_optimizers]

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)


        # sync weights
        sync_weights(comm, ac.q.parameters())
        [sync_weights(comm, pi.parameters()) for pi in ac.unique_pis]
        sync_weights(comm, ac_targ.parameters())

        # save weights
        if policy_record is not None:
            torch.save(ac.state_dict(), data_dir+"ac.pt")
            torch.save(ac_targ.state_dict(), data_dir+"ac_targ.pt")


    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if eval_mode:
            a = get_action(o, 0.0)
        else:
            if t > start_steps:
                a = get_action(o, act_noise)
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
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if not eval_mode:
            if t >= update_after and t % update_every == 0:
                for _ in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs):
            #     logger.save_state({'env': env}, None)

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
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ddpg(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs)