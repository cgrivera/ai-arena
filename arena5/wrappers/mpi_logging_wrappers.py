

# gym wrapper which maintains episode results across all workers
# and adds them to a policy record
class MPISynchronizedPRUpdater():

    def __init__(self, proxy_env, policy_comms, policy_record=None, save_every=20, sum_over=False, channel="main"):

        self.multiagent = proxy_env.is_multiagent
        if self.multiagent:
            self.num_agents = len(proxy_env.entity_idxs)
            self.observation_spaces = proxy_env.observation_spaces
            self.action_spaces = proxy_env.action_spaces
        else:
            self.num_agents = 1
            self.observation_space = proxy_env.observation_space
            self.action_space = proxy_env.action_space

        self.sum_over = sum_over

        self.env = proxy_env
        self.comm = policy_comms
        self.ep_len = ([0.0]*self.num_agents if not self.sum_over else [0.0])
        self.ep_rew = ([0.0]*self.num_agents if not self.sum_over else [0.0])
        self.policy_record = policy_record
        self.channel = channel

        self.save_every = save_every
        self.episodes_until_save = self.save_every

    def reset(self):
        self.ep_len = ([0.0]*self.num_agents if not self.sum_over else [0.0])
        self.ep_rew = ([0.0]*self.num_agents if not self.sum_over else [0.0])
        return self.env.reset()

    def step(self, action):
        s, r, d, info = self.env.step(action)

        #synchronize results ---------------------------------
        if self.multiagent:
            if self.sum_over:
                self.ep_len[0] += 1.0
                self.ep_rew[0] += sum(r)
            else:
                for i in range(len(r)):
                    self.ep_len[i] += 1.0 
                    self.ep_rew[i] += r[i]
        else:
            self.ep_len[0] += 1.0
            self.ep_rew[0] += r

        if d:
            lrlocal = (self.ep_len, self.ep_rew)
        else:
            lrlocal = ([],[])

        listoflrpairs = self.comm.allgather(lrlocal)
        lens, rews = map(self.flatten_lists, zip(*listoflrpairs))

        if self.policy_record is not None:
            for idx in range(len(lens)):
                self.policy_record.add_result(rews[idx], lens[idx], channel=self.channel)

            self.episodes_until_save -= len(lens)
            if self.episodes_until_save <= 0:
                self.policy_record.save()
                self.episodes_until_save = self.save_every

        # -----------------------------------------------------

        return s, r, d, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    # utility from stable baselines
    def flatten_lists(self, listoflists):
        """
        Flatten a python list of list
        :param listoflists: (list(list))
        :return: (list)
        """
        return [el for list_ in listoflists for el in list_]