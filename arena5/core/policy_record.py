import pickle, os
from shutil import copy
import matplotlib.pyplot as plt

from arena5.core.plot_utils import *


def get_dir_for_policy(policy_id, log_comms_dir):
    return log_comms_dir + "policy_"+str(policy_id)+"/"


class RecordChannel():
    def __init__(self, data_dir, name="main", color="#eb0033", ylabel="Episodic Reward",
        windows=[20,50,100], alphas=[0.1,0.3,1.0]):

        self.data_dir = data_dir
        self.ep_results = []
        self.ep_lengths = []
        self.ep_cumlens = []
        self.color = color
        self.name = name
        self.ylabel = ylabel
        self.windows = windows
        self.alphas = alphas

    def save(self):
        data = [self.ep_results, self.ep_lengths, self.ep_cumlens]
        pickle.dump(data, open(self.data_dir+"policy_record_"+self.name+".p", "wb"))

        #save a plot also
        plot_policy_records([self], self.windows, self.alphas, 
            self.data_dir+"plot_"+self.name+".png", colors=[self.color], 
            episodic=False)
        plot_policy_records([self], self.windows, self.alphas, 
            self.data_dir+"plot_"+self.name+".png", colors=[self.color], 
            episodic=True)

    def load(self):
        path = self.data_dir+"policy_record_"+self.name+".p"
        if os.path.exists(path):
            data = pickle.load(open(path, "rb"))
            self.ep_results, self.ep_lengths, self.ep_cumlens = data


    def get_copy(self):
        rc = RecordChannel(self.data_dir, 
            name=self.name,
            color=self.color,
            ylabel=self.ylabel,
            windows=self.windows,
            alphas=self.alphas)

        rc.ep_results = self.ep_results[:]
        rc.ep_lengths = self.ep_lengths[:]
        rc.ep_cumlens = self.ep_cumlens[:]
        return rc



class PolicyRecord():

    def __init__(self, policy_id, log_comms_dir, plot_color="#eb0033"):

        self.plot_color = plot_color

        self.log_comms_dir = log_comms_dir
        self.data_dir = get_dir_for_policy(policy_id, log_comms_dir)

        self.channels = {
            "main":RecordChannel(self.data_dir, name="main")
        }

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        else:
            self.load()


    def add_result(self, total_reward, ep_len, channel="main"):

        if channel not in self.channels:
            self.add_channel(channel)

        ch = self.channels[channel]

        ch.ep_results.append(total_reward)
        ch.ep_lengths.append(ep_len)
        if len(ch.ep_cumlens) == 0:
            ch.ep_cumlens.append(ep_len)
        else:
            ch.ep_cumlens.append(ch.ep_cumlens[-1]+ep_len)


    def add_channel(self, channel_name, **kwargs):

        if channel_name not in self.channels:
            ch = RecordChannel(self.data_dir, name=channel_name, **kwargs)
            ch.load()
            self.channels[channel_name] = ch


    def save(self):
        for ch in self.channels:
            self.channels[ch].save()


    def load(self):
        for ch in self.channels:
            self.channels[ch].load()


    def fork(self, new_id):
        new_pr = PolicyRecord(new_id, self.log_comms_dir)

        for ch in self.channels:
            new_ch = self.channels[ch].get_copy()
            new_pr.channels[ch] = new_ch
        
        # copy files over from existing log dir
        for f in os.listdir(self.data_dir):

            # HACK: check for f not having any extension
            if "." not in f:
                continue

            copy(self.data_dir+f, new_pr.data_dir)

        return new_pr
