# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.

# Touchdown is a multiagent game in which both teams are trying to reach
# the opposing end of the play area

# if entities from opposite teams touch, they both respawn to their respective starting lines

# teams may independently have discrete or continuous action spaces, and vector or image observation spaces

import math, random, time
import numpy as np
import cv2

from arena5.core.utils import mpi_print
from gym.spaces import Box, Discrete


class TouchdownEnv():

    def __init__(self, team_size, blue_obs="image", blue_actions="discrete", red_obs="image", red_actions="discrete"):

        self.team_size = team_size
        self.blue_obs = blue_obs
        self.blue_actions = blue_actions
        self.red_obs = red_obs
        self.red_actions = red_actions

        self.start_pos = 0.9
        self.capture_radius = 0.1
        self.player_movement = 0.1

        # track player movement in a normalized space
        self.blue_team = []
        for t in range(self.team_size):
            self.blue_team.append([0.0, self.start_pos])

        self.red_team = []
        for t in range(self.team_size):
            self.red_team.append([0.0, -self.start_pos])

        self.all_players = self.blue_team + self.red_team

        # obs spaces
        self.observation_spaces = []
        for bp in self.blue_team:
            if self.blue_obs == "image":
                self.observation_spaces.append(Box(-10, 10, (84, 84, 1)))
            elif self.blue_obs == "vector":
                self.observation_spaces.append(Box(-10, 10, (len(self.all_players) * 2,)))
            else:
                raise ValueError

        for rp in self.red_team:
            if self.red_obs == "image":
                self.observation_spaces.append(Box(-10, 10, (84, 84, 1)))
            elif self.red_obs == "vector":
                self.observation_spaces.append(Box(-10, 10, (len(self.all_players) * 2,)))
            else:
                raise ValueError

        # action spaces
        self.action_spaces = []
        for bp in self.blue_team:
            if self.blue_actions == "discrete":
                self.action_spaces.append(Discrete(5))
            elif self.blue_actions == "continuous":
                self.action_spaces.append(Box(-1.0, 1.0, (2,)))
            else:
                raise ValueError

        for rp in self.red_team:
            if self.red_actions == "discrete":
                self.action_spaces.append(Discrete(5))
            elif self.red_actions == "continuous":
                self.action_spaces.append(Box(-1.0, 1.0, (2,)))
            else:
                raise ValueError

    def dist(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.sqrt(dx * dx + dy * dy)

    def reset(self):

        self.blue_team = []
        for t in range(self.team_size):
            self.blue_team.append([0.0, self.start_pos])

        self.red_team = []
        for t in range(self.team_size):
            self.red_team.append([0.0, -self.start_pos])

        self.all_players = self.blue_team + self.red_team

        self.epstep = 0

        # get states and return
        states = []
        for i in range(len(self.all_players)):
            states.append(self.get_state_for_player(i))

        return states

    def step(self, actions):

        self.epstep += 1

        for i in range(len(actions)):
            self.step_player(actions[i], i)

        # check for collisions
        for bp in self.blue_team:
            for rp in self.red_team:
                d = self.dist(bp, rp)
                if d < self.capture_radius:
                    bp[0] = 0.0
                    bp[1] = self.start_pos
                    rp[0] = 0.0
                    rp[1] = -self.start_pos

        # check for end-of-game
        done = False
        blue_reward = 0.0
        for bp in self.blue_team:
            if bp[1] < -1.0:
                done = True
                blue_reward += 1.0
        for rp in self.red_team:
            if rp[1] > 1.0:
                done = True
                blue_reward -= 1.0

        done = done or self.epstep >= 100
        rewards = [blue_reward for p in self.blue_team] + [-blue_reward for p in self.red_team]
        infos = [{} for p in self.all_players]

        # get states and return
        states = []
        for i in range(len(self.all_players)):
            states.append(self.get_state_for_player(i))

        return states, rewards, done, infos

    def step_player(self, action, idx):

        if idx < self.team_size:
            # blue player
            if self.blue_actions == "discrete":

                action = action[0]

                if action == 1:
                    self.all_players[idx][0] += self.player_movement
                elif action == 2:
                    self.all_players[idx][0] -= self.player_movement
                elif action == 3:
                    self.all_players[idx][1] += self.player_movement
                elif action == 4:
                    self.all_players[idx][1] -= self.player_movement

            elif self.blue_actions == "continuous":
                self.all_players[idx][0] += action[0] * self.player_movement
                self.all_players[idx][1] += action[1] * self.player_movement

            # blue player cannot exceed y=1.0
            if self.all_players[idx][1] > 1.0:
                self.all_players[idx][1] = 1.0

        else:
            # red player - actions are all reversed
            if self.red_actions == "discrete":

                action = action[0]

                if action == 1:
                    self.all_players[idx][0] += (-self.player_movement)
                elif action == 2:
                    self.all_players[idx][0] -= (-self.player_movement)
                elif action == 3:
                    self.all_players[idx][1] += (-self.player_movement)
                elif action == 4:
                    self.all_players[idx][1] -= (-self.player_movement)

            elif self.red_actions == "continuous":
                self.all_players[idx][0] += action[0] * (-self.player_movement)
                self.all_players[idx][1] += action[1] * (-self.player_movement)

            # red player cannot preceed y=-1.0
            if self.all_players[idx][1] < -1.0:
                self.all_players[idx][1] = -1.0

        # check for left-right walls
        if self.all_players[idx][0] > 1.0:
            self.all_players[idx][0] = 1.0
        if self.all_players[idx][0] < -1.0:
            self.all_players[idx][0] = -1.0

        self.blue_team = self.all_players[:self.team_size]
        self.red_team = self.all_players[self.team_size:]

    def render(self):

        img = np.ones((300, 300, 3), np.uint8) * 255.0

        # draw all blue
        for bp in self.blue_team:
            color = (0, 0, 255.0)
            pix_x = int((bp[0] + 1.0) * 0.5 * 300.0)
            pix_y = int((bp[1] + 1.0) * 0.5 * 300.0)
            cv2.circle(img, (pix_x, pix_y), 10, color, -1)

        # draw all red
        for rp in self.red_team:
            color = (255.0, 0, 0)
            pix_x = int((rp[0] + 1.0) * 0.5 * 300.0)
            pix_y = int((rp[1] + 1.0) * 0.5 * 300.0)
            cv2.circle(img, (pix_x, pix_y), 10, color, -1)

        # we actually are drawing in BGR, so flip last axis
        img = np.flip(img, axis=-1)

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.waitKey(1)

    def get_state_for_player(self, entity_idx):
        if entity_idx < self.team_size:
            if self.blue_obs == "image":
                return self.get_image_state_for_player(entity_idx)
            elif self.blue_obs == "vector":
                return self.get_vector_state_for_player(entity_idx)
            else:
                raise ValueError
        else:
            if self.red_obs == "image":
                return self.get_image_state_for_player(entity_idx)
            elif self.red_obs == "vector":
                return self.get_vector_state_for_player(entity_idx)
            else:
                raise ValueError

    # TODO: support vector states as well
    def get_vector_state_for_player(self, entity_idx):
        self_state = []
        ally_states = []
        enemy_states = []

        for idx in range(len(self.all_players)):

            if idx == entity_idx:
                self_state += self.all_players[idx]

            else:
                if entity_idx < self.team_size:
                    if idx < self.team_size:
                        ally_states += self.all_players[idx]
                    else:
                        enemy_states += self.all_players[idx]
                else:
                    if idx < self.team_size:
                        enemy_states += self.all_players[idx]
                    else:
                        ally_states += self.all_players[idx]

        state = self_state + ally_states + enemy_states
        return np.asarray(state)

    def get_image_state_for_player(self, entity_idx):

        img = np.zeros((84, 84, 1), np.int8)

        # draw all blue
        for bp in self.blue_team:
            color = 1.0 if entity_idx < self.team_size else -1.0
            pix_x = int((bp[0] + 1.0) * 0.5 * 84.0)
            pix_y = int((bp[1] + 1.0) * 0.5 * 84.0)
            cv2.circle(img, (pix_x, pix_y), 3, (color), -1)

        # draw all red
        for rp in self.red_team:
            color = 1.0 if entity_idx >= self.team_size else -1.0
            pix_x = int((rp[0] + 1.0) * 0.5 * 84.0)
            pix_y = int((rp[1] + 1.0) * 0.5 * 84.0)
            cv2.circle(img, (pix_x, pix_y), 3, (color), -1)

        # now re-draw this entity
        color = 2.0
        p = self.all_players[entity_idx]
        pix_x = int((p[0] + 1.0) * 0.5 * 84.0)
        pix_y = int((p[1] + 1.0) * 0.5 * 84.0)
        cv2.circle(img, (pix_x, pix_y), 3, (color), -1)

        # flip
        if entity_idx >= self.team_size:
            img = np.flipud(img)
            img = np.fliplr(img)

        return img
