
import numpy as np
import random
import gym
import pygame
from pygame_base.GameEngine import GameEngine
from pygame_base.GameElement import GameElement


# element classes =======================================================
class Player(GameElement):
    def __init__(self, color=[0,0,0], centered=False):
        w = 0.075
        super().__init__(0, 0, w, w, fill=color, stroke=color, centered=centered)

    def reset(self):
        self.x, self.y = random.random(), random.random()
        self.vx = 0
        self.vy = 0

    def draw(self, display):
        r = display.cvtX(self.w*0.5)
        pygame.draw.circle(display.display, self.fill, display.cvtPoint(self.x+self.w*0.5, self.y+self.w*0.5), r)


class Target(GameElement):
    def __init__(self, color=[0,0,0], centered=False):
        w = 0.15
        super().__init__(0, 0, w, w, fill=color, stroke=color, centered=centered)
        self.occupied = False
        self.collision_adjust = -0.05

    def reset(self):
        self.x, self.y = random.uniform(0.0, 1.0-self.w), random.uniform(0.0, 1.0-self.w)
        self.vx = 0
        self.vy = 0

    def draw(self, display):
        if not self.occupied:
            pygame.draw.rect(display.display, self.fill, pygame.Rect(*display.cvtRect(self.x, self.y, self.w, self.h)))
        else:
            pygame.draw.rect(display.display, self.stroke, pygame.Rect(*display.cvtRect(self.x, self.y, self.w, self.h)), 2)

    def subtick(self, game_elements):

        self.occupied = False
        for e in game_elements:
            if isinstance(e, Player):
                pcx, pcy = e.center
                if pcx > self.x and pcx<(self.x+self.w):
                    if pcy > self.y and pcy<(self.y+self.h):
                        self.occupied = True

        if self.occupied:
            return 0.25, False
        return 0.0, False


# environment ===========================================================
class CoverEnv(GameEngine):

    def __init__(self, N, headless=False, draws=False):

        def import_and_create_display():
            from pygame_base.GameDisplay import GameDisplay
            return GameDisplay()

        self.N = N

        super().__init__(headless=headless, make_display_fn=import_and_create_display, 
            step_limit=300, bg_color=[255,255,255], draws=draws)


    def setup(self):
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obspc = 3*self.N + 2*(self.N-1)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obspc,), dtype=np.float32)

        self.action_spaces = [self.action_space]*self.N
        self.observation_spaces = [self.observation_space]*self.N

        colors = [
            [255,0,0],
            [0,255,0],
            [0,0,255],
            [255,255,0],
            [255,0,255],
            [0,255,255],
            [0,0,0]
        ]
        self.players = []
        for i in range(self.N):
            p = Player(color=colors[i])
            self.players.append(p)

        self.targets = []
        for i in range(self.N):
            self.targets.append(Target())


    def apply_action(self, action):
        for i in range(self.N):
            self.players[i].vx = action[i][0]*0.06
            self.players[i].vy = action[i][1]*0.06

    def poststep_hook(self):
        for p in self.players:
            p.x = np.clip(p.x, 0, 1)
            p.y = np.clip(p.y, 0, 1)


    def reset_hook(self):

        for p in self.players:
            p.reset()

        valid=False
        while not valid:
            for t in self.targets:
                t.reset()
            valid = True

            for i in range(self.N):
                for j in range(self.N):
                    if i==j:
                        continue
                    if self.targets[i].check_collision(self.targets[j]):
                        valid = False


        self.game_elements = self.targets + self.players


    # completely ignore the image state and produce vectors
    def process_state(self, imgstate):
        states = []

        for i in range(self.N):
            origin = self.players[i]
            state = []

            # other players
            for j in range(self.N):
                if i != j:
                    other = self.players[j]
                    state.append(other.center[0] - origin.center[0])
                    state.append(other.center[1] - origin.center[1])

            # targets
            for j in range(self.N):
                other = self.targets[j]
                state.append(other.center[0] - origin.center[0])
                state.append(other.center[1] - origin.center[1])
                state.append(1.0 if other.occupied else 0.0)

            states.append(state)

        return states

    def process_rewards(self, rewards):
        return [rewards/self.N]*self.N

    def process_infos(self, infos):
        return [infos]*self.N



if __name__ == "__main__":

    from pygame_base.human_agent import human_agent
    from pygame_base.random_agent import random_agent

    env = CoverEnv(N=3, headless=False)
    random_agent(env, show_cv2=False, multiagent_interface=True)