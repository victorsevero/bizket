import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env

from server import Server
from emulator_grid import start_emulator


class Mmx4Env(gym.Env):
    ARROW_TO_BUTTON = {
        0: None,
        1: "left",
        2: "right",
    }
    ACTION_TO_BUTTON = {
        1: "cross",
        2: "square",
    }

    def __init__(self, boss, port=6969, time=600, enjoy=False):
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 1),
            dtype=np.uint8,
        )

        self.action_space = spaces.MultiDiscrete([3, 2, 2])

        self.server = Server(port=port)
        self._process = start_emulator(boss=boss, port=port, enjoy=enjoy)
        self.server.accept_connection()

        self.max_steps = 60 // 6 * time
        self.frame = 0
        self.first_load = True

        self.boss = boss

    def _get_obs(self):
        return self._screen_matrix

    def _get_info(self):
        return {"player_hp": self._player_hp, "boss_hp": self._boss_hp}

    def reset(self, seed=None, options=None):
        self.frame = 0
        self.server.load_state(self.boss, self.first_load)
        if self.first_load:
            self.first_load = False

        (
            self._screen_matrix,
            self._player_hp,
            self._boss_hp,
        ) = self.server.get_msg()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        past_player_hp = self._player_hp
        past_boss_hp = self._boss_hp

        self.server.send_msg(self._actions_to_buttons(action))

        (
            self._screen_matrix,
            self._player_hp,
            self._boss_hp,
        ) = self.server.get_msg()

        self.frame += 1

        terminated = not (self._player_hp and self._boss_hp)
        boss_dmg = (past_boss_hp - self._boss_hp) / 48
        player_dmg = (past_player_hp - self._player_hp) / 32
        reward = boss_dmg - player_dmg

        truncated = self.frame >= self.max_steps
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        self._process.terminate()

    def _actions_to_buttons(self, action):
        buttons = [
            self.ACTION_TO_BUTTON[i]
            for i, act in enumerate(action[1:], start=1)
            if act
        ]
        arrow = self.ARROW_TO_BUTTON[action[0]]
        if arrow is not None:
            buttons.append(arrow)
        if not buttons:
            buttons = ["nothing"]

        return buttons


if __name__ == "__main__":
    env = Mmx4Env(boss=0, port=6969)
    check_env(env)
    print("Environment passed the check")
