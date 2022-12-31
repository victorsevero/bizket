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
        3: "circle",
    }

    def __init__(
        self,
        boss,
        port=6969,
        time=600,
        image_size=(84, 84),
        allow_circle=True,
        enjoy=False,
    ):

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1,) + image_size,
            dtype=np.uint8,
        )

        if allow_circle:
            self.action_space = spaces.MultiDiscrete([3, 2, 2, 2])
        else:
            self.action_space = spaces.MultiDiscrete([3, 2, 2])

        self.server = Server(port=port, img_size=image_size)
        self._process = start_emulator(boss=boss, port=port, enjoy=enjoy)
        self.server.accept_connection()

        self.max_steps = 60 // 6 * time
        self.frame = 0
        self.first_load = True

        self.enjoy = enjoy
        self.boss = boss

    def _get_obs(self):
        return self._screen_matrix

    def _get_info(self):
        return {
            "player_hp": self.player_hp,
            "boss_hp": self.boss_hp,
        }

    def reset(self, seed=None, options=None):
        self.frame = 0
        self.server.load_state(self.boss, self.first_load)
        if self.first_load:
            self.first_load = False

        (
            self._screen_matrix,
            self.player_hp,
            self.boss_hp,
        ) = self.server.get_msg()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        past_player_hp = self.player_hp
        past_boss_hp = self.boss_hp

        self.server.send_msg(self._actions_to_buttons(action))

        (
            self._screen_matrix,
            self.player_hp,
            self.boss_hp,
        ) = self.server.get_msg()

        self.frame += 1

        self.boss_dmg = past_boss_hp - self.boss_hp
        self.player_dmg = past_player_hp - self.player_hp

        observation = self._get_obs()
        reward = self.boss_dmg - self.player_dmg
        if self.enjoy:
            if self.player_dmg or self.boss_dmg:
                print(f"Player HP: {self.player_hp}; Boss HP: {self.boss_hp}")
            # until any of them dies
            terminated = not (self.player_hp and self.boss_hp)

            if not self.boss_hp:
                for _ in range(100):
                    self.server.send_msg(["nothing"])
                    self.server.get_msg()
        else:
            # until player takes damage or kills boss
            terminated = bool(self.player_dmg) or not self.boss_hp
        truncated = self.frame >= self.max_steps
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
