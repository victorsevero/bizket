import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env

from server import Server
from emulator_grid import start_emulator, set_emulator_grid, close_emulators


class Mmx4Env(gym.Env):
    CENTER_X = (0x12EF + 0x115D) / 2
    CENTER_Y = (0x01BA + 0x008D) / 2

    def __init__(self, n_connections=1, time=60):
        self.observation_space = spaces.Dict(
            {
                "player_hp": spaces.Discrete(32 + 1, start=0),
                "player_position": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(2,),
                    dtype=float,
                ),
                "boss_hp": spaces.Discrete(48 + 1, start=0),
                "boss_position": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(2,),
                    dtype=float,
                ),
            }
        )

        self.action_space = spaces.Discrete(5)
        self._action_to_button = {
            0: "left",
            1: "right",
            2: "cross",
            3: "circle",
            4: "square",
        }

        self._init_game(n_connections=n_connections)
        # 60 frames = 1 second, but it always skips 4 frames on each iteration
        self.max_steps = (60 // 4) * time
        self.frame = 0

    def _get_obs(self):
        return {
            "player_hp": self._player_hp,
            "player_position": self._player_position,
            "boss_hp": self._boss_hp,
            "boss_position": self._boss_position,
        }

    def _get_info(self):
        return self._get_obs()

    # Error while checking key=boss_hp: The observation returned by the `reset()`
    # method does not match the given observation space
    def reset(self, seed=None, options=None):
        self.frame = 0
        self.server.load_state(0)

        data = self.server.get_msg(0)
        (
            self._player_hp,
            self._player_position,
            self._boss_hp,
            self._boss_position,
        ) = self._msg_to_observation(data)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        past_player_hp = self._player_hp
        past_boss_hp = self._boss_hp

        self.server.send_msg(self._action_to_button[action], 0)
        data = self.server.get_msg(0)
        (
            self._player_hp,
            self._player_position,
            self._boss_hp,
            self._boss_position,
        ) = self._msg_to_observation(data)
        self.frame += 1

        terminated = (not self._player_hp) or (not self._boss_hp)
        boss_weight = 1
        # weight * boss taken damage - player taken damage
        reward = boss_weight * (past_boss_hp - self._boss_hp) - (
            past_player_hp - self._player_hp
        )
        truncated = self.frame >= self.max_steps
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        self.server.close()
        close_emulators(self.handles)

    @staticmethod
    def _scaler(value, value_min, value_max, scale_min=-1, scale_max=1):
        value_std = (value - value_min) / (value_max - value_min)
        return value_std * (scale_max - scale_min) + scale_min

    def _init_game(self, n_connections):
        self.server = Server(n_connections)
        for _ in range(n_connections):
            start_emulator()
            self.server.accept_connection()
        # self.handles = set_emulator_grid()

    def _msg_to_observation(self, data):
        data = data.copy()
        data["boss_x"] = data["boss_x"] - data["player_x"]
        data["boss_y"] = data["boss_y"] - data["player_y"]
        data["player_x"] = data["player_x"] - (0x12EF + 0x115D) / 2
        data["player_y"] = data["player_y"] - (0x01BB + 0x008D) / 2

        data["boss_x"] = self._scaler(
            data["boss_x"],
            value_min=0x115D - 0x12EF,
            value_max=0x12EF - 0x115D,
        )
        data["boss_y"] = self._scaler(
            data["boss_y"],
            value_min=0x008D - 0x01BA,
            value_max=0x01BA - 0x008D,
            scale_min=1,
            scale_max=-1,  # TODO: try centralizing char to jump distance (0 here and change Y center point above)
        )
        data["player_x"] = self._scaler(
            data["player_x"],
            value_min=(0x115D - 0x12EF) / 2,
            value_max=(0x12EF - 0x115D) / 2,
        )
        data["player_y"] = self._scaler(
            data["player_y"],
            value_min=(0x008D - 0x01BB) / 2,
            value_max=(0x01BB - 0x008D) / 2,
            scale_min=1,
            scale_max=-1,
        )

        return (
            data["player_hp"],
            np.array([data["player_x"], data["player_y"]]),
            data["boss_hp"],
            np.array([data["boss_x"], data["boss_y"]]),
        )


if __name__ == "__main__":
    env = Mmx4Env()
    check_env(env)
