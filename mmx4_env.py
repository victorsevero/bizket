import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env

from server import Server
from emulator_grid import start_emulator


class Mmx4Env(gym.Env):
    CENTER_X = (0x12EF + 0x115D) / 2
    CENTER_Y = (0x01BA + 0x008D) / 2

    def __init__(self, connection=None, time=60):
        self.observation_space = spaces.Dict(
            {
                "player_position": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(2,),
                    dtype=float,
                ),
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

        if connection is None:
            self._init_single_game()
        else:
            self.connection = connection

        # 60 frames = 1 second, but it always skips 4 frames on each iteration
        self.max_steps = (60 // 15) * time
        self.frame = 0

    def _get_obs(self):
        return {
            "player_position": self._player_position,
            "boss_position": self._boss_position,
        }

    def _get_info(self):
        return self._get_obs()

    def reset(self, seed=None, options=None):
        self.frame = 0
        self.connection.load_state()

        data = self.connection.get_msg()
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

        self.connection.send_msg(self._action_to_button[action])
        data = self.connection.get_msg()
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

        reward = self._scaler(
            reward,
            value_min=-5,  # highest dmg from Boss
            value_max=boss_weight * 5,  # X charged buster or Zero saber
        )
        truncated = self.frame >= self.max_steps
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        self.connection.close()

    @staticmethod
    def _scaler(value, value_min, value_max, scale_min=-1, scale_max=1):
        value_std = (value - value_min) / (value_max - value_min)
        return value_std * (scale_max - scale_min) + scale_min

    def _init_single_game(self):
        server = Server(n_connections=1)
        start_emulator()
        server.accept_connection()

        self.connection = server.connections[0]

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
