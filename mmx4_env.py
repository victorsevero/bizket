import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env

from server import Server
from emulator_grid import start_emulator, set_emulator_grid


N_PROCESSES = 2
server = Server(n_connections=N_PROCESSES)
for i in range(N_PROCESSES):
    start_emulator()
    server.accept_connection()
set_emulator_grid(N_PROCESSES)
# for _ in range(100):
#     for i in range(N_PROCESSES):
#         server.get_game_data(i)


class Mmx4Env(gym.Env):
    ACTION_TO_BUTTON = {
        0: "nothing",
        1: "left",
        2: "right",
        3: "cross",
        4: "square",
        # 5: "circle",
    }

    def __init__(self, connection_idx, time=600):
        global server
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(240, 320, 3),
            dtype=np.uint8,
        )

        self.action_space = spaces.Discrete(5)

        self.connection = server[connection_idx]

        # 60 frames = 1 second, but it always skips 10 frames on each iteration
        self.max_steps = (60 // 10) * time
        # self.max_steps = 60 * time
        self.frame = 0
        self.first_load = True

    def _get_obs(self):
        return self._screen_matrix

    def _get_info(self):
        return {"player_hp": self._player_hp, "boss_hp": self._boss_hp}

    def reset(self, seed=None, options=None):
        self.frame = 0
        self.connection.load_state(self.first_load)
        if self.first_load:
            self.first_load = False

        (
            self._screen_matrix,
            self._player_hp,
            self._boss_hp,
        ) = self.connection.get_msg()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        past_player_hp = self._player_hp
        past_boss_hp = self._boss_hp

        self.connection.send_msg(self.ACTION_TO_BUTTON[action])

        (
            self._screen_matrix,
            self._player_hp,
            self._boss_hp,
        ) = self.connection.get_msg()

        self.frame += 1

        terminated = not (self._player_hp and self._boss_hp)
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
        ...


if __name__ == "__main__":
    env = Mmx4Env(0)
    check_env(env)
