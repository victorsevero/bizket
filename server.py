import socket
from io import BytesIO
from typing import Dict

import numpy as np
import cv2


class Server:
    ACTIONS_MAP = {
        "nothing": "n",
        "left": "l",
        "right": "r",
        "cross": "x",
        "square": "s",
        "circle": "o",
    }

    def __init__(self, ip="localhost", port=6969, img_size=(84, 84)):
        self.ip = ip
        self.port = port

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        print(f"Starting server on {self.ip}:{self.port}")
        self._socket.bind((self.ip, self.port))
        self._socket.listen()

        self._img_size = img_size

    def accept_connection(self):
        self._connection, _ = self._socket.accept()
        print(f"Connection established in port {self.port}")

    def get_game_data(self):
        msg = self.get_msg()
        self.send_msg("nothing")

        return msg

    def get_msg(self):
        size = bytearray()
        while not size.endswith(b" "):
            size += self._connection.recv(1)
        size = int(size[:-1].decode())

        data = self._connection.recv(size)
        data_dict = self._decode_msg(data)

        data = bytearray()
        while not data.endswith(b"IEND\xaeB`\x82"):
            data += self._connection.recv(4096)
        screen_matrix = self._decode_img(
            bytes(data).split(b" ", maxsplit=1)[-1]
        )

        return screen_matrix, data_dict["player_hp"], data_dict["boss_hp"]

    @staticmethod
    def _decode_msg(data: bytes) -> Dict[str, int]:
        data_list = data.decode().split(" ")
        data_dict = {x[:2]: int(x[2:]) for x in data_list}
        data_dict["player_hp"] = data_dict.pop("ph")
        data_dict["boss_hp"] = data_dict.pop("bh")

        return data_dict

    def _decode_img(self, data: bytes):
        arr = np.frombuffer(data, np.uint8)
        arr = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        arr = arr[:, 18:-12]
        # arr = arr[:, 86:-74]
        arr = cv2.resize(arr, self._img_size, interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(arr, axis=2).transpose(2, 0, 1)

    def send_msg(self, msg: str):
        if isinstance(msg, list):
            joined_msg = "".join([self.ACTIONS_MAP[x] for x in msg])
            encoded_msg = self._encode_msg(joined_msg)
        else:
            encoded_msg = self._encode_msg(self.ACTIONS_MAP.get(msg, msg))
        self._connection.sendall(encoded_msg)

    @staticmethod
    def _encode_msg(string: str) -> bytes:
        msg = string.encode()
        msg = b" ".join((str(len(msg)).encode(), msg))

        return msg

    def close(self):
        self._connection.close()

    def load_state(self, state: int, need_msg: bool = True):
        if need_msg:
            self.get_msg()
        self.send_msg(f"load{state}")


if __name__ == "__main__":
    server = Server()
