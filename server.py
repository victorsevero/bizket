import socket
from io import BytesIO
from typing import Dict, List

import numpy as np
from PIL import Image


class Server:
    def __init__(self, n_connections, ip="localhost", port=6969):
        self.ip = ip
        self.port = port

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._bind_server(n_connections)
        self._n_connections = n_connections
        self.connections: List[Connection] = []

    def __getitem__(self, i):
        return self.connections[i]

    def __len__(self):
        return len(self.connections)

    def _bind_server(self, n_connections):
        print(f"Starting server on {self.ip}:{self.port}")
        self._socket.bind((self.ip, self.port))
        self._socket.listen(n_connections)

    def accept_connection(self):
        if len(self.connections) == self._n_connections:
            raise Exception("Already at maximum number of connections")
        print("Waiting for a connection")
        connection, client_address = self._socket.accept()
        self.connections.append(Connection(connection))
        print("Connection from {}:{}".format(*client_address))

    def get_game_data(self, i):
        return self.connections[i].get_game_data()

    def get_msg(self, i):
        return self.connections[i].get_msg()

    def send_msg(self, msg, i):
        self.connections[i].send_msg(msg)

    def square_spam_strat(self, i):
        self.connections[i].square_spam_strat()

    def x_spam_strat(self, i):
        self.connections[i].x_spam_strat()

    def load_state(self, i, need_msg: bool = True):
        self.connections[i].load_state(need_msg)

    def close(self):
        for connection in self.connections:
            connection.close()


class Connection:
    ACTIONS_MAP = {
        "nothing": "n",
        "left": "l",
        "right": "r",
        "cross": "x",
        "square": "s",
        "circle": "o",
        "load": "load",
        "close": "close",
        "ok": "ok",
    }

    def __init__(self, connection):
        self._connection: socket = connection
        self.frame = 0

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
        screen_matrix = self._decode_img(bytes(data))

        return screen_matrix, data_dict["player_hp"], data_dict["boss_hp"]

    @staticmethod
    def _decode_msg(data: bytes) -> Dict[str, int]:
        data_list = data.decode().split(" ")
        data_dict = {x[:2]: int(x[2:]) for x in data_list}
        data_dict["player_hp"] = data_dict.pop("ph")
        data_dict["boss_hp"] = data_dict.pop("bh")

        return data_dict

    @staticmethod
    def _decode_img(data: bytes):
        im = Image.open(BytesIO(data))
        return np.array(im.convert("RGB"))[:, 18:-12]

    def send_msg(self, msg: str):
        encoded_msg = self._encode_msg(self.ACTIONS_MAP[msg])
        self._connection.sendall(encoded_msg)
        self.frame += 1

    @staticmethod
    def _encode_msg(string: str) -> bytes:
        msg = string.encode()
        msg = b" ".join((str(len(msg)).encode(), msg))

        return msg

    def square_spam_strat(self):
        if self.frame % 21 == 0:
            msg = "square"
        else:
            msg = "ok"
        self.send_msg(msg)

    def x_spam_strat(self):
        if self.frame % 21 == 0:
            msg = "cross"
        else:
            msg = "ok"
        self.send_msg(msg)

    def close(self):
        # self.get_msg()
        # self.send_msg("close")
        self._connection.close()

    def load_state(self, need_msg: bool = True):
        if need_msg:
            self.get_msg()
        self.send_msg("load")
        self.frame = 0


if __name__ == "__main__":
    server = Server()
