import socket
from typing import Dict, List


class Server:
    def __init__(self, n_connections, ip="localhost", port=6969):
        self.ip = ip
        self.port = port

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._bind_server(n_connections)
        self._n_connections = n_connections
        self.connections: List[Connection] = []

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

    def load_state(self, i):
        self.connections[i].load_state()

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
    }

    def __init__(self, connection):
        self._connection = connection
        self.frame = 0

    def get_game_data(self):
        msg = self.get_msg()
        self.send_msg("ok")

        return msg

    def get_msg(self):
        data = self._connection.recv(1024)
        if data:
            return self._decode_msg(data)

    @staticmethod
    def _decode_msg(data: bytes) -> Dict[str, int]:
        data_list = data.decode().split(" ")[1:]
        data_dict = {x[:2]: int(x[2:]) for x in data_list}
        data_dict["player_hp"] = data_dict.pop("ph")
        data_dict["player_x"] = data_dict.pop("px")
        data_dict["player_y"] = data_dict.pop("py")
        data_dict["boss_hp"] = data_dict.pop("bh")
        data_dict["boss_x"] = data_dict.pop("bx")
        data_dict["boss_y"] = data_dict.pop("by")

        return data_dict

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
        self.get_msg()
        self.send_msg("close")
        self._connection.close()

    def load_state(self):
        self.get_msg()
        self.send_msg("load")
        self.frame = 0


if __name__ == "__main__":
    server = Server()
