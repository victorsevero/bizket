import socket
from typing import Dict


class Server:
    def __init__(self, ip="localhost", port=6969, n_connections=2):
        self.ip = ip
        self.port = port

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._bind_server(n_connections)
        self._n_connections = n_connections
        self._connections = []

    def _bind_server(self, n_connections):
        print(f"Starting server on {self.ip}:{self.port}")
        self._socket.bind((self.ip, self.port))
        self._socket.listen(n_connections)

    def accept_connection(self):
        if len(self._connections) == self._n_connections:
            raise Exception("Already at maximum number of connections")
        print("Waiting for a connection")
        connection, client_address = self._socket.accept()
        self._connections.append(_Connection(connection))
        print("Connection from {}:{}".format(*client_address))

    def get_game_data(self, i):
        msg = self.get_msg(i)
        self.send_msg("ok")

        return msg

    def get_msg(self, i):
        data = self._connections[i].recv(1024)
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

    def send_msg(self, msg: str, i: int = 0):
        self._connections[i].sendall(self._encode_msg(msg))

    @staticmethod
    def _encode_msg(string: str) -> bytes:
        msg = string.encode()
        msg = b" ".join((str(len(msg)).encode(), msg))

        return msg

    def square_spam_strat(self, i):
        if self._connections[i].frame % 21 == 0:
            msg = "s"
        else:
            msg = "ok"
        self.send_msg(msg, i)

    def x_spam_strat(self, i):
        if self._connections[i].frame % 21 == 0:
            msg = "x"
        else:
            msg = "ok"
        self.send_msg(msg, i)

    def close(self):
        for connection in self._connections:
            connection.close()

    def load_state(self, i):
        self.get_msg(i)
        self.send_msg("load", i)
        self._connections[i].frame = 0


class _Connection:
    def __init__(self, connection):
        self.connection = connection
        self.frame = 0

    def recv(self, *args, **kwargs):
        self.connection.recv(*args, **kwargs)

    def sendall(self, *args, **kwargs):
        self.connection.sendall(*args, **kwargs)
        self.frame += 1

    def close(self):
        self.connection.close()


if __name__ == "__main__":
    server = Server()
