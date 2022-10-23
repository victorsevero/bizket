import socket
from typing import Dict


class Server:
    def __init__(self, ip="localhost", port=6969):
        self.ip = ip
        self.port = port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._bind_server()
        self._accept_conn()

        self._i = 0

    def _bind_server(self):
        print(f"Starting server on {self.ip}:{self.port}")
        self.sock.bind((self.ip, self.port))
        self.sock.listen(1)

    def _accept_conn(self):
        print("Waiting for a connection")
        self._conn, client_address = self.sock.accept()
        print("Connection from {}:{}".format(*client_address))

    def get_game_data(self):
        data = self._conn.recv(1024)
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
        print(data_dict)

        return data_dict

    def send_msg(self, msg: str):
        self._conn.sendall(self._encode_msg(msg))
        self._i += 1

    @staticmethod
    def _encode_msg(string: str) -> bytes:
        msg = string.encode()
        msg = b" ".join((str(len(msg)).encode(), msg))

        return msg

    def square_spam_strat(self):
        if self._i % 21 == 0:
            msg = "s"
        else:
            msg = "ok"
        self.send_msg(msg)

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    server = Server()
    try:
        while True:
            game_data = server.get_game_data()
            server.square_spam_strat()
    except:
        server.close()
