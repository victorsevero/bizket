import socket


def encode_msg(s: str) -> bytes:
    msg = s.encode()
    msg = b" ".join((str(len(msg)).encode(), msg))
    print(msg)

    return msg


def start_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ("localhost", 6969)
    print("Starting server on {}:{}".format(*server_address))
    sock.bind(server_address)
    sock.listen(1)

    print("Waiting for a connection")
    connection, client_address = sock.accept()
    print("Connection from {}:{}".format(*client_address))

    response = "s"

    try:
        while True:
            data = connection.recv(1024)
            if data:
                data_list = data.decode().split(" ")[1:]
                data_dict = {x[:2]: x[2:] for x in data_list}
                player_hp = data_dict["ph"]
                player_x = data_dict["px"]
                player_y = data_dict["py"]
                boss_hp = data_dict["bh"]
                boss_x = data_dict["bx"]
                boss_y = data_dict["by"]
                print("Player")
                print(f"HP: {player_hp}, X: {player_x}, Y: {player_y}\n")
                print("Boss")
                print(f"HP: {boss_hp}, X: {boss_x}, Y: {boss_y}\n\n")
                connection.sendall(encode_msg(response))
                if response != "ok":
                    response = "ok"
                else:
                    response = "s"
    except:
        print("Closing server")
        connection.close()


if __name__ == "__main__":
    start_server()
