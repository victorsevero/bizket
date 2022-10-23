import socket


def start_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ("localhost", 6969)
    print("Starting server on {}:{}".format(*server_address))
    sock.bind(server_address)
    sock.listen(1)

    print("Waiting for a connection")
    connection, client_address = sock.accept()
    print("Connection from {}:{}".format(*client_address))

    response = "OK".encode()
    response = b" ".join((str(len(response)).encode(), response))

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
                connection.sendall(response)
    except:
        print("Closing server")
        connection.close()


if __name__ == "__main__":
    start_server()
