from subprocess import Popen

from server import Server


def start_game():
    Popen(
        "/mnt/c/Users/victo/Documents/BizHawk/EmuHawk.exe"
        + " --config=bizhawk_config.ini"
        + " --load-slot=3"
        + " --lua=client.lua"
        + " --socket_ip=127.0.1"
        + " --socket_port=6969",
        shell=True,
        stdin=None,
        stdout=None,
        stderr=None,
    )


if __name__ == "__main__":
    server = Server()

    for i in range(2):
        start_game()
        server.accept_connection()
    server.load_state(0)
    server.load_state(1)
    try:
        while True:
            game_data = server.get_msg(0)
            server.square_spam_strat(0)
            game_data = server.get_msg(1)
            server.x_spam_strat(1)
    except:
        server.close()
