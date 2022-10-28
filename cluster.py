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


def cluster_test(n):
    server = Server(n_connections=n)
    for i in range(n):
        start_game()
        server.accept_connection()
    try:
        while True:
            for i in range(n):
                server.get_msg(i)
                if i % 2 == 0:
                    server.square_spam_strat(i)
                else:
                    server.x_spam_strat(i)
    except:
        server.close()


if __name__ == "__main__":
    cluster_test(8)
