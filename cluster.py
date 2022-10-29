from subprocess import Popen
import multiprocessing as mp

from server import Server
from emulator_grid import set_emulator_grid, close_emulators


def start_game():
    Popen(
        "/mnt/c/Users/victo/Documents/BizHawk/EmuHawk.exe"
        # + " --config=python.ini"
        + " --load-slot=3"
        # + " --lua=client.lua"
        + " --socket_ip=127.0.1" + " --socket_port=6969",
        # + ' "roms/Mega Man X4 (USA).bin"',
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


def mp_cluster_test(n):
    server = Server(n_connections=n)
    for _ in range(n):
        start_game()
        server.accept_connection()
    handles = set_emulator_grid()
    try:
        with mp.Pool(n) as p:
            p.starmap(process_func, zip(range(n), server.connections))
    finally:
        close_emulators(handles)


def process_func(i, connection):
    try:
        for _ in range(60 * 3 * 5):
            connection.get_msg()
            if i % 2:
                connection.square_spam_strat()
            else:
                connection.x_spam_strat()
    finally:
        connection.close()


if __name__ == "__main__":
    mp_cluster_test(2)
