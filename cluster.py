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
    with mp.Pool(n) as p:
        p.map(process_func, range(n))


def process_func(i):
    try:
        for _ in range(60 * 3 * 5):
            server.get_msg(i)
            if i % 2:
                server.square_spam_strat(i)
            else:
                server.x_spam_strat(i)
    finally:
        server.connections[i].close()


if __name__ == "__main__":
    n = 8
    server = Server(n_connections=n)
    for i in range(n):
        start_game()
        server.accept_connection()
    handles = set_emulator_grid()
    try:
        mp_cluster_test(n)
    except:
        close_emulators(handles)
