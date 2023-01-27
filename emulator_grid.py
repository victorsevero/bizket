import subprocess
import threading
from pathlib import Path

import pygetwindow as gw


EMULATORS_PER_ROW = 8
SCREEN_X = 2560
SIZE_X = SCREEN_X // EMULATORS_PER_ROW
SIZE_Y = 240


def start_emulator(boss, port, bizhawk_path, ini_path):
    lua_client_path = Path("client.lua").resolve()
    return subprocess.Popen(
        [
            bizhawk_path,
            f"--config={ini_path}",
            f"--lua={lua_client_path}",
            f"--load-slot={boss}",
            "--socket_ip=127.0.0.1",
            f"--socket_port={port}",
        ]
    )


def set_emulator_grid(n):
    windows = []
    while len(windows) != n:
        windows = gw.getWindowsWithTitle("[PlayStation] - BizHawk (interim)")

    for position, window in enumerate(windows):
        thread = threading.Thread(
            target=set_single_emulator,
            args=(position, window),
            daemon=True,
        )
        thread.start()


def set_single_emulator(position, window: gw.Win32Window, resize=False):
    x = (position % EMULATORS_PER_ROW) * SIZE_X
    y = (position // EMULATORS_PER_ROW) * SIZE_Y

    window.restore()
    if resize:
        window.resizeTo(SIZE_X, SIZE_Y)
    window.moveTo(x, y)


if __name__ == "__main__":
    set_emulator_grid()
