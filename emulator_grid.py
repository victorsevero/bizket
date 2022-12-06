import subprocess
import threading

import pygetwindow as gw


SCREEN_X = 2560
SIZE_X = SCREEN_X // 4
SIZE_Y = 3 * SIZE_X // 4


def start_emulator(port=6969):
    return subprocess.Popen(
        [
            r"C:\Users\victo\Documents\BizHawk\EmuHawk.exe",
            r"--config=C:\Users\victo\Documents\bizket\config.ini",
            "--load-slot=3",
            "--socket_ip=127.0.0.1",
            f"--socket_port={port}",
        ]
    )


def set_emulator_grid(n):
    windows = []
    while len(windows) != n:
        windows = gw.getWindowsWithTitle(
            "Mega Man X4 (USA) [PlayStation] - BizHawk"
        )

    for position, window in enumerate(windows):
        thread = threading.Thread(
            target=set_single_emulator,
            args=(position, window),
            daemon=True,
        )
        thread.start()


def set_single_emulator(position, window: gw.Win32Window):
    x = (position % 4) * SIZE_X
    y = (position // 4) * SIZE_Y

    window.restore()
    window.resizeTo(SIZE_X, SIZE_Y)
    window.moveTo(x, y)


if __name__ == "__main__":
    set_emulator_grid()
