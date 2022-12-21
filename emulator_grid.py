import subprocess
import threading

import pygetwindow as gw


EMULATORS_PER_ROW = 8
SCREEN_X = 2560
SIZE_X = SCREEN_X // EMULATORS_PER_ROW
SIZE_Y = 335


def start_emulator(boss, port=6969, enjoy=False):
    if enjoy:
        ini_file = r"--config=C:\Users\victo\Documents\bizket\emulators_configs\enjoy.ini"
    else:
        ini_file = r"--config=C:\Users\victo\Documents\bizket\emulators_configs\training.ini"
    return subprocess.Popen(
        [
            r"C:\Users\victo\Documents\BizHawk-rc2\EmuHawk.exe",
            ini_file,
            r"--lua=C:\Users\victo\Documents\bizket\client.lua",
            f"--load-slot={boss}",
            "--socket_ip=127.0.0.1",
            f"--socket_port={port}",
        ]
    )


def set_emulator_grid(n):
    windows = []
    while len(windows) != n:
        windows = gw.getWindowsWithTitle(
            "Mega Man X4 (USA) [PlayStation] - BizHawk (interim)"
        )

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
