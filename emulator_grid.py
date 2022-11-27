import subprocess

import pygetwindow as gw


SCREEN_X = 2560
SIZE_X = SCREEN_X // 4
SIZE_Y = 3 * SIZE_X // 4


def start_emulator():
    subprocess.Popen(
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


def close_emulators():
    subprocess.run(
        ["/mnt/c/Windows/System32/taskkill.exe", "/IM", "EmuHawk.exe", "/F"],
    )


def set_emulator_grid():
    subprocess.Popen(
        ["/mnt/c/Program Files/Python310/python.exe", "emulator_grid.py"]
    )


def windows_grid():
    windows = gw.getWindowsWithTitle(
        "Mega Man X4 (USA) [PlayStation] - BizHawk"
    )
    for position, window in enumerate(windows):
        x = (position % 4) * SIZE_X
        y = (position // 4) * SIZE_Y

        window.restore()
        window.resizeTo(SIZE_X, SIZE_Y)
        window.moveTo(x, y)


if __name__ == "__main__":
    windows_grid()
