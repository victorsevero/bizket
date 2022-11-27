import subprocess


SCREEN_X = 2560
SIZE_X = SCREEN_X // 4
SIZE_Y = 3 * SIZE_X // 4


def start_emulator():
    subprocess.Popen(
        [
            "/mnt/c/Users/victo/Documents/BizHawk/EmuHawk.exe",
            "--load-slot=3",
            "--socket_ip=127.0.1",
            "--socket_port=6969",
        ]
    )


def close_emulators():
    subprocess.Popen(
        ["/mnt/c/Windows/System32/taskkill.exe", "/IM", "EmuHawk.exe", "/F"],
    )


def set_emulator_grid():
    subprocess.Popen(
        ["/mnt/c/Program Files/Python310/python.exe", "emulator_grid.py"]
    )


def windows_grid():
    import pygetwindow as gw

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
