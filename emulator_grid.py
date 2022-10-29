import subprocess


SCREEN_X = 2560


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


def get_handles():
    proc = subprocess.Popen(
        [
            "/mnt/c/Users/victo/Documents/BizHawk/cmdow.exe",
            "Mega Man X4 (USA) [Playstation] - BizHawk",
        ],
        stdout=subprocess.PIPE,
    )

    handles = []
    for line in proc.stdout:
        if line[:8] != b"Handle  ":
            handles.append(line[:8].decode())

    return handles


def close_emulators(handles):
    for handle in handles:
        subprocess.Popen(
            [
                "/mnt/c/Users/victo/Documents/BizHawk/cmdow.exe",
                handle,
                "/end",
            ],
            stdout=subprocess.PIPE,
        )


def move_window(handle, position):
    size_x = SCREEN_X // 4
    size_y = 3 * size_x // 4

    x = (position % 4) * size_x
    y = (position // 4) * size_y

    subprocess.Popen(
        [
            "/mnt/c/Users/victo/Documents/BizHawk/cmdow.exe",
            handle,
            "/res",
            "/mov",
            str(x),
            str(y),
            "/siz",
            str(size_x),
            str(size_y),
        ],
        stdout=subprocess.PIPE,
    )


def set_emulator_grid():
    handles = get_handles()
    for i, handle in enumerate(handles):
        move_window(handle, i)

    return handles


if __name__ == "__main__":
    set_emulator_grid()
