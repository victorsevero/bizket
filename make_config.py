import json


def make_training_config(
    ini_path,
    bios_path,
    rom_path,
    display_method="Direct3D9",
):
    with open("ini_templates/training.ini") as fp:
        template = json.load(fp)

    _make_config(template, ini_path, bios_path, rom_path, display_method)


def make_enjoy_config(
    ini_path,
    bios_path,
    rom_path,
    display_method="Direct3D9",
):
    with open("ini_templates/enjoy.ini") as fp:
        template = json.load(fp)

    _make_config(template, ini_path, bios_path, rom_path, display_method)


def _make_config(template, ini_path, bios_path, rom_path, display_method):
    if display_method == "OpenGL":
        template["DispMethod"] = 0
    elif display_method == "Direct3D9":
        template["DispMethod"] = 2
    else:
        display_methods = ["Direct3D9", "OpenGL"]
        raise ValueError(f"display_method should be one of {display_methods}")

    duo_slash_bios_path = str(bios_path).replace("\\", "\\\\")
    duo_slash_rom_path = str(rom_path).replace("\\", "\\\\")

    template["FirmwareUserSpecifications"] = {"PSX+U": duo_slash_bios_path}
    template["RecentRoms"]["recentlist"] = [f"*OpenRom*{duo_slash_rom_path}"]

    with open(ini_path, "w") as fp:
        json.dump(template, fp, indent=2)
