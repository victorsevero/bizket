# bizket

## How to reproduce

1. Extract ROM image from your own Mega Man X4 (US version) original PSX disk
2. Extract PSX BIOS from your own console
3. Download BizHawk 2.9-rc2 from [here](https://github.com/TASEmulators/BizHawk/releases/tag/2.9-rc2) and extract it locally
4. Copy `states\eregion.State` to your BizHawk `BizHawk-rc2\PSX\State` directory and rename it to `Mega Man X4 (USA).Nymashock.QuickSave0.State`
5. Create a virtual environment and install dependencies from `pyproject.toml` with [Poetry](https://python-poetry.org/) or install dependencies with pip from `requirements.txt`
6. Create a copy of `models_configs\example.yml` inside the same directory and rename it to `sevs.yml`; fill in all the necessary parameters inside the `paths` level. Adjust any other parameters if you like.
7. Run `main.py`.
8. Results will show up during runtime inside `logs` directory. You can visualize them with TensorBoard by running `tensorboard --logdir logs` inside project root directory.

PS.: Windows only!

PS².: This instructions reproduce only the first boss of the game with Zero. I will make a more detailed guide in the future when I turn this into a library.

## Explanatory Video in Portuguese
https://www.youtube.com/watch?v=zVA7WZxvtyA

## AI playing all main bosses of the game
https://www.youtube.com/watch?v=uYRemfDmwTk
