# Used Packages (Repo-wide)

This file lists packages and modules imported anywhere in this workspace. Versions are not captured here; run the commands below in your environment to get exact versions.

## External (third-party) packages

- numpy (2.4.6)
- matplotlib (3.11.0)
- scipy (1.18.0)
- pandas (3.0.3)
- dstapi (0.2)

## Standard library modules

- types
- time
- re

## Local / In-repo modules (imported from other files in this workspace)

- A1
- A2
- A5
- grid_solve
- ASADModel
- ExchangeEconomyModel
- Worker
- Government
- LaborMarketModel
- LaborSupplyModel

## Notes

- `mpl_toolkits.mplot3d` and submodules under `matplotlib` are provided by `matplotlib`.
- Some imports reference modules that are local to this repository (see Local section).
- To capture package versions from your active environment, run:

```bash
python3 -m pip show numpy matplotlib scipy pandas dstapi || true
echo '---'
python3 -m pip freeze | grep -E "numpy|matplotlib|scipy|pandas|dstapi" || true
```

## Detected versions (from the environment where I ran `pip`)

The following versions were detected when querying the active environment on this machine:

- `numpy`: 2.4.6
- `matplotlib`: 3.11.0
- `scipy`: 1.18.0
- `pandas`: 3.0.3
- `dstapi`: 0.2 (installed from git: `git+https://github.com/alemartinello/dstapi@d9eeb5a82cbc70b7d63b2ff44d9`) 

If you want, I can also generate a `requirements.txt` with exact versions from your chosen Python environment.
