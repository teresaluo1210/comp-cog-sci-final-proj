# Reproducibility Guide

## Prerequisites
- Ensure your virtual environment is set up and dependencies are installed.
- From the project root, create the output folder:

```zsh
mkdir -p figs
```

## Run: Tactful (model-only)
Saves: `figs/bar-modelonly-kind-tactful.pdf`

```zsh
$PWD/.venv/bin/python -c 'import utils; params={"uc":5,"urc":2,"tc":-5.08,"tca":2,"sc":30,"scc":0.09,"alpha":0.7,"alphac":0.3}; utils.main_model_only(params, kind=True, output_dir="figs", file_tag="tactful")'
```

## Run: Candid (model-only)
Saves: `figs/bar-modelonly-unkind-candid.pdf`

```zsh
$PWD/.venv/bin/python -c 'import utils; params={"uc":3,"urc":7,"tc":-2.7,"tca":-0.2,"sc":0.01,"scc":0.01,"alpha":0.5,"alphac":0.5}; utils.main_model_only(params, kind=False, output_dir="figs", file_tag="candid")'
```

## Parameter Presets
These values reflect qualitative fits intended to balance understanding/inference against social costs (shame vs self-efficacy).

| Param   | Tactful | Candid |
|---------|---------|--------|
| `uc`    | 5       | 3      |
| `urc`   | 2       | 7      |
| `tc`    | -5.08   | -2.7   |
| `tca`   | 2       | -0.2   |
| `sc`    | 30      | 0.01   |
| `scc`   | 0.09    | 0.01   |
| `alpha` | 0.7     | 0.5    |
| `alphac`| 0.3     | 0.5    |

## Notes
- `kind=True` renders the tactful color scheme; `kind=False` renders candid.
- Figures are saved as PDFs; open the files under `figs/` to verify labels and distributions.
- If you prefer running with the active Python from your shell, you can replace `$PWD/.venv/bin/python` with `python` once your environment is activated.
