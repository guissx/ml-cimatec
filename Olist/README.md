# Olist

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Brazilian E-Commerce Dive Deep using SQL

## Notebooks de qualidade dos dados

Os notebooks `01_data_quality.ipynb` e `profilling.ipynb` usam um ambiente
separado em Python 3.12, com versões fixadas em `requirements-profiling.txt`.
Na pasta `Olist`, execute:

```powershell
python -m uv venv .venv-profiling --python 3.12
python -m uv pip sync --python .venv-profiling/Scripts/python.exe requirements-profiling.txt
.venv-profiling/Scripts/python.exe -m jupyterlab notebooks
```

Selecione o kernel Python desse ambiente. Coloque `olist_orders_dataset.csv`
em `Olist/data/raw/` antes de executar as células em ordem. Os relatórios são
salvos em `Olist/reports/`. No Linux/macOS, use `bin/python` no lugar de
`Scripts/python.exe`.

Para verificar os notebooks com dados sintéticos, sem baixar a base:

```powershell
.venv-profiling/Scripts/python.exe -m pytest tests/test_data_quality_notebook.py -q
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         module_olist and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── module_olist   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes module_olist a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

