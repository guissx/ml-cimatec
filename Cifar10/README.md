# Cifar10

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Classificador de imagens CIFAR-10 com rede neural convolucional (Keras/TensorFlow).

É a versão em projeto do notebook `deep_learning/notebooks/Classificador_Imagens.ipynb`:
o mesmo modelo (3 blocos convolucionais com BatchNorm e Dropout, data augmentation,
EarlyStopping e ReduceLROnPlateau), separado em módulos reutilizáveis.

## Como rodar

```bash
python -m uv sync                              # instala as dependências
python -m module_cifar10.dataset               # baixa a base e salva em data/raw
python -m module_cifar10.modeling.train        # treina e salva models/cnn_cifar10.keras
python -m module_cifar10.modeling.evaluate     # acurácia e loss no conjunto de teste
python -m module_cifar10.plots                 # figuras em reports/figures
python -m pytest tests
```

Ou pelo `make`: `make data`, `make train`, `make evaluate`, `make plots`, `make test`.

O treino completo (60 épocas) leva bem mais tempo em CPU; no Colab com GPU T4 o
notebook original chegou a **89,64%** de acurácia no teste.

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── raw            <- CIFAR-10 salvo em cifar10.npz.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Modelo treinado (.keras) e histórico de treino (.json)
│
├── notebooks          <- Jupyter notebooks.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         module_cifar10 and configuration for tools like ruff
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── tests              <- Testes com pytest
│
└── module_cifar10     <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes module_cifar10 a Python module
    │
    ├── config.py               <- Caminhos, classes e hiperparâmetros
    │
    ├── dataset.py              <- Download, normalização e cache da base
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── model.py            <- Arquitetura da CNN
    │   ├── train.py            <- Treino com callbacks e persistência do modelo
    │   ├── evaluate.py         <- Avaliação no conjunto de teste
    │   └── predict.py          <- Inferência com o modelo treinado
    │
    └── plots.py                <- Exemplos da base, predições e curvas de treino
```

--------
