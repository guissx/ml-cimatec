# Machine Learning — CIMATEC

Atividades e projetos da disciplina de Machine Learning.

## Sumário

- [Commits por aula](#commits-por-aula)
- [Projetos](#projetos)

## Commits por aula

Os links abaixo apontam para os commits e arquivos de cada entrega. A pontuação corresponde ao valor previsto na rubrica da disciplina.

| Aula | Entrega | Commits | Arquivos | Pontos |
|---|---|---|---|---|
| 03/04 | Estrutura inicial do projeto Olist | [c2a3156](https://github.com/guissx/ml-cimatec/commit/c2a3156) | [Olist/](Olist/) | 0,5 |
| 04 | Notebook de qualidade dos dados | [337e837](https://github.com/guissx/ml-cimatec/commit/337e837) | [01_data_quality.ipynb](Olist/notebooks/01_data_quality.ipynb) | 0,5 |
| 05 | Análise exploratória (EDA) | [98deab4](https://github.com/guissx/ml-cimatec/commit/98deab4) | [02_EDA_template.ipynb](Olist/notebooks/02_EDA_template.ipynb) | 0,5 |
| 06 | Engenharia de features | [214cf13](https://github.com/guissx/ml-cimatec/commit/214cf13) [0169e8e](https://github.com/guissx/ml-cimatec/commit/0169e8e) [2226691](https://github.com/guissx/ml-cimatec/commit/2226691) | [features.py](Olist/module_olist/features.py) · [main.py](Olist/module_olist/main.py) | 0,5 |
| 07 | Modelo treinado | [84f6d4e](https://github.com/guissx/ml-cimatec/commit/84f6d4e), [e2c6be7](https://github.com/guissx/ml-cimatec/commit/e2c6be7) | [Treino](Olist/module_olist/modeling/train.py), [modelo salvo](Olist/models/model.joblib) | 1,0 |
| 08/09 | Validação com cross-validation | [e2c6be7](https://github.com/guissx/ml-cimatec/commit/e2c6be7) | [cross_validation.py](Olist/module_olist/modeling/cross_validation.py) | 1,0 |
| 10 | Inferência do modelo | [88a8f49](https://github.com/guissx/ml-cimatec/commit/88a8f49) | [inference.py](Olist/module_olist/inference.py) | 0,5 |
| 11 | Explicabilidade com SHAP | [33fe379](https://github.com/guissx/ml-cimatec/commit/33fe379) | [explain.py](Olist/module_olist/explain.py) · [interpret.py](Olist/module_olist/modeling/interpret.py) | 0,5 |
| 12 | Active learning | [2af1509](https://github.com/guissx/ml-cimatec/commit/2af1509) | [active_learning.py](Olist/module_olist/modeling/active_learning.py) | 0,5 |
| 13 | Estrutura inicial do projeto de deep learning | [2bdc451](https://github.com/guissx/ml-cimatec/commit/2bdc451) [e79f4d4](https://github.com/guissx/ml-cimatec/commit/e79f4d4) [587f727](https://github.com/guissx/ml-cimatec/commit/587f727) | [deep_learning/](deep_learning/) | 0,5 |
| 14 | Deep learning com Keras | [abd2aa6](https://github.com/guissx/ml-cimatec/commit/abd2aa6) [3c75832](https://github.com/guissx/ml-cimatec/commit/3c75832) | [atividade_tensorflow.ipynb](deep_learning/notebooks/atividade_tensorflow.ipynb) | 0,5 |
| 16 | Estrutura inicial do projeto de visão computacional | [7fd262b](https://github.com/guissx/ml-cimatec/commit/7fd262b) | [Cifar10/](Cifar10/) | 0,5 |
| 17/18 | Notebook com o classificador de imagens | [4571c35](https://github.com/guissx/ml-cimatec/commit/4571c35), [f92b9db](https://github.com/guissx/ml-cimatec/commit/f92b9db), [70426a1](https://github.com/guissx/ml-cimatec/commit/70426a1), [e66f0a2](https://github.com/guissx/ml-cimatec/commit/e66f0a2), [71ad1ac](https://github.com/guissx/ml-cimatec/commit/71ad1ac), [f703018](https://github.com/guissx/ml-cimatec/commit/f703018) | [Notebook com resultados](deep_learning/notebooks/Classificador_Imagens.ipynb), [versão modular](Cifar10/notebooks/01_classificador_cifar10.ipynb) | 1,0 |
| 19 | Solução do projeto de reconhecimento facial | [61d6e67](https://github.com/guissx/ml-cimatec/commit/61d6e67) [0c58d14](https://github.com/guissx/ml-cimatec/commit/0c58d14) [ca53550](https://github.com/guissx/ml-cimatec/commit/ca53550) | [Reconhecimento_Facial/](Reconhecimento_Facial/) | 1,0 |
| 20 | Cópia do projeto realizado em grupo | [bd76fcc](https://github.com/guissx/ml-cimatec/commit/bd76fcc) | [vaes_poc/](vaes_poc/) | 1,0 |
| | | | **Total previsto na rubrica** | **10,0** |

## Projetos

- [Olist](Olist/): qualidade dos dados, análise exploratória e previsão de atraso nas entregas.
- [Deep learning](deep_learning/): atividade com TensorFlow/Keras e classificação de imagens.
- [Cifar10](Cifar10/README.md): classificador de imagens organizado em módulos, com notebook e testes.
- [Reconhecimento facial](Reconhecimento_Facial/README.md): cadastro e reconhecimento de pessoas em imagens e pela webcam.
- [VAE POC](vaes_poc/README.md): cópia do projeto em grupo sobre detecção de anomalias em radiografias de tórax.
