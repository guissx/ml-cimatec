"""Arquitetura da rede convolucional.

Mesma rede do notebook Classificador_Imagens: data augmentation, três blocos
convolucionais com BatchNorm e Dropout e um classificador denso no final.
"""

from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    RandomFlip,
    RandomTranslation,
)

from module_cifar10.config import (
    CONV_BLOCKS,
    DENSE_DROPOUT,
    DENSE_UNITS,
    INPUT_SHAPE,
    NUM_CLASSES,
)


def conv_block(model: keras.Sequential, filters: int, dropout: float) -> None:
    """Adiciona um bloco conv-conv-pool que reduz a imagem pela metade.

    Args:
        model (keras.Sequential): Modelo em construção.
        filters (int): Número de filtros das duas convoluções.
        dropout (float): Fração de neurônios desligada ao fim do bloco.

    """
    # padding="same" mantém o tamanho da imagem nas convoluções; só o pooling
    # reduz, o que deixa o encolhimento previsível: 32 -> 16 -> 8 -> 4.
    model.add(Conv2D(filters, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout))


def build_model() -> keras.Sequential:
    """Monta e compila a CNN.

    Returns:
        keras.Sequential: Modelo compilado, pronto para o fit.

    """
    model = keras.Sequential(name="cnn_cifar10")
    model.add(Input(shape=INPUT_SHAPE))

    # Data augmentation: essas camadas só atuam no treino e viram identidade na
    # predição, então podem ficar dentro do próprio modelo.
    model.add(RandomFlip("horizontal"))
    model.add(RandomTranslation(0.1, 0.1))

    for filters, dropout in CONV_BLOCKS:
        conv_block(model, filters, dropout)

    model.add(Flatten())
    model.add(Dense(DENSE_UNITS, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(DENSE_DROPOUT))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    # Loss esparsa porque os rótulos são inteiros 0-9, sem one-hot.
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model
