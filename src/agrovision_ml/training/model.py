"""Modelo de doble salida: logits + embedding.

**La doble salida es requisito de arquitectura, no una comodidad** (§13.1 y §23,
error nº 3). El embedding alimenta dos subsistemas que sin él no existen:

- la compuerta 2 en el teléfono, que decide si el modelo está viendo algo que
  nunca vio (distancia de Mahalanobis sobre el embedding, no sobre el softmax);
- la agrupación por similitud en el servidor, que junta las solicitudes de
  revisión parecidas para que el agrónomo resuelva veinticuatro de una vez en
  lugar de una por una.

Un modelo exportado solo con logits compila, corre, predice — y deja muertos los
dos. Por eso el cargador de Android rechaza explícitamente cualquier `.tflite`
con menos de dos salidas.

**Toda la preparación de la imagen vive DENTRO del grafo.** El teléfono manda
bytes crudos 0–255 y el grafo escala. Es lo que evita el error clásico de
normalizar en Kotlin *y* volver a escalar en el grafo: no falla, no avisa, solo
empeora las predicciones lo justo para que nadie sepa por qué.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

from ..config import TrainingConfig

_BACKBONES = {
    "MobileNetV3Large": keras.applications.MobileNetV3Large,
    "MobileNetV3Small": keras.applications.MobileNetV3Small,
    "EfficientNetV2B0": keras.applications.EfficientNetV2B0,
}

LOGITS = "logits"
EMBEDDING = "embedding"


class ModelError(RuntimeError):
    pass


def build(config: TrainingConfig) -> tuple[keras.Model, keras.Model, keras.Model]:
    """Construye tronco, modelo de entrenamiento y modelo de exportación.

    Los tres comparten las mismas capas: entrenar el segundo actualiza los pesos
    que exporta el tercero. Se separan porque **entrenar con dos salidas obliga a
    inventarse una etiqueta para el embedding**, que no tiene ninguna. Con dos
    vistas del mismo grafo, la pérdida se define una sola vez sobre los logits y
    el embedding sale gratis en la exportación.

    Devuelve `(backbone, trainer, exporter)`.
    """
    name = config.section("model", "backbone")
    if name not in _BACKBONES:
        raise ModelError(
            f"Tronco «{name}» desconocido. Disponibles: {', '.join(sorted(_BACKBONES))}"
        )

    size = config.image_size
    dropout = float(config.section("model", "dropout"))

    # dtype float32 y rango 0–255: exactamente lo que escribe
    # `TensorPreprocessor.toInputBuffer(quantized = true)` en el teléfono.
    inputs = keras.Input(shape=(size, size, 3), dtype="float32", name="image")

    backbone = _BACKBONES[name](
        input_shape=(size, size, 3),
        include_top=False,
        weights="imagenet",
        # ⚠️ La normalización propia del tronco se queda DENTRO. Ver el docstring.
        include_preprocessing=True,
    )
    backbone.trainable = False

    features = backbone(inputs, training=False)
    pooled = keras.layers.GlobalAveragePooling2D(name="pooling")(features)
    pooled = keras.layers.Dropout(dropout, name="dropout_pooling")(pooled)

    # Cuello de botella lineal. Es EL embedding: la representación sobre la que
    # se calculan los centroides de clase y la covarianza compartida que usa
    # Mahalanobis. Sin activación a propósito — Mahalanobis asume una gaussiana
    # por clase, y un ReLU trunca en cero la mitad de cada distribución.
    embedding = keras.layers.Dense(
        config.embedding_dim,
        activation=None,
        name=EMBEDDING,
    )(pooled)

    head = keras.layers.Dropout(dropout / 2, name="dropout_head")(embedding)
    logits = keras.layers.Dense(config.class_count, activation=None, name=LOGITS)(head)

    trainer = keras.Model(inputs, logits, name="agrovision_trainer")
    exporter = keras.Model(inputs, [logits, embedding], name="agrovision")
    return backbone, trainer, exporter


def unfreeze_top(backbone: keras.Model, last_layers: int) -> int:
    """Descongela las últimas `last_layers` capas del tronco para el ajuste fino.

    **Las capas de normalización por lotes se quedan congeladas.** Es el error
    clásico del ajuste fino: al descongelarlas, con lotes de 32 imágenes las
    estadísticas móviles heredadas de ImageNet —calculadas sobre millones— se
    sobrescriben con ruido, y la exactitud de validación se desploma mientras la
    de entrenamiento sube. Parece sobreajuste y no lo es.

    Devuelve cuántas capas quedaron entrenables.
    """
    backbone.trainable = True
    trainable = 0

    for layer in backbone.layers[:-last_layers]:
        layer.trainable = False

    for layer in backbone.layers[-last_layers:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            trainable += 1

    return trainable


def classification_loss(class_count: int, smoothing: float):
    """Entropía cruzada con etiquetado suave sobre etiquetas enteras.

    El etiquetado suave necesita vectores one-hot, pero mantener las etiquetas
    como enteros en toda la canalización evita duplicar el formato entre
    conjuntos. Se convierte aquí dentro, en el único sitio que lo necesita.

    Sin suavizado, la red sale sobreconfiada: JMuBEN tiene lesiones ambiguas
    —una hoja con roya *y* minador etiquetada solo como una de las dos— y la red
    aprende a apostar al 0.999. Después la temperatura tiene que corregir tanto
    que el margen entre primera y segunda clase deja de discriminar.
    """
    objective = keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=smoothing,
    )

    def loss(y_true, y_pred):
        indices = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        return objective(tf.one_hot(indices, class_count), y_pred)

    return loss


def compile_for(model: keras.Model, config: TrainingConfig, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=classification_loss(
            config.class_count,
            float(config.section("model", "label_smoothing")),
        ),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acierto")],
    )
