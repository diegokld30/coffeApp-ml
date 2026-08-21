"""Construcción de los conjuntos de entrenamiento, validación y prueba.

Dos decisiones que condicionan todo lo demás:

**1. El reparto es estratificado por clase y determinista.**
Un reparto global sobre 58 555 imágenes con clases de 6 572 y 18 985 elementos
deja, por azar, clases enteras infrarrepresentadas en validación — y entonces la
métrica de esa clase se calcula sobre un puñado de imágenes y no significa nada.
Se reparte dentro de cada clase, con la semilla de la receta.

**2. Las imágenes salen en el rango 0–255, sin normalizar.**
La normalización vive DENTRO del grafo (ver `training/model.py`). Así el mismo
tensor crudo sirve para entrenar en Keras, para cuantizar a INT8 y para lo que
el teléfono mete en el intérprete. Normalizar aquí obligaría a replicar
exactamente las mismas constantes en Kotlin, y una discrepancia de medio punto
en la media produce predicciones sutilmente peores que nadie detecta.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..config import TrainingConfig

AUTOTUNE = tf.data.AUTOTUNE
_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Split:
    paths: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.paths)

    def distribution(self, class_count: int) -> list[int]:
        counts = [0] * class_count
        for label in self.labels:
            counts[label] += 1
        return counts


@dataclass(frozen=True)
class Splits:
    train: Split
    val: Split
    test: Split


class DatasetError(RuntimeError):
    pass


def build_splits(images_root: Path, config: TrainingConfig) -> Splits:
    """Reparte los archivos en tres conjuntos disjuntos, de forma reproducible.

    El reparto no usa el generador aleatorio global: se deriva del hash de la
    ruta relativa de cada archivo. Consecuencia práctica —y es la que importa—:
    **una imagen cae siempre en el mismo conjunto**, aunque el dataset crezca.
    Con un `shuffle` normal, añadir fotos del piloto reubicaría imágenes de
    prueba a entrenamiento, y las métricas de la versión nueva dejarían de ser
    comparables con las de la anterior sin que nadie se diera cuenta.
    """
    ratios = config.section("data", "split")
    train_cut = float(ratios["train"])
    val_cut = train_cut + float(ratios["val"])

    buckets: dict[str, tuple[list[str], list[int]]] = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }

    for spec in config.classes:
        directory = images_root / spec.source_dir
        if not directory.is_dir():
            available = sorted(p.name for p in images_root.iterdir() if p.is_dir())
            raise DatasetError(
                f"No existe la carpeta «{spec.source_dir}» para la clase "
                f"{spec.class_id}. Carpetas disponibles: {', '.join(available) or '(ninguna)'}"
            )

        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in _SUFFIXES)
        if not files:
            raise DatasetError(f"La carpeta «{spec.source_dir}» no tiene imágenes")

        for path in files:
            fraction = _stable_fraction(f"{spec.source_dir}/{path.name}", config.seed)
            key = "train" if fraction < train_cut else "val" if fraction < val_cut else "test"
            buckets[key][0].append(str(path))
            buckets[key][1].append(spec.index)

    splits = Splits(
        train=Split(*buckets["train"]),
        val=Split(*buckets["val"]),
        test=Split(*buckets["test"]),
    )

    for name, split in (("validación", splits.val), ("prueba", splits.test)):
        empty = [
            config.classes[i].class_id
            for i, count in enumerate(split.distribution(config.class_count))
            if count == 0
        ]
        if empty:
            raise DatasetError(
                f"El conjunto de {name} quedó sin ejemplos de: {', '.join(empty)}. "
                "Revisa las proporciones de `data.split` o la cantidad de imágenes."
            )

    return splits


def _stable_fraction(key: str, seed: int) -> float:
    """Reparte `key` en [0, 1) de forma estable entre ejecuciones y máquinas.

    `hash()` de Python está aleatorizado por proceso (PYTHONHASHSEED), así que
    no sirve: dos corridas darían repartos distintos. BLAKE2b sí es estable.
    """
    digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


# ── Canalización tf.data ─────────────────────────────────────────────────────


def _decode(path: tf.Tensor, label: tf.Tensor, size: int) -> tuple[tf.Tensor, tf.Tensor]:
    data = tf.io.read_file(path)
    # `expand_animations=False` es obligatorio: sin él, decode_image devuelve un
    # tensor de rango desconocido y `resize` falla en tiempo de grafo.
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image = tf.image.resize(image, (size, size), method="bilinear")
    return tf.cast(image, tf.float32), label


def _augment(image: tf.Tensor, label: tf.Tensor, cfg: dict) -> tuple[tf.Tensor, tf.Tensor]:
    if cfg.get("horizontal_flip"):
        image = tf.image.random_flip_left_right(image)
    if cfg.get("vertical_flip"):
        image = tf.image.random_flip_up_down(image)

    brightness = float(cfg.get("brightness", 0.0))
    if brightness > 0:
        # En escala 0–255, así que el delta se expresa en niveles, no en fracción.
        image = tf.image.random_brightness(image, max_delta=brightness * 255.0)

    contrast = float(cfg.get("contrast", 0.0))
    if contrast > 0:
        image = tf.image.random_contrast(image, 1.0 - contrast, 1.0 + contrast)

    # ⚠️ El tono NO se altera. El color de la lesión ES el síntoma: una roya
    # desplazada 15° en matiz deja de ser naranja y el modelo aprendería que el
    # color no importa. Es justo lo contrario de lo que un agrónomo hace.

    return tf.clip_by_value(image, 0.0, 255.0), label


def to_tf_dataset(
    split: Split,
    config: TrainingConfig,
    *,
    training: bool,
    batch_size: int | None = None,
) -> tf.data.Dataset:
    size = config.image_size
    batch = batch_size or int(config.section("training", "batch_size"))

    dataset = tf.data.Dataset.from_tensor_slices((split.paths, split.labels))

    if training:
        dataset = dataset.shuffle(
            min(len(split), 8_192), seed=config.seed, reshuffle_each_iteration=True
        )

    dataset = dataset.map(lambda p, y: _decode(p, y, size), num_parallel_calls=AUTOTUNE)

    if training:
        aug = config.section("augmentation")
        dataset = dataset.map(lambda x, y: _augment(x, y, aug), num_parallel_calls=AUTOTUNE)
        geometric = _geometric_layers(config)
        if geometric is not None:
            dataset = dataset.map(
                lambda x, y: (geometric(x, training=True), y), num_parallel_calls=AUTOTUNE
            )

    return dataset.batch(batch).prefetch(AUTOTUNE)


def _geometric_layers(config: TrainingConfig):
    """Rotación y zoom como capas de Keras, que rellenan el borde correctamente.

    Hacerlo a mano con `tf.image` deja bordes negros que el modelo aprende como
    señal: acabaría asociando «esquina oscura» con la clase más aumentada.
    """
    cfg = config.section("augmentation")
    rotation = float(cfg.get("rotation", 0.0))
    zoom = float(cfg.get("zoom", 0.0))
    if rotation <= 0 and zoom <= 0:
        return None

    layers = []
    if rotation > 0:
        layers.append(tf.keras.layers.RandomRotation(rotation, fill_mode="reflect"))
    if zoom > 0:
        layers.append(tf.keras.layers.RandomZoom(zoom, fill_mode="reflect"))
    return tf.keras.Sequential(layers, name="aumentacion_geometrica")


def class_weights(split: Split, config: TrainingConfig) -> dict[int, float] | None:
    """Pesos inversamente proporcionales a la frecuencia.

    JMuBEN está desbalanceado casi 3:1. Sin pesos, predecir siempre «sana»
    acierta un tercio de las veces y el descenso de gradiente encuentra ese
    mínimo local antes que cualquier cosa útil.
    """
    if config.section("training", "class_weight") != "balanced":
        return None

    counts = np.asarray(split.distribution(config.class_count), dtype=np.float64)
    total = counts.sum()
    # La fórmula estándar de scikit-learn: n / (k · n_c).
    weights = total / (len(counts) * np.maximum(counts, 1.0))
    return {i: float(w) for i, w in enumerate(weights)}


def representative_batches(
    split: Split, config: TrainingConfig, sample_count: int
) -> tf.data.Dataset:
    """Muestra para calibrar los rangos del cuantizador INT8.

    Se toma del conjunto de ENTRENAMIENTO y sin aumentación: representa la
    distribución real de activaciones, no una versión distorsionada de ella.
    """
    indices = np.linspace(0, len(split) - 1, num=min(sample_count, len(split)), dtype=int)
    subset = Split([split.paths[i] for i in indices], [split.labels[i] for i in indices])
    return to_tf_dataset(subset, config, training=False, batch_size=1)
