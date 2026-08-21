"""Exportación a LiteRT con cuantización entera.

Tres decisiones que el teléfono da por hechas:

**Entrada `uint8` cruda 0–255.** Es exactamente lo que escribe
`TensorPreprocessor.toInputBuffer(quantized = true)`. Toda la normalización vive
dentro del grafo. Si la entrada fuese float32 habría que replicar las constantes
de ImageNet en Kotlin, y una discrepancia ahí no falla: solo empeora las
predicciones lo justo para que nadie sepa por qué.

**Salidas `float32`.** El grafo se cuantiza entero, pero logits y embedding se
dequantizan en el borde. El embedding es la entrada de Mahalanobis, que es una
forma cuadrática: el error de cuantización se eleva al cuadrado y se acumula
sobre 256 dimensiones. Ahorrar 1 KB por inferencia ahí saldría carísimo en
precisión de la compuerta 2.

**Cuantización entera completa, no híbrida.** La híbrida deja pesos en int8 pero
activaciones en float, y depende de que el delegate lo soporte. Los NNAPI de gama
baja —justo el segmento objetivo— caen a CPU con la híbrida y triplican la
latencia. La entera completa corre en el acelerador.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from rich.console import Console
from tensorflow import keras

console = Console()


@dataclass(frozen=True)
class ExportResult:
    path: Path
    size_bytes: int
    input_dtype: str
    output_shapes: list[tuple[str, list[int]]]
    fully_quantized: bool

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1_048_576


class ExportError(RuntimeError):
    pass


def _representative_factory(dataset: tf.data.Dataset) -> Callable[[], Iterator[list]]:
    def generator() -> Iterator[list]:
        for batch in dataset:
            images = batch[0] if isinstance(batch, tuple) else batch
            # Lote de uno y float32 en rango 0–255: el calibrador mide los rangos
            # reales de activación por capa, y para eso necesita ver los datos tal
            # como entran en producción.
            yield [tf.cast(images, tf.float32)]

    return generator


def convert(
    model: keras.Model,
    representative: tf.data.Dataset,
    destination: Path,
) -> ExportResult:
    destination.parent.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_factory(representative)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32

    fully_quantized = True
    try:
        flatbuffer = converter.convert()
    except Exception as error:  # noqa: BLE001 — el converter lanza de todo
        # Reintento permisivo. Se avisa fuerte: un modelo con operaciones en
        # float corre, pero puede caer a CPU en el teléfono y salirse del
        # presupuesto de latencia de 900 ms.
        console.print(
            f"[yellow]  ! cuantización entera estricta falló ({type(error).__name__}). "
            "Se reintenta permitiendo operaciones float — revisa la latencia en "
            "dispositivo antes de publicar.[/]"
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_factory(representative)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        flatbuffer = converter.convert()
        fully_quantized = False

    destination.write_bytes(flatbuffer)
    return _inspect(destination, fully_quantized)


def _inspect(path: Path, fully_quantized: bool) -> ExportResult:
    """Abre el artefacto con el intérprete y comprueba el contrato de salidas.

    Se verifica aquí, en el pipeline, y no en el teléfono: un `.tflite` con una
    sola salida se detectaría en campo como un cierre al abrir la cámara.
    """
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    if len(outputs) < 2:
        raise ExportError(
            f"El artefacto exporta {len(outputs)} salida(s). La doble salida "
            "—logits + embedding— es requisito de arquitectura (§13.1): sin "
            "embedding no existen ni la compuerta OOD ni la agrupación por "
            "similitud. El cargador de Android lo rechazaría."
        )

    dtype = np.dtype(inputs[0]["dtype"]).name
    if dtype not in ("uint8", "int8"):
        raise ExportError(
            f"La entrada quedó en {dtype}. El teléfono escribe bytes crudos 0–255 "
            "(`quantized = true`), y con una entrada float los interpretaría como "
            "valores diminutos: todas las predicciones saldrían iguales."
        )

    shapes = [(str(o["name"]), [int(v) for v in o["shape"]]) for o in outputs]

    return ExportResult(
        path=path,
        size_bytes=path.stat().st_size,
        input_dtype=dtype,
        output_shapes=shapes,
        fully_quantized=fully_quantized,
    )


def run_tflite(
    path: Path,
    images: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ejecuta el artefacto imagen a imagen y devuelve `(logits, embeddings)`.

    Las salidas se identifican **por tamaño**, igual que hace `LiteRtPestClassifier`:
    la que tiene tantos elementos como clases son los logits, la otra es el
    embedding. No se asume el orden, porque el converter no lo garantiza entre
    versiones y confundirlas produce predicciones absurdas sin ningún error.
    """
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    sizes = [int(np.prod(o["shape"])) for o in output_details]
    try:
        logits_index = sizes.index(class_count)
    except ValueError as error:
        raise ExportError(
            f"Ninguna salida tiene {class_count} elementos (hay {sizes}). "
            "El .tflite y la receta son de versiones distintas."
        ) from error
    embedding_index = next(i for i in range(len(sizes)) if i != logits_index)

    logits = np.zeros((len(images), class_count), dtype=np.float32)
    embeddings = np.zeros((len(images), sizes[embedding_index]), dtype=np.float32)

    scale, zero_point = input_detail["quantization"]
    target_dtype = np.dtype(input_detail["dtype"])

    for row, image in enumerate(images):
        if target_dtype in (np.uint8, np.int8):
            # `scale`/`zero_point` describen cómo el grafo interpreta el entero.
            # Con la entrada declarada en 0–255 la escala es 1 y el cero es 0,
            # pero se aplica la fórmula general por si el converter eligiera otra.
            if scale and abs(scale - 1.0) > 1e-6:
                prepared = np.round(image / scale + zero_point)
            else:
                prepared = np.round(image)
            prepared = np.clip(
                prepared, np.iinfo(target_dtype).min, np.iinfo(target_dtype).max
            ).astype(target_dtype)
        else:
            prepared = image.astype(target_dtype)

        interpreter.set_tensor(input_detail["index"], prepared[None, ...])
        interpreter.invoke()
        logits[row] = interpreter.get_tensor(output_details[logits_index]["index"]).reshape(-1)
        embeddings[row] = interpreter.get_tensor(
            output_details[embedding_index]["index"]
        ).reshape(-1)

    return logits, embeddings
