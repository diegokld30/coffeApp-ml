"""Verificación de paridad Keras ↔ TFLite (§18.4).

La cuantización a INT8 cambia los números. La pregunta no es *si* cambian, sino
si cambian lo bastante como para alterar la decisión.

Se comprueban tres cosas, y la tercera es la que más importa:

1. **Diferencia máxima en probabilidad** — cuánto se movió la confianza.
2. **Coincidencia de la clase predicha** — con qué frecuencia ambos dicen lo mismo.
3. **Deriva del embedding** — el coseno entre el embedding float y el cuantizado.

La tercera se pasa por alto casi siempre y es la que rompe la compuerta 2 en
silencio. Los centroides y la matriz de precisión se calculan sobre embeddings
**float**, mientras que en el teléfono se comparan contra embeddings
**cuantizados**. Si la cuantización desplaza el embedding, la distancia de
Mahalanobis se mide contra un centroide que ya no está donde debería, y la app
empieza a mandar a revisión fotos perfectamente reconocibles. Nada falla, nadie
se entera, y la tasa de «no la reconozco» sube sin explicación.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tensorflow import keras

from . import to_tflite


@dataclass(frozen=True)
class ParityReport:
    samples: int
    max_probability_delta: float
    mean_probability_delta: float
    agreement: float
    embedding_cosine_min: float
    embedding_cosine_mean: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return (
            self.max_probability_delta <= self.tolerance
            and self.agreement >= 0.98
            and self.embedding_cosine_min >= 0.95
        )

    def to_json(self) -> dict:
        return {
            "samples": self.samples,
            "max_probability_delta": round(self.max_probability_delta, 6),
            "mean_probability_delta": round(self.mean_probability_delta, 6),
            "prediction_agreement": round(self.agreement, 6),
            "embedding_cosine_min": round(self.embedding_cosine_min, 6),
            "embedding_cosine_mean": round(self.embedding_cosine_mean, 6),
            "tolerance": self.tolerance,
            "passed": self.passed,
        }

    def describe(self) -> str:
        mark = "✓" if self.passed else "✗"
        return (
            f"  {mark} paridad sobre {self.samples} imágenes · "
            f"Δp máx {self.max_probability_delta:.4f} (tolerancia {self.tolerance}) · "
            f"coincidencia {self.agreement:.2%} · "
            f"coseno del embedding mín {self.embedding_cosine_min:.4f}"
        )


class ParityError(RuntimeError):
    pass


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def verify(
    keras_model: keras.Model,
    tflite_path: Path,
    images: np.ndarray,
    class_count: int,
    tolerance: float,
) -> ParityReport:
    """Compara ambas implementaciones sobre las mismas imágenes.

    `images` va en float32 y rango 0–255, la misma forma que consume el grafo.
    """
    if len(images) == 0:
        raise ParityError("No hay imágenes con las que verificar la paridad")

    reference_logits, reference_embeddings = keras_model.predict(images, verbose=0)
    reference_logits = np.asarray(reference_logits)
    reference_embeddings = np.asarray(reference_embeddings)

    quantized_logits, quantized_embeddings = to_tflite.run_tflite(
        tflite_path, images, class_count
    )

    reference_probabilities = _softmax(reference_logits)
    quantized_probabilities = _softmax(quantized_logits)
    delta = np.abs(reference_probabilities - quantized_probabilities)

    agreement = float(
        (reference_logits.argmax(axis=1) == quantized_logits.argmax(axis=1)).mean()
    )

    # Coseno en lugar de distancia euclídea: Mahalanobis es invariante a la
    # escala global del embedding —queda absorbida por la covarianza— pero no a
    # un cambio de dirección. El coseno mide justo eso.
    numerator = np.sum(reference_embeddings * quantized_embeddings, axis=1)
    denominator = np.linalg.norm(reference_embeddings, axis=1) * np.linalg.norm(
        quantized_embeddings, axis=1
    )
    cosine = numerator / np.maximum(denominator, 1e-12)

    return ParityReport(
        samples=len(images),
        max_probability_delta=float(delta.max()),
        mean_probability_delta=float(delta.mean()),
        agreement=agreement,
        embedding_cosine_min=float(cosine.min()),
        embedding_cosine_mean=float(cosine.mean()),
        tolerance=tolerance,
    )
