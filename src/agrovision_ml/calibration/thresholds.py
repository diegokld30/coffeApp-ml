"""Umbrales de las compuertas 2 y 3, y ensamblado de `calibration.json`.

Tres números gobiernan cuándo la app dice «no sé» en lugar de arriesgar un
diagnóstico:

    ood_threshold   compuerta 2 · puntuación combinada por encima ⇒ nunca visto
    τ  (confidence) compuerta 3 · probabilidad máxima por debajo  ⇒ no me comprometo
    δ  (margin)     compuerta 3 · primera y segunda muy juntas    ⇒ ambiguo

El umbral OOD **no se inventa**: se fija en el percentil de las puntuaciones de
validación indicado en la receta. Con `ood_percentile: 0.95` se acepta que un 5 %
de las hojas legítimas acaben en «no la reconozco».

Ese 5 % es una decisión de producto, no un residuo estadístico. En un cultivo,
un falso positivo con confianza alta cuesta una aplicación de fungicida
innecesaria —dinero, veneno y desconfianza—; un falso negativo cuesta una
solicitud de revisión que un agrónomo resuelve en un día. Los costos no son
simétricos, y el umbral se inclina hacia el error barato.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import TrainingConfig
from . import mahalanobis as maha
from . import temperature as temp


def _compact(value: float, digits: int = 7) -> float:
    """Recorta a `digits` cifras significativas, conservando el valor float32.

    Redondear a un número fijo de DECIMALES no serviría: la matriz de precisión
    mezcla entradas del orden de 1e+2 con otras de 1e-5, y un `round(v, 6)`
    aplastaría estas últimas a cero. Justo las pequeñas son las que codifican las
    correlaciones débiles entre dimensiones del embedding.
    """
    number = float(value)
    if number == 0.0 or not np.isfinite(number):
        return 0.0
    magnitude = int(np.floor(np.log10(abs(number))))
    return round(number, digits - 1 - magnitude)


@dataclass(frozen=True)
class Calibration:
    version: str
    temperature: float
    confidence_threshold: float
    margin_threshold: float
    ood_threshold: float
    energy_weight: float
    energy_mean: float
    energy_std: float
    centroids: np.ndarray
    precision: np.ndarray

    # Diagnóstico — no viaja al teléfono, pero sí a `metrics.json`.
    temperature_fit: temp.TemperatureFit
    mahalanobis_stats: maha.MahalanobisStats
    in_domain_rejected: float

    def to_json(self) -> dict:
        """Formato que consume `ArtifactSchema.CalibrationJson`.

        La matriz de precisión viaja **aplanada por filas**, que es como la
        recorre el Kotlin: `precision[fila · d + columna]`. Aplanarla por columnas
        daría su transpuesta — que para una matriz simétrica es la misma, pero
        depender de esa coincidencia es frágil: el día que se use una precisión
        por clase dejaría de serlo, en silencio.

        Los números se recortan a siete cifras significativas. No es cosmética:
        `repr()` de un float32 promovido a float64 escribe cosas como
        `0.00023400001036934555` —veinte caracteres para siete cifras reales de
        información—. Con d=256 la matriz son 65 536 números, y la diferencia
        entre escribirlos completos o recortados es de 1,4 MB a 0,4 MB de
        descarga sobre datos móviles en zona rural. El valor float32 es idéntico:
        lo que se elimina es ruido de la conversión binario→decimal.
        """
        return {
            "version": self.version,
            "temperature": round(float(self.temperature), 6),
            "confidence_threshold": round(float(self.confidence_threshold), 6),
            "margin_threshold": round(float(self.margin_threshold), 6),
            "ood_threshold": round(float(self.ood_threshold), 6),
            "energy_weight": round(float(self.energy_weight), 6),
            "energy_mean": round(float(self.energy_mean), 6),
            "energy_std_dev": round(float(self.energy_std), 6),
            "class_centroids": [[_compact(v) for v in row] for row in self.centroids],
            "precision_matrix": [
                _compact(v) for v in self.precision.reshape(-1, order="C")
            ],
        }

    def write(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return path


def fit(
    config: TrainingConfig,
    *,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_logits: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
) -> Calibration:
    section = config.section("calibration")

    # ── Compuerta 3 · temperatura ────────────────────────────────────────────
    fitted = temp.fit(val_logits, val_labels)

    # ── Compuerta 2 · energía y Mahalanobis ──────────────────────────────────
    energy_stats = maha.fit_energy(val_logits, fitted.temperature)
    stats = maha.fit(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=val_embeddings,
        val_predicted=val_logits.argmax(axis=1),
        class_count=config.class_count,
    )

    energy_weight = float(section["energy_weight"])
    scores = maha.combined_score(
        logits=val_logits,
        embeddings=val_embeddings,
        stats=stats,
        energy_stats=energy_stats,
        temperature=fitted.temperature,
        energy_weight=energy_weight,
    )

    percentile = float(section["ood_percentile"])
    ood_threshold = float(np.quantile(scores, percentile))
    rejected = float((scores > ood_threshold).mean())

    return Calibration(
        version=config.version,
        temperature=fitted.temperature,
        confidence_threshold=float(section["min_confidence"]),
        margin_threshold=float(section["min_margin"]),
        ood_threshold=ood_threshold,
        energy_weight=energy_weight,
        energy_mean=energy_stats.mean,
        energy_std=energy_stats.std,
        centroids=stats.centroids,
        precision=stats.precision,
        temperature_fit=fitted,
        mahalanobis_stats=stats,
        in_domain_rejected=rejected,
    )


def simulate_gates(
    calibration: Calibration,
    logits: np.ndarray,
    embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    """Aplica las tres compuertas tal como lo haría el teléfono.

    Sirve para responder la pregunta que de verdad importa antes de publicar:
    *de cada cien fotos, ¿cuántas acaban en diagnóstico, cuántas en «no estoy
    seguro» y cuántas en «no la reconozco»?* Un modelo con F1 excelente que manda
    el 40 % a revisión es inservible: satura al agrónomo y el productor deja de
    usar la app.
    """
    scores = maha.combined_score(
        logits=logits,
        embeddings=embeddings,
        stats=calibration.mahalanobis_stats,
        energy_stats=maha.EnergyStats(calibration.energy_mean, calibration.energy_std),
        temperature=calibration.temperature,
        energy_weight=calibration.energy_weight,
    )

    scaled = logits.astype(np.float64) / calibration.temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)

    ordered = np.sort(probabilities, axis=1)[:, ::-1]
    top = ordered[:, 0]
    margin = top - (ordered[:, 1] if ordered.shape[1] > 1 else 0.0)

    is_ood = scores > calibration.ood_threshold
    low_confidence = ~is_ood & (top < calibration.confidence_threshold)
    ambiguous = ~is_ood & ~low_confidence & (margin < calibration.margin_threshold)

    return {
        "ood": is_ood,
        "low_confidence": low_confidence,
        "ambiguous": ambiguous,
        "identified": ~is_ood & ~low_confidence & ~ambiguous,
        "scores": scores,
        "confidence": top,
    }
