"""Estadísticas de Mahalanobis y de energía para la compuerta 2.

Este módulo es un **espejo exacto** de `EnergyMahalanobisGate.kt`. Cada fórmula
de aquí tiene su gemela en Kotlin, y si divergen el umbral calibrado en Python
deja de significar nada en el teléfono: la app rechazaría fotos buenas o
aceptaría plagas que nunca vio, sin ningún error visible.

Cuando toques una fórmula de aquí, toca la de allí en el mismo commit.

## La corrección de escala, que no es obvia

El teléfono combina así:

    combinada = w · energía_estandarizada + (1 − w) · mahalanobis

La energía llega estandarizada (media 0, desviación 1) porque el Kotlin le resta
`energyMean` y la divide por `energyStdDev`. Mahalanobis **no**: llega cruda, y
en un embedding de 256 dimensiones su magnitud típica ronda √256 ≈ 16.

Combinar sin más un término de magnitud ~1 con otro de magnitud ~16 hace que
`energy_weight: 0.5` sea una mentira: la energía aportaría un 6 % de la decisión.

Como el Kotlin solo aplica `sqrt(δᵀ Λ δ)`, un factor de escala **sí** se puede
plegar dentro de Λ: multiplicar la matriz de precisión por k² multiplica la
distancia por k. Se aprovecha eso para embarcar una Λ ya escalada de forma que
las distancias en validación tengan desviación 1. Así `energy_weight` pondera lo
que dice que pondera, y el Kotlin no necesita saber nada de esto.

El desplazamiento (la media) no se puede plegar, y no hace falta: el umbral se
calibra sobre la misma puntuación combinada que se embarca.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MahalanobisStats:
    """Lo que viaja en `calibration.json`, ya en su forma final."""

    centroids: np.ndarray  # (k, d)
    precision: np.ndarray  # (d, d), ya escalada
    scale_factor: float  # k aplicado, solo para el informe
    raw_distance_mean: float
    raw_distance_std: float
    condition_number: float
    shrinkage: float


@dataclass(frozen=True)
class EnergyStats:
    mean: float
    std: float


def log_sum_exp(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    maximum = scaled.max(axis=1)
    return maximum + np.log(np.exp(scaled - maximum[:, None]).sum(axis=1))


def energy(logits: np.ndarray, temperature: float) -> np.ndarray:
    """`E(x) = −T · log Σ exp(z_i / T)`. Energía alta = muestra menos probable.

    Idéntica a `EnergyMahalanobisGate.energyScore` antes de estandarizar.
    """
    return -temperature * log_sum_exp(logits, temperature)


def fit_energy(logits: np.ndarray, temperature: float) -> EnergyStats:
    values = energy(logits, temperature)
    std = float(values.std())
    # Una desviación nula significaría que todas las muestras tienen la misma
    # energía, lo que solo pasa con un modelo degenerado. Se protege igual: el
    # Kotlin hace lo mismo, y dividir por cero aquí propagaría NaN al artefacto.
    return EnergyStats(mean=float(values.mean()), std=std if std > 1e-6 else 1.0)


def _shared_covariance(
    embeddings: np.ndarray, labels: np.ndarray, class_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Centroides por clase y covarianza **compartida** entre todas.

    Una covarianza por clase estaría mal condicionada: con d=256, estimar una
    matriz de 65 536 entradas necesita bastantes más muestras de las que tiene la
    clase más pequeña. La covarianza compartida —cada muestra centrada en el
    centroide de SU clase— usa las 58 555 imágenes para una sola matriz.
    """
    dimension = embeddings.shape[1]
    centroids = np.zeros((class_count, dimension), dtype=np.float64)
    centered = np.empty_like(embeddings, dtype=np.float64)

    for index in range(class_count):
        mask = labels == index
        if not mask.any():
            raise ValueError(
                f"La clase {index} no tiene muestras para calcular su centroide. "
                "Sin centroide, Mahalanobis devuelve 0 y la compuerta 2 queda ciega "
                "para esa clase."
            )
        centroids[index] = embeddings[mask].mean(axis=0)
        centered[mask] = embeddings[mask] - centroids[index]

    # Denominador N − k: corrección de Bessel para las k medias estimadas.
    denominator = max(len(embeddings) - class_count, 1)
    covariance = (centered.T @ centered) / denominator
    return centroids, covariance


def _invert(covariance: np.ndarray, shrinkage: float) -> tuple[np.ndarray, float]:
    """Invierte la covarianza con encogimiento hacia la diagonal.

    Sin encogimiento, Σ puede salir singular o casi —con embeddings muy
    correlacionados es lo normal— y su inversa amplifica el ruido hasta que la
    distancia deja de significar nada. Se mezcla con un múltiplo de la identidad,
    que es el estimador de contracción clásico.
    """
    dimension = covariance.shape[0]
    average_variance = float(np.trace(covariance) / dimension)
    regularized = (1.0 - shrinkage) * covariance + shrinkage * average_variance * np.eye(dimension)

    # Cholesky en lugar de `inv` directa: falla ruidosamente si la matriz no es
    # definida positiva, en vez de devolver una inversa basura que solo se
    # detectaría por predicciones absurdas en campo.
    try:
        factor = np.linalg.cholesky(regularized)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "La covarianza compartida no es definida positiva ni tras el encogimiento. "
            "Suele indicar un embedding colapsado: revisa que el entrenamiento haya "
            "convergido antes de calibrar."
        ) from error

    identity = np.eye(dimension)
    precision = np.linalg.solve(factor.T, np.linalg.solve(factor, identity))
    # Se simetriza: la resolución numérica deja asimetrías del orden de 1e-15 que,
    # aplanadas y cuantizadas a float32, pueden hacer negativa la forma cuadrática.
    precision = (precision + precision.T) / 2.0

    return precision, float(np.linalg.cond(regularized))


def raw_distances(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    precision: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    """`d = sqrt((z − μ_pred)ᵀ Λ (z − μ_pred))`, vectorizado.

    ⚠️ Al centroide de la clase **predicha**, no al más cercano. Es lo que hace
    `EnergyMahalanobisGate.mahalanobisDistance(embedding, topIndex)`, y la
    diferencia importa: preguntar «¿está lejos de lo que el modelo cree que es?»
    detecta la ignorancia; preguntar «¿está lejos de todo?» detecta solo lo
    extremo.
    """
    delta = embeddings.astype(np.float64) - centroids[predicted]
    quadratic = np.einsum("ij,jk,ik->i", delta, precision, delta)
    # El Kotlin acota en cero por la misma razón: redondeo a float32 puede dar
    # negativos diminutos, y sqrt(negativo) es NaN — y un NaN comparado con el
    # umbral devuelve `false`, dejando pasar la muestra en silencio.
    return np.sqrt(np.maximum(quadratic, 0.0))


def fit(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_predicted: np.ndarray,
    class_count: int,
    *,
    shrinkage: float = 0.10,
) -> MahalanobisStats:
    """Calcula centroides y precisión, y pliega el factor de escala.

    Los centroides y la covarianza salen de ENTRENAMIENTO —es la distribución que
    el modelo considera «conocida»—. La escala se mide en VALIDACIÓN, sobre datos
    que el modelo no vio, que es donde la dispersión es realista.
    """
    centroids, covariance = _shared_covariance(train_embeddings, train_labels, class_count)
    precision, condition = _invert(covariance, shrinkage)

    distances = raw_distances(val_embeddings, centroids, precision, val_predicted)
    std = float(distances.std())
    scale = 1.0 / std if std > 1e-9 else 1.0

    # Plegado: (k·d)² = k²·d² ⇒ multiplicar Λ por k² escala la distancia por k.
    return MahalanobisStats(
        centroids=centroids.astype(np.float32),
        precision=(precision * scale**2).astype(np.float32),
        scale_factor=scale,
        raw_distance_mean=float(distances.mean()),
        raw_distance_std=std,
        condition_number=condition,
        shrinkage=shrinkage,
    )


def combined_score(
    logits: np.ndarray,
    embeddings: np.ndarray,
    stats: MahalanobisStats,
    energy_stats: EnergyStats,
    temperature: float,
    energy_weight: float,
) -> np.ndarray:
    """Reproduce bit a bit lo que calcula el teléfono.

    Se usa tanto para fijar el umbral como para medir el AUROC. Que sea la misma
    función en ambos casos es lo que garantiza que el percentil 95 medido aquí sea
    el percentil 95 real allí.
    """
    standardized = (energy(logits, temperature) - energy_stats.mean) / energy_stats.std
    predicted = logits.argmax(axis=1)
    distance = raw_distances(embeddings, stats.centroids, stats.precision, predicted)
    weight = float(np.clip(energy_weight, 0.0, 1.0))
    return weight * standardized + (1.0 - weight) * distance
