"""Escalado por temperatura (Guo et al., 2017).

Una red entrenada con entropía cruzada sale **sistemáticamente sobreconfiada**:
dice 0,97 cuando acierta el 78 % de las veces. No es un defecto del
entrenamiento, es una consecuencia de minimizar la entropía cruzada sobre
etiquetas duras.

Importa porque la compuerta 3 compara la probabilidad máxima contra un umbral τ.
Si el 0,97 real vale 0,78, cualquier τ que se elija está midiendo otra cosa.

La corrección es un único escalar `T` que divide los logits antes del softmax:

    p_i = softmax(z_i / T)

`T` se ajusta minimizando la log-verosimilitud negativa **sobre validación**, no
sobre entrenamiento. Sobre entrenamiento daría T≈1 porque ahí la red sí acierta
lo que dice — y ese es justo el error que se quiere medir.

Propiedad clave: dividir todos los logits por la misma constante **no cambia cuál
es el máximo**. La exactitud queda idéntica; solo se corrige la confianza.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float

    @property
    def improved(self) -> bool:
        return self.nll_after <= self.nll_before


def _log_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    # Estabilización idéntica a la de `EnergyMahalanobisGate.softmaxWithTemperature`:
    # sin restar el máximo, exp() desborda con logits por encima de ~709 en
    # doble precisión y de ~88 en simple.
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    log_probabilities = _log_softmax(logits, temperature)
    return float(-log_probabilities[np.arange(len(labels)), labels].mean())


def expected_calibration_error(
    logits: np.ndarray, labels: np.ndarray, temperature: float, bins: int = 15
) -> float:
    """ECE: brecha media entre confianza declarada y acierto real.

    Se agrupan las predicciones en `bins` tramos de confianza y en cada uno se
    compara la confianza media con la exactitud media. Es la métrica que traduce
    «está sobreconfiado» a un número que se puede poner en un informe.
    """
    probabilities = np.exp(_log_softmax(logits, temperature))
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        # El primer tramo incluye su borde inferior; el resto no, para que cada
        # predicción caiga en exactamente un tramo.
        mask = (confidence > low) & (confidence <= high) if low > 0 else (confidence <= high)
        if not mask.any():
            continue
        weight = mask.mean()
        error += weight * abs(correct[mask].mean() - confidence[mask].mean())

    return float(error)


def fit(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    low: float = 0.05,
    high: float = 10.0,
) -> TemperatureFit:
    """Encuentra la `T` que minimiza la NLL en validación.

    Búsqueda en dos fases —rejilla gruesa y después sección áurea— en lugar de
    descenso por gradiente. Es un problema escalar: la rejilla evita quedarse en
    un mínimo local si la curva tuviera más de uno, y la sección áurea afina sin
    necesidad de derivadas ni de otra dependencia.
    """
    if logits.ndim != 2:
        raise ValueError(f"Se esperaban logits de rango 2 y llegaron de rango {logits.ndim}")
    if len(logits) != len(labels):
        raise ValueError(f"{len(logits)} logits frente a {len(labels)} etiquetas")

    grid = np.geomspace(low, high, num=60)
    losses = [negative_log_likelihood(logits, labels, t) for t in grid]
    best = int(np.argmin(losses))

    # Intervalo alrededor del mejor punto de la rejilla, acotado a los extremos.
    left = grid[max(best - 1, 0)]
    right = grid[min(best + 1, len(grid) - 1)]

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        if right - left < 1e-4:
            break
        a = right - phi * (right - left)
        b = left + phi * (right - left)
        if negative_log_likelihood(logits, labels, a) < negative_log_likelihood(logits, labels, b):
            right = b
        else:
            left = a

    temperature = float((left + right) / 2.0)

    return TemperatureFit(
        temperature=temperature,
        nll_before=negative_log_likelihood(logits, labels, 1.0),
        nll_after=negative_log_likelihood(logits, labels, temperature),
        ece_before=expected_calibration_error(logits, labels, 1.0),
        ece_after=expected_calibration_error(logits, labels, temperature),
    )
