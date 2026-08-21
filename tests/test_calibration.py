"""Pruebas de la calibración.

La prueba que justifica el archivo entero es `test_espejo_kotlin`: reimplementa
`EnergyMahalanobisGate.kt` línea a línea, con sus bucles, y comprueba que da el
mismo número que el pipeline. Si alguien cambia una fórmula en un lado y no en el
otro, el umbral calibrado aquí deja de significar nada allí — y el síntoma sería
que la app empieza a decir «no la reconozco» a hojas perfectamente normales, sin
ningún error en los registros.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from agrovision_ml import config
from agrovision_ml.calibration import mahalanobis as maha
from agrovision_ml.calibration import temperature as temp
from agrovision_ml.calibration import thresholds
from agrovision_ml.evaluation import metrics

CONFIG = "configs/coffee_v1.yaml"


def _config(**overrides) -> config.TrainingConfig:
    base = config.load(CONFIG)
    raw = copy.deepcopy(base.raw)
    for section, values in overrides.items():
        raw[section].update(values)
    return config.TrainingConfig(base.version, base.channel, base.seed, raw, base.classes)


def _world(cfg: config.TrainingConfig, seed: int, *, sharpness: float, noise: float):
    """Genera un mundo sintético con `class_count` gaussianas separadas.

    `sharpness` controla cuánto se estiran los logits: valores altos simulan la
    sobreconfianza real de una red entrenada con entropía cruzada.
    """
    rng = np.random.default_rng(seed)
    k, d = cfg.class_count, cfg.embedding_dim
    centers = rng.normal(0.0, 3.0, size=(k, d))

    def draw(n: int, shift: float = 0.0):
        labels = rng.integers(0, k, n)
        embeddings = centers[labels] + rng.normal(0.0, noise, size=(n, d)) + shift
        distances = ((embeddings[:, None, :] - centers[None]) ** 2).sum(-1)
        logits = -sharpness * (distances - distances.mean(1, keepdims=True))
        return embeddings.astype(np.float32), logits.astype(np.float32), labels

    return draw


# ── Temperatura ──────────────────────────────────────────────────────────────


def _overconfident(seed: int, n: int, classes: int, *, gap: float, error_rate: float):
    """Simula la patología real: la red declara 0,95 y acierta el 78 %.

    ⚠️ El mundo gaussiano de `_world` **no sirve** para esto. Con clases bien
    separadas en 128 dimensiones el modelo acierta el 100 % y sus logits son
    coherentes con ese acierto, así que no hay sobreconfianza que corregir y la T
    óptima se va al extremo inferior del intervalo. La prueba pasaría sin medir
    nada.

    Aquí se construye la discrepancia a mano: se generan logits con una brecha
    grande —confianza alta— y después se corrompe una fracción de las etiquetas
    verdaderas. El resultado es exactamente lo que produce el entrenamiento con
    entropía cruzada sobre etiquetas duras.
    """
    rng = np.random.default_rng(seed)
    predicted = rng.integers(0, classes, n)

    logits = rng.normal(0.0, 1.0, size=(n, classes))
    logits[np.arange(n), predicted] += gap

    labels = predicted.copy()
    wrong = rng.random(n) < error_rate
    # A las equivocadas se les asigna otra clase cualquiera distinta de la predicha.
    displacement = rng.integers(1, classes, wrong.sum())
    labels[wrong] = (predicted[wrong] + displacement) % classes

    return logits.astype(np.float32), labels


def test_temperatura_corrige_sobreconfianza():
    """Con una red sobreconfiada, T > 1 y el ECE baja."""
    cfg = _config()
    logits, labels = _overconfident(3, 2000, cfg.class_count, gap=6.0, error_rate=0.22)

    exactitud = (logits.argmax(1) == labels).mean()
    confianza = temp.expected_calibration_error(logits, labels, 1.0)
    assert 0.70 < exactitud < 0.85, f"El montaje no simula el caso real (exactitud {exactitud})"
    assert confianza > 0.05, "Sin brecha inicial no hay nada que calibrar"

    fit = temp.fit(logits, labels)

    assert fit.temperature > 1.0, (
        f"T = {fit.temperature:.3f}. Con logits inflados debe aplanarlos, no afilarlos."
    )
    assert fit.nll_after < fit.nll_before
    assert fit.ece_after < fit.ece_before, (
        f"ECE {fit.ece_before:.4f} → {fit.ece_after:.4f}: la calibración no mejoró"
    )


def test_temperatura_no_altera_la_exactitud():
    """Dividir todos los logits por una constante no mueve el argmax.

    Es la propiedad que hace segura la calibración: corrige la confianza sin
    tocar ni una predicción.
    """
    cfg = _config()
    logits, labels = _overconfident(5, 1200, cfg.class_count, gap=5.0, error_rate=0.20)

    fit = temp.fit(logits, labels)
    antes = (logits.argmax(1) == labels).mean()
    despues = ((logits / fit.temperature).argmax(1) == labels).mean()

    assert antes == despues


def test_temperatura_no_toca_una_red_ya_calibrada():
    """Si la confianza declarada ya corresponde al acierto, T debe quedar ≈ 1.

    Es el control negativo. Sin él, un ajuste que siempre devolviera T > 1
    pasaría las dos pruebas anteriores.
    """
    cfg = _config()
    rng = np.random.default_rng(23)
    n, k = 3000, cfg.class_count

    # Se muestrea la etiqueta DESDE el softmax de los propios logits: por
    # construcción, la probabilidad declarada es la frecuencia real de acierto.
    logits = rng.normal(0.0, 1.6, size=(n, k))
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(k, p=row) for row in probabilities])

    fit = temp.fit(logits.astype(np.float32), labels)

    assert fit.temperature == pytest.approx(1.0, abs=0.15), (
        f"T = {fit.temperature:.3f} sobre una red ya calibrada: el ajuste está sesgado."
    )


# ── Mahalanobis y umbrales ───────────────────────────────────────────────────


def test_escala_plegada_deja_desviacion_unitaria():
    """El factor plegado en la matriz debe dar distancias de desviación ≈ 1.

    Sin esto, `energy_weight: 0.5` sería mentira: Mahalanobis cruda ronda √d y
    aplastaría por completo la contribución de la energía, que está estandarizada.
    """
    cfg = _config()
    draw = _world(cfg, seed=7, sharpness=0.5, noise=1.0)
    tr_z, _, tr_y = draw(2000)
    va_z, va_logits, _ = draw(600)

    stats = maha.fit(tr_z, tr_y, va_z, va_logits.argmax(1), cfg.class_count)
    distances = maha.raw_distances(va_z, stats.centroids, stats.precision, va_logits.argmax(1))

    assert distances.std() == pytest.approx(1.0, abs=0.05)


def test_umbral_rechaza_el_percentil_pedido():
    cfg = _config(calibration={"ood_percentile": 0.95})
    draw = _world(cfg, seed=11, sharpness=0.5, noise=1.0)
    tr_z, _, tr_y = draw(2000)
    va_z, va_logits, va_y = draw(600)

    calibration = thresholds.fit(
        cfg,
        train_embeddings=tr_z,
        train_labels=tr_y,
        val_logits=va_logits,
        val_embeddings=va_z,
        val_labels=va_y,
    )

    assert calibration.in_domain_rejected == pytest.approx(0.05, abs=0.02)


def test_detecta_lo_que_nunca_vio():
    """AUROC alto frente a muestras desplazadas fuera de todos los centroides.

    Es la razón de ser de la compuerta 2 y del error prohibido nº 4: un umbral de
    softmax no distingue estas muestras, porque la red les asigna con aplomo la
    clase visualmente más parecida.
    """
    cfg = _config()
    draw = _world(cfg, seed=13, sharpness=0.5, noise=1.0)
    tr_z, _, tr_y = draw(2000)
    va_z, va_logits, va_y = draw(600)
    ood_z, ood_logits, _ = draw(400, shift=7.0)

    calibration = thresholds.fit(
        cfg,
        train_embeddings=tr_z,
        train_labels=tr_y,
        val_logits=va_logits,
        val_embeddings=va_z,
        val_labels=va_y,
    )

    dentro = thresholds.simulate_gates(calibration, va_logits, va_z)["scores"]
    fuera = thresholds.simulate_gates(calibration, ood_logits, ood_z)["scores"]

    assert metrics.auroc(dentro, fuera) > 0.85


def test_covarianza_colapsada_falla_ruidosamente():
    """Un embedding degenerado tiene que reventar aquí, no en el teléfono."""
    cfg = _config()
    embeddings = np.zeros((200, cfg.embedding_dim), dtype=np.float32)
    labels = np.arange(200) % cfg.class_count

    with pytest.raises(ValueError, match="definida positiva"):
        maha.fit(embeddings, labels, embeddings, labels, cfg.class_count)


# ── El espejo ────────────────────────────────────────────────────────────────


def _kotlin_combined_score(payload: dict, logits: list[float], embedding: list[float]) -> float:
    """Traducción literal de `EnergyMahalanobisGate.evaluate`, bucles incluidos.

    Deliberadamente ingenua y sin numpy: si se vectorizara, se estaría probando
    numpy contra numpy en lugar de probar el contrato contra el Kotlin real.
    """
    temperature = payload["temperature"]
    weight = payload["energy_weight"]
    energy_mean = payload["energy_mean"]
    energy_std = payload["energy_std_dev"]
    centroids = payload["class_centroids"]
    precision = payload["precision_matrix"]

    # energyScore
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    total = sum(math.exp(value - maximum) for value in scaled)
    energy = -temperature * (maximum + math.log(total))
    energy = (energy - energy_mean) / (energy_std if energy_std > 1e-6 else 1.0)

    # mahalanobisDistance, al centroide de la clase PREDICHA
    top = max(range(len(logits)), key=lambda i: logits[i])
    centroid = centroids[top]
    dimension = len(embedding)
    delta = [embedding[i] - centroid[i] for i in range(dimension)]

    quadratic = 0.0
    for row in range(dimension):
        accumulated = 0.0
        offset = row * dimension
        for column in range(dimension):
            accumulated += precision[offset + column] * delta[column]
        quadratic += delta[row] * accumulated
    distance = math.sqrt(max(quadratic, 0.0))

    clamped = min(max(weight, 0.0), 1.0)
    return clamped * energy + (1.0 - clamped) * distance


def test_espejo_kotlin():
    """El pipeline y el teléfono tienen que calcular el mismo número."""
    cfg = _config()
    draw = _world(cfg, seed=17, sharpness=0.5, noise=1.0)
    tr_z, _, tr_y = draw(1500)
    va_z, va_logits, va_y = draw(300)

    calibration = thresholds.fit(
        cfg,
        train_embeddings=tr_z,
        train_labels=tr_y,
        val_logits=va_logits,
        val_embeddings=va_z,
        val_labels=va_y,
    )
    esperado = thresholds.simulate_gates(calibration, va_logits, va_z)["scores"]

    # Se serializa y se relee: es exactamente el camino que recorre el dato hasta
    # el teléfono, con su recorte a siete cifras significativas incluido.
    payload = json.loads(json.dumps(calibration.to_json()))

    for row in range(25):
        obtenido = _kotlin_combined_score(
            payload,
            [float(v) for v in va_logits[row]],
            [float(v) for v in va_z[row]],
        )
        assert obtenido == pytest.approx(float(esperado[row]), abs=1e-3), (
            f"Divergen en la muestra {row}: Python {esperado[row]:.6f} "
            f"frente a Kotlin {obtenido:.6f}. Alguien tocó una fórmula en un solo lado."
        )


def test_forma_del_artefacto_es_la_que_valida_android():
    """`ArtifactInstaller` rechaza una calibración incoherente con las clases."""
    cfg = _config()
    draw = _world(cfg, seed=19, sharpness=0.5, noise=1.0)
    tr_z, _, tr_y = draw(1200)
    va_z, va_logits, va_y = draw(300)

    payload = thresholds.fit(
        cfg,
        train_embeddings=tr_z,
        train_labels=tr_y,
        val_logits=va_logits,
        val_embeddings=va_z,
        val_labels=va_y,
    ).to_json()

    dimension = cfg.embedding_dim
    assert len(payload["class_centroids"]) == cfg.class_count
    assert all(len(row) == dimension for row in payload["class_centroids"])
    assert len(payload["precision_matrix"]) == dimension * dimension
    assert payload["temperature"] > 0.0

    for key in (
        "version",
        "temperature",
        "confidence_threshold",
        "margin_threshold",
        "ood_threshold",
        "energy_weight",
        "energy_mean",
        "energy_std_dev",
        "class_centroids",
        "precision_matrix",
    ):
        assert key in payload, f"Falta «{key}»: `CalibrationJson` no podría deserializarlo"
