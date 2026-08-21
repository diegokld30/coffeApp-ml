"""Métricas y compuertas de aceptación.

El pipeline **falla** si el modelo no llega a los mínimos de la receta. No avisa
y sigue: falla. Un modelo que no cumple no se firma, y sin firma la app se niega
a instalarlo (`Ed25519SignatureVerifier`). La barrera es de construcción, no de
criterio: no depende de que alguien se acuerde de mirar el informe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import TrainingConfig


@dataclass(frozen=True)
class ClassMetrics:
    class_id: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    macro_f1: float
    per_class: list[ClassMetrics]
    confusion: np.ndarray
    ood_auroc: float | None = None
    gate_distribution: dict[str, float] = field(default_factory=dict)

    @property
    def weakest(self) -> ClassMetrics:
        return min(self.per_class, key=lambda c: c.f1)

    def to_json(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "ood_auroc": None if self.ood_auroc is None else round(self.ood_auroc, 4),
            "per_class": [
                {
                    "class_id": c.class_id,
                    "support": c.support,
                    "precision": round(c.precision, 4),
                    "recall": round(c.recall, 4),
                    "f1": round(c.f1, 4),
                }
                for c in self.per_class
            ],
            "confusion_matrix": self.confusion.tolist(),
            "gate_distribution": {k: round(v, 4) for k, v in self.gate_distribution.items()},
        }


def evaluate(
    logits: np.ndarray,
    labels: np.ndarray,
    config: TrainingConfig,
) -> Evaluation:
    predicted = logits.argmax(axis=1)
    k = config.class_count

    confusion = np.zeros((k, k), dtype=np.int64)
    for true, guess in zip(labels, predicted, strict=True):
        confusion[true, guess] += 1

    per_class: list[ClassMetrics] = []
    for spec in config.classes:
        i = spec.index
        true_positive = int(confusion[i, i])
        predicted_positive = int(confusion[:, i].sum())
        actual_positive = int(confusion[i, :].sum())

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class.append(
            ClassMetrics(spec.class_id, actual_positive, precision, recall, f1)
        )

    return Evaluation(
        accuracy=float((predicted == labels).mean()),
        # Macro y no micro: con clases desbalanceadas 3:1, la micro esconde que la
        # clase pequeña va mal. Y la clase pequeña aquí es Phoma, una enfermedad
        # real que el productor necesita distinguir.
        macro_f1=float(np.mean([c.f1 for c in per_class])),
        per_class=per_class,
        confusion=confusion,
    )


def auroc(in_domain_scores: np.ndarray, out_domain_scores: np.ndarray) -> float:
    """AUROC por el estadístico U de Mann–Whitney, sin barrer umbrales.

    Interpretación directa: probabilidad de que una muestra fuera de dominio
    tomada al azar reciba mayor puntuación OOD que una de dentro tomada al azar.
    0,5 es una moneda; 1,0 es separación perfecta.

    El cálculo por rangos maneja los empates correctamente —les asigna el rango
    medio—, cosa que la aproximación por trapecios sobre una rejilla de umbrales
    no hace, y con puntuaciones cuantizadas hay empates de sobra.
    """
    if len(in_domain_scores) == 0 or len(out_domain_scores) == 0:
        raise ValueError("Hacen falta muestras de ambos lados para calcular AUROC")

    combined = np.concatenate([in_domain_scores, out_domain_scores])
    order = combined.argsort()
    ranks = np.empty(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=np.float64)

    # Rango medio para los empates.
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]

    n_out = len(out_domain_scores)
    n_in = len(in_domain_scores)
    rank_sum_out = ranks[n_in:].sum()
    u = rank_sum_out - n_out * (n_out + 1) / 2.0
    return float(u / (n_in * n_out))


class AcceptanceError(RuntimeError):
    """El modelo no alcanza los mínimos. Nunca se firma ni se publica."""


@dataclass(frozen=True)
class AcceptanceReport:
    checks: list[tuple[str, bool, str]]

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    def raise_if_failed(self) -> None:
        if self.passed:
            return
        failures = "\n".join(f"  ✗ {name}: {detail}" for name, ok, detail in self.checks if not ok)
        raise AcceptanceError(
            "El modelo no cumple los criterios de aceptación de la receta:\n" + failures
        )


def check_acceptance(
    evaluation: Evaluation,
    config: TrainingConfig,
    *,
    size_mb: float | None = None,
    int8_f1_drop: float | None = None,
) -> AcceptanceReport:
    limits = config.section("acceptance")
    checks: list[tuple[str, bool, str]] = []

    min_macro = float(limits["min_macro_f1"])
    checks.append(
        (
            "F1 macro",
            evaluation.macro_f1 >= min_macro,
            f"{evaluation.macro_f1:.4f} frente al mínimo {min_macro}",
        )
    )

    min_class = float(limits["min_class_f1"])
    weakest = evaluation.weakest
    checks.append(
        (
            "F1 de la clase más débil",
            weakest.f1 >= min_class,
            f"{weakest.class_id} = {weakest.f1:.4f} frente al mínimo {min_class}",
        )
    )

    # El AUROC solo se comprueba si hay muestras fuera de dominio con las que
    # medirlo. Sin ellas se deja constancia en vez de inventar un número: un
    # `null` honesto en metrics.json impide después promover a producción.
    if evaluation.ood_auroc is not None:
        min_auroc = float(limits["min_ood_auroc"])
        checks.append(
            (
                "AUROC de fuera de dominio",
                evaluation.ood_auroc >= min_auroc,
                f"{evaluation.ood_auroc:.4f} frente al mínimo {min_auroc}",
            )
        )

    if size_mb is not None:
        maximum = float(limits["max_size_mb"])
        checks.append(
            ("Tamaño del artefacto", size_mb <= maximum, f"{size_mb:.2f} MB frente a {maximum} MB")
        )

    if int8_f1_drop is not None:
        maximum = float(limits["max_int8_f1_drop"])
        checks.append(
            (
                "Pérdida por cuantización",
                int8_f1_drop <= maximum,
                f"{int8_f1_drop:.4f} frente al máximo {maximum}",
            )
        )

    return AcceptanceReport(checks)
