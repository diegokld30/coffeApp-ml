"""Lectura y validación de la receta de entrenamiento.

La configuración se valida **al cargar**, no al usarse. Un error tipográfico en
`coffee_v1.yaml` tiene que reventar en el segundo cero, no cuarenta minutos
después de empezar a entrenar en una GPU alquilada.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Receta inválida. Siempre indica qué campo y por qué."""


@dataclass(frozen=True)
class ClassSpec:
    """Una clase del modelo.

    `index` es la posición en el vector de logits y se asigna por el orden de la
    lista en el YAML. Es el único vínculo entre lo que dice el grafo y lo que
    muestra la app.
    """

    index: int
    source_dir: str
    class_id: str
    display_name: str
    pest_id: str | None

    @property
    def is_pest(self) -> bool:
        return self.pest_id is not None


@dataclass(frozen=True)
class TrainingConfig:
    version: str
    channel: str
    seed: int
    raw: dict[str, Any] = field(repr=False)
    classes: tuple[ClassSpec, ...] = ()

    # ── Accesos tipados a las secciones que más se usan ──────────────────────

    @property
    def image_size(self) -> int:
        return int(self.raw["data"]["image_size"])

    @property
    def embedding_dim(self) -> int:
        return int(self.raw["model"]["embedding_dim"])

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def dataset_ids(self) -> list[str]:
        return [s["dataset_id"] for s in self.raw["data"]["sources"]]

    def section(self, *keys: str) -> Any:
        node: Any = self.raw
        for key in keys:
            node = node[key]
        return node


_REQUIRED_SECTIONS = (
    "data",
    "classes",
    "model",
    "training",
    "augmentation",
    "calibration",
    "acceptance",
    "export",
)


def load(path: str | Path) -> TrainingConfig:
    """Carga la receta y la valida. Lanza `ConfigError` con el motivo exacto."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"No existe la receta {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} no contiene un mapa YAML en la raíz")

    missing = [s for s in _REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ConfigError(f"{path}: faltan las secciones {', '.join(missing)}")

    classes = _parse_classes(raw["classes"], path)
    _validate_split(raw["data"].get("split", {}), path)

    return TrainingConfig(
        version=str(raw["version"]),
        channel=str(raw.get("channel", "draft")),
        seed=int(raw.get("seed", 0)),
        raw=raw,
        classes=classes,
    )


def _parse_classes(node: Any, path: Path) -> tuple[ClassSpec, ...]:
    if not isinstance(node, list) or not node:
        raise ConfigError(f"{path}: `classes` tiene que ser una lista no vacía")

    classes = tuple(
        ClassSpec(
            index=i,
            source_dir=str(entry["source_dir"]),
            class_id=str(entry["class_id"]),
            display_name=str(entry["display_name"]),
            pest_id=entry.get("pest_id") or None,
        )
        for i, entry in enumerate(node)
    )

    # Un `class_id` repetido produce dos logits con el mismo nombre: la app
    # mostraría una ficha y el modelo estaría prediciendo la otra.
    ids = [c.class_id for c in classes]
    if len(set(ids)) != len(ids):
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        raise ConfigError(f"{path}: `class_id` repetido: {', '.join(duplicated)}")

    dirs = [c.source_dir for c in classes]
    if len(set(dirs)) != len(dirs):
        raise ConfigError(f"{path}: dos clases apuntan a la misma carpeta de origen")

    return classes


def _validate_split(split: Any, path: Path) -> None:
    if not isinstance(split, dict):
        raise ConfigError(f"{path}: `data.split` tiene que ser un mapa")

    try:
        total = sum(float(split[k]) for k in ("train", "val", "test"))
    except KeyError as error:
        raise ConfigError(f"{path}: falta `data.split.{error.args[0]}`") from error

    # Tolerancia por la aritmética de coma flotante: 0.70 + 0.15 + 0.15 no da
    # exactamente 1.0 en binario.
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"{path}: las proporciones de `data.split` suman {total}, no 1.0")
