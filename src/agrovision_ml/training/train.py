"""Entrenamiento en dos etapas.

    Etapa 1 · tronco congelado, solo la cabeza
      Una cabeza recién inicializada produce gradientes enormes. Si el tronco
      está descongelado, esos gradientes destrozan en las primeras iteraciones
      unos pesos que costaron millones de imágenes. Congelar primero es lo que
      hace que el ajuste fino posterior parta de un punto sensato.

    Etapa 2 · ajuste fino de la parte alta, tasa 20× menor
      Las capas bajas detectan bordes y texturas: sirven igual para un gato que
      para una hoja, y no hay nada que reajustar. Las altas sí son específicas de
      ImageNet, y son las que hay que reorientar hacia el café.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from tensorflow import keras

from ..config import TrainingConfig
from ..data import dataset as data
from . import model as arch

console = Console()


@dataclass
class TrainingResult:
    exporter: keras.Model
    trainer: keras.Model
    splits: data.Splits
    history: dict[str, list[float]]
    epochs_run: int


def run(images_root: Path, config: TrainingConfig, work_dir: Path) -> TrainingResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(config.seed)

    console.rule("[bold]Reparto de datos")
    splits = data.build_splits(images_root, config)
    _report_splits(splits, config)

    train_ds = data.to_tf_dataset(splits.train, config, training=True)
    val_ds = data.to_tf_dataset(splits.val, config, training=False)
    weights = data.class_weights(splits.train, config)

    console.rule("[bold]Modelo")
    backbone, trainer, exporter = arch.build(config)
    total = trainer.count_params()
    console.print(
        f"  tronco [bold]{config.section('model', 'backbone')}[/] · "
        f"{total / 1e6:.2f} M parámetros · embedding de {config.embedding_dim} dimensiones"
    )

    checkpoint = work_dir / "mejor.weights.h5"
    history: dict[str, list[float]] = {}
    epochs_run = 0

    # ── Etapa 1 ──────────────────────────────────────────────────────────────
    warmup = config.section("training", "warmup")
    console.rule(f"[bold]Etapa 1 · cabeza ({warmup['epochs']} épocas)")
    arch.compile_for(trainer, config, float(warmup["learning_rate"]))
    stage1 = trainer.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(warmup["epochs"]),
        class_weight=weights,
        callbacks=_callbacks(config, checkpoint, restore=False),
        verbose=2,
    )
    _merge(history, stage1.history)
    epochs_run += len(stage1.history.get("loss", []))

    # ── Etapa 2 ──────────────────────────────────────────────────────────────
    finetune = config.section("training", "finetune")
    unfrozen = arch.unfreeze_top(backbone, int(finetune["unfreeze_last_layers"]))
    console.rule(f"[bold]Etapa 2 · ajuste fino ({finetune['epochs']} épocas)")
    console.print(f"  {unfrozen} capas descongeladas · normalización por lotes intacta")

    # Recompilar es obligatorio: sin ello Keras sigue usando el grafo anterior y
    # los pesos recién descongelados no reciben gradiente. El entrenamiento corre
    # sin error y no mejora nada.
    arch.compile_for(trainer, config, float(finetune["learning_rate"]))
    stage2 = trainer.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(finetune["epochs"]),
        class_weight=weights,
        callbacks=_callbacks(config, checkpoint, restore=True),
        verbose=2,
    )
    _merge(history, stage2.history)
    epochs_run += len(stage2.history.get("loss", []))

    if checkpoint.exists():
        trainer.load_weights(checkpoint)
        console.print("[green]  ✓[/] restaurados los pesos de la mejor época")

    return TrainingResult(
        exporter=exporter,
        trainer=trainer,
        splits=splits,
        history=history,
        epochs_run=epochs_run,
    )


def _callbacks(config: TrainingConfig, checkpoint: Path, *, restore: bool) -> list:
    patience = int(config.section("training", "early_stopping_patience"))
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint),
            monitor="val_acierto",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_acierto",
            mode="max",
            patience=patience,
            # En la etapa 1 no se restaura: los pesos que importan son los que
            # entran a la etapa 2, y ahí conviene el último estado, no el mejor.
            restore_best_weights=restore,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, patience - 2),
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def _merge(target: dict[str, list[float]], source: dict) -> None:
    for key, values in source.items():
        target.setdefault(key, []).extend(float(v) for v in values)


def _report_splits(splits: data.Splits, config: TrainingConfig) -> None:
    header = f"  {'clase':30}{'train':>9}{'val':>8}{'test':>8}"
    console.print(header)
    console.print("  " + "─" * (len(header) - 2))

    train = splits.train.distribution(config.class_count)
    val = splits.val.distribution(config.class_count)
    test = splits.test.distribution(config.class_count)

    for spec in config.classes:
        i = spec.index
        console.print(f"  {spec.class_id:30}{train[i]:>9}{val[i]:>8}{test[i]:>8}")

    console.print(
        f"  {'TOTAL':30}{sum(train):>9}{sum(val):>8}{sum(test):>8}"
    )

    # Un desbalance extremo no impide entrenar, pero sí cambia cómo leer las
    # métricas. Mejor decirlo antes que descubrirlo al interpretar el F1.
    ratio = max(train) / max(min(train), 1)
    if ratio > 4:
        console.print(
            f"[yellow]  ! desbalance {ratio:.1f}:1 entre la clase mayor y la menor — "
            "se compensa con `class_weight: balanced`[/]"
        )


def predict_logits_and_embeddings(
    exporter: keras.Model,
    split: data.Split,
    config: TrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pasa un conjunto por el modelo y devuelve `(logits, embeddings, etiquetas)`.

    Sin aumentación y sin barajar: el orden de las etiquetas tiene que
    corresponder fila a fila con el de las predicciones, o la matriz de confusión
    sale permutada y todo lo que se derive de ella es falso.
    """
    ds = data.to_tf_dataset(split, config, training=False)
    logits, embeddings = exporter.predict(ds, verbose=0)
    return np.asarray(logits), np.asarray(embeddings), np.asarray(split.labels, dtype=np.int64)
