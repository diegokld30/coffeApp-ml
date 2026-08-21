"""Interfaz de línea de comandos.

Un solo comando lleva de cero a artefacto firmado:

    agrovision-ml train --config configs/coffee_v1.yaml

Descarga los datos si faltan, entrena, calibra, evalúa, exporta, verifica la
paridad, comprueba los criterios de aceptación y firma. Si algo no cumple, se
detiene **antes** de firmar: un modelo sin firma no lo instala ningún teléfono.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from . import config as configuration
from .calibration import thresholds
from .data import dataset as data
from .data import download
from .evaluation import metrics
from .export import to_tflite, verify_parity
from .signing import sign_artifact
from .training import train as training

app = typer.Typer(
    add_completion=False,
    help="Entrenamiento y publicación de modelos de AgroVisión.",
    no_args_is_help=True,
)
console = Console()

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "coffee_v1.yaml"
DEFAULT_DATA = ROOT / "data"
DEFAULT_ARTIFACTS = ROOT / "artifacts"
DEFAULT_KEY = ROOT.parent / "infra" / "secrets" / "model_signing_private.pem"


# ── Comandos ─────────────────────────────────────────────────────────────────


@app.command("download")
def download_command(
    config_path: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    data_dir: Path = typer.Option(DEFAULT_DATA, "--data"),
) -> None:
    """Descarga JMuBEN y JMuBEN2 desde Mendeley (~1,75 GB, reanudable)."""
    cfg = configuration.load(config_path)
    images = download.ensure(cfg.dataset_ids, data_dir)

    table = Table(title="Imágenes por carpeta", header_style="bold")
    table.add_column("carpeta")
    table.add_column("imágenes", justify="right")
    for name, count in sorted(download.count_by_class(images).items()):
        table.add_row(name, f"{count:,}")
    console.print(table)


@app.command("train")
def train_command(
    config_path: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    data_dir: Path = typer.Option(DEFAULT_DATA, "--data"),
    output_dir: Path = typer.Option(DEFAULT_ARTIFACTS, "--out"),
    key_path: Path = typer.Option(DEFAULT_KEY, "--key", help="Clave privada Ed25519"),
    ood_dir: Path | None = typer.Option(
        None,
        "--ood",
        help="Carpeta con imágenes FUERA de dominio para medir el AUROC. "
        "Sin ella el AUROC queda en null y el modelo no puede promoverse a producción.",
    ),
    skip_signing: bool = typer.Option(False, "--sin-firma", help="No firmar (solo pruebas)"),
) -> None:
    """Pipeline completo: datos → modelo → calibración → artefacto firmado."""
    cfg = configuration.load(config_path)
    artifact_dir = output_dir / f"v{cfg.version}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold]AgroVisión · modelo v{cfg.version} ({cfg.channel})")

    images_root = data_dir / "images"
    if not images_root.exists() or not any(images_root.iterdir()):
        console.print("[yellow]No hay imágenes en disco — descargando primero.[/]")
        images_root = download.ensure(cfg.dataset_ids, data_dir)

    # ── Entrenamiento ────────────────────────────────────────────────────────
    result = training.run(images_root, cfg, artifact_dir / "trabajo")

    # ── Calibración ──────────────────────────────────────────────────────────
    console.rule("[bold]Calibración")
    console.print("  pasando entrenamiento y validación por el modelo…")

    _, train_embeddings, train_labels = training.predict_logits_and_embeddings(
        result.exporter, result.splits.train, cfg
    )
    val_logits, val_embeddings, val_labels = training.predict_logits_and_embeddings(
        result.exporter, result.splits.val, cfg
    )

    calibration = thresholds.fit(
        cfg,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_logits=val_logits,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
    )
    _report_calibration(calibration)

    # ── Evaluación en float ──────────────────────────────────────────────────
    console.rule("[bold]Evaluación · modelo float")
    test_logits, test_embeddings, test_labels = training.predict_logits_and_embeddings(
        result.exporter, result.splits.test, cfg
    )
    evaluation = metrics.evaluate(test_logits, test_labels, cfg)
    _report_evaluation(evaluation, cfg)

    gates = thresholds.simulate_gates(calibration, test_logits, test_embeddings)
    gate_distribution = {
        key: float(gates[key].mean())
        for key in ("identified", "low_confidence", "ambiguous", "ood")
    }
    _report_gates(gate_distribution)

    # ── AUROC de fuera de dominio ────────────────────────────────────────────
    ood_auroc = None
    if ood_dir is not None:
        ood_auroc = _measure_ood(ood_dir, cfg, result, calibration, gates["scores"])

    evaluation = metrics.Evaluation(
        accuracy=evaluation.accuracy,
        macro_f1=evaluation.macro_f1,
        per_class=evaluation.per_class,
        confusion=evaluation.confusion,
        ood_auroc=ood_auroc,
        gate_distribution=gate_distribution,
    )

    # ── Exportación ──────────────────────────────────────────────────────────
    console.rule("[bold]Exportación a LiteRT")
    representative = data.representative_batches(
        result.splits.train, cfg, int(cfg.section("export", "representative_samples"))
    )
    exported = to_tflite.convert(result.exporter, representative, artifact_dir / "model.tflite")
    console.print(
        f"  {exported.size_mb:.2f} MB · entrada {exported.input_dtype} · "
        f"salidas {[s for _, s in exported.output_shapes]}"
        + ("" if exported.fully_quantized else " [yellow](cuantización parcial)[/]")
    )

    # ── Paridad y pérdida por cuantización ───────────────────────────────────
    console.rule("[bold]Verificación de paridad")
    sample = _materialize(result.splits.test, cfg, limit=200)
    parity = verify_parity.verify(
        result.exporter,
        exported.path,
        sample["images"],
        cfg.class_count,
        float(cfg.section("export", "parity_tolerance")),
    )
    console.print(parity.describe())

    int8_logits, _ = to_tflite.run_tflite(exported.path, sample["images"], cfg.class_count)
    int8_evaluation = metrics.evaluate(int8_logits, sample["labels"], cfg)
    float_subset = metrics.evaluate(
        np.asarray(result.exporter.predict(sample["images"], verbose=0)[0]),
        sample["labels"],
        cfg,
    )
    f1_drop = float_subset.macro_f1 - int8_evaluation.macro_f1
    console.print(
        f"  F1 macro float {float_subset.macro_f1:.4f} → INT8 "
        f"{int8_evaluation.macro_f1:.4f} (caída {f1_drop:+.4f})"
    )

    # ── Aceptación ───────────────────────────────────────────────────────────
    console.rule("[bold]Criterios de aceptación")
    report = metrics.check_acceptance(
        evaluation, cfg, size_mb=exported.size_mb, int8_f1_drop=max(f1_drop, 0.0)
    )
    for name, ok, detail in report.checks:
        console.print(f"  {'[green]✓[/]' if ok else '[red]✗[/]'} {name}: {detail}")
    if not parity.passed:
        console.print("  [red]✗[/] Paridad Keras ↔ TFLite fuera de tolerancia")

    # ── Artefactos ───────────────────────────────────────────────────────────
    _write_labels(artifact_dir, cfg)
    calibration.write(artifact_dir / "calibration.json")
    _write_metrics(
        artifact_dir,
        cfg,
        evaluation=evaluation,
        calibration=calibration,
        parity=parity,
        exported=exported,
        result=result,
        f1_drop=f1_drop,
        acceptance=report,
    )

    report.raise_if_failed()
    if not parity.passed:
        raise verify_parity.ParityError(
            "La paridad Keras ↔ TFLite no cumple. Un artefacto que no reproduce al "
            "modelo entrenado no se firma."
        )

    # ── Firma ────────────────────────────────────────────────────────────────
    if skip_signing:
        console.print("[yellow]  ! sin firmar (--sin-firma): ningún teléfono lo aceptará[/]")
    else:
        console.rule("[bold]Firma Ed25519")
        signed = sign_artifact.sign(
            artifact_dir, key_path, version=cfg.version, channel=cfg.channel
        )
        manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        if not sign_artifact.verify(artifact_dir, manifest["signing_public_key"]):
            raise sign_artifact.SigningError(
                "La firma recién escrita no verifica. No se publica."
            )
        console.print(f"  sha256 del modelo  {signed.model_sha256}")
        console.print(f"  digest del conjunto {signed.set_digest_hex}")
        console.print("[green]  ✓[/] firma verificada con la clave pública correspondiente")

    console.rule("[bold green]Listo")
    console.print(f"  artefactos en [bold]{artifact_dir}[/]")
    if ood_auroc is None:
        console.print(
            "[yellow]  ! AUROC sin medir: este modelo puede ir a `draft` o `internal`, "
            "no a `production`. Pásale --ood con imágenes que NO sean de las cinco "
            "clases para medirlo.[/]"
        )


@app.command("keys")
def keys_command(
    out_dir: Path = typer.Option(ROOT.parent / "infra" / "secrets", "--out"),
) -> None:
    """Genera el par Ed25519 de firma de modelos."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    out_dir.mkdir(parents=True, exist_ok=True)
    private_path = out_dir / "model_signing_private.pem"

    if private_path.exists():
        console.print(f"[red]Ya existe {private_path}.[/]")
        console.print(
            "Sobrescribirla dejaría [bold]huérfanos a todos los modelos ya publicados[/]: "
            "los teléfonos rechazarían las actualizaciones firmadas con la clave nueva. "
            "Bórrala a mano solo si estás seguro."
        )
        raise typer.Exit(code=1)

    key = Ed25519PrivateKey.generate()
    private_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    private_path.chmod(0o600)

    public_b64 = sign_artifact.public_key_base64(key)
    (out_dir / "model_signing_public.b64").write_text(public_b64 + "\n", encoding="utf-8")

    console.print(f"[green]✓[/] clave privada en {private_path} (modo 600)")
    console.print("\nPega esta línea en [bold]mobile-android/gradle.properties[/]:\n")
    console.print(f"  agrovision.model.signingPublicKeyBase64={public_b64}\n")
    console.print(
        "[yellow]La clave privada NUNCA entra al repositorio ni a un contenedor de "
        "producción. Vive en esta máquina o en un HSM (§15.6).[/]"
    )


@app.command("verify")
def verify_command(
    artifact_dir: Path = typer.Argument(..., help="Carpeta con los cinco artefactos"),
    public_key: str = typer.Option(..., "--public-key", help="Clave pública en Base64"),
) -> None:
    """Verifica la firma exactamente como lo hace el teléfono."""
    valid = sign_artifact.verify(artifact_dir, public_key)
    model_hash, digest = sign_artifact.build_set_digest(artifact_dir)
    console.print(f"  sha256 del modelo   {model_hash}")
    console.print(f"  digest del conjunto {digest.hex()}")
    if valid:
        console.print("[green]✓ Firma válida[/]")
    else:
        console.print("[red]✗ Firma INVÁLIDA — la app rechazaría este artefacto[/]")
        raise typer.Exit(code=1)


# ── Auxiliares ───────────────────────────────────────────────────────────────


def _materialize(split: data.Split, cfg: configuration.TrainingConfig, limit: int) -> dict:
    """Carga un subconjunto en memoria como arreglos numpy, sin aumentación."""
    indices = np.linspace(0, len(split) - 1, num=min(limit, len(split)), dtype=int)
    subset = data.Split([split.paths[i] for i in indices], [split.labels[i] for i in indices])
    dataset = data.to_tf_dataset(subset, cfg, training=False, batch_size=32)

    images = np.concatenate([batch.numpy() for batch, _ in dataset], axis=0)
    return {"images": images, "labels": np.asarray(subset.labels, dtype=np.int64)}


def _measure_ood(
    ood_dir: Path,
    cfg: configuration.TrainingConfig,
    result: training.TrainingResult,
    calibration: thresholds.Calibration,
    in_domain_scores: np.ndarray,
) -> float | None:
    """AUROC contra imágenes que el modelo nunca vio y que no son ninguna clase."""
    suffixes = {".jpg", ".jpeg", ".png"}
    paths = sorted(str(p) for p in ood_dir.rglob("*") if p.suffix.lower() in suffixes)
    if not paths:
        console.print(f"[yellow]  ! {ood_dir} no tiene imágenes; el AUROC queda sin medir[/]")
        return None

    console.print(f"  {len(paths)} imágenes fuera de dominio desde {ood_dir}")
    # Las etiquetas son irrelevantes aquí —solo importa la puntuación OOD— pero
    # la canalización las exige, así que se rellena con ceros.
    split = data.Split(paths, [0] * len(paths))
    logits, embeddings, _ = training.predict_logits_and_embeddings(result.exporter, split, cfg)
    out_scores = thresholds.simulate_gates(calibration, logits, embeddings)["scores"]

    value = metrics.auroc(in_domain_scores, out_scores)
    detected = float((out_scores > calibration.ood_threshold).mean())
    console.print(f"  AUROC {value:.4f} · detectadas como desconocidas {detected:.1%}")
    return value


def _write_labels(directory: Path, cfg: configuration.TrainingConfig) -> None:
    payload = {
        "version": cfg.version,
        "class_count": cfg.class_count,
        "embedding_dim": cfg.embedding_dim,
        "classes": [
            {
                "index": spec.index,
                "class_id": spec.class_id,
                "display_name": spec.display_name,
                "pest_id": spec.pest_id,
            }
            for spec in cfg.classes
        ],
    }
    (directory / "labels.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_metrics(
    directory: Path,
    cfg: configuration.TrainingConfig,
    *,
    evaluation: metrics.Evaluation,
    calibration: thresholds.Calibration,
    parity: verify_parity.ParityReport,
    exported: to_tflite.ExportResult,
    result: training.TrainingResult,
    f1_drop: float,
    acceptance: metrics.AcceptanceReport,
) -> None:
    payload = {
        "version": cfg.version,
        "channel": cfg.channel,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": cfg.seed,
        "backbone": cfg.section("model", "backbone"),
        "epochs_run": result.epochs_run,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "dataset": {
            "name": "JMuBEN + JMuBEN2",
            "license": "CC BY 4.0",
            "citation": (
                "Jepkoech, J.; Kenduiywo, B.; Mugo, D.; Chebet, E. (2021). "
                "Arabica coffee leaf images dataset for coffee leaf disease detection "
                "and classification. Data in Brief 36, 107142. "
                "DOI 10.17632/t2r6rszp5c.1 y 10.17632/tgv3zb82nd.1"
            ),
            "train": len(result.splits.train),
            "val": len(result.splits.val),
            "test": len(result.splits.test),
        },
        "evaluation": evaluation.to_json(),
        "calibration": {
            "temperature": round(calibration.temperature, 6),
            "nll_before": round(calibration.temperature_fit.nll_before, 6),
            "nll_after": round(calibration.temperature_fit.nll_after, 6),
            "ece_before": round(calibration.temperature_fit.ece_before, 6),
            "ece_after": round(calibration.temperature_fit.ece_after, 6),
            "ood_threshold": round(calibration.ood_threshold, 6),
            "in_domain_rejected": round(calibration.in_domain_rejected, 6),
            "mahalanobis_scale_factor": round(calibration.mahalanobis_stats.scale_factor, 6),
            "mahalanobis_condition_number": round(
                calibration.mahalanobis_stats.condition_number, 2
            ),
        },
        "export": {
            "size_bytes": exported.size_bytes,
            "input_dtype": exported.input_dtype,
            "fully_quantized": exported.fully_quantized,
            "int8_macro_f1_drop": round(f1_drop, 6),
            "parity": parity.to_json(),
        },
        "acceptance": {
            "passed": acceptance.passed,
            "checks": [
                {"name": name, "passed": ok, "detail": detail}
                for name, ok, detail in acceptance.checks
            ],
        },
    }
    (directory / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _report_calibration(calibration: thresholds.Calibration) -> None:
    fit = calibration.temperature_fit
    console.print(
        f"  temperatura T = [bold]{calibration.temperature:.4f}[/] "
        f"· ECE {fit.ece_before:.4f} → {fit.ece_after:.4f}"
    )
    if calibration.temperature > 1.0:
        console.print(
            "  [dim]T > 1 confirma sobreconfianza: los logits se aplanan para que la "
            "probabilidad declarada corresponda al acierto real.[/]"
        )
    console.print(
        f"  umbral OOD = {calibration.ood_threshold:.4f} "
        f"(rechaza el {calibration.in_domain_rejected:.1%} de las hojas conocidas)"
    )
    console.print(
        f"  Mahalanobis: escala ×{calibration.mahalanobis_stats.scale_factor:.4f} · "
        f"condición {calibration.mahalanobis_stats.condition_number:.1f}"
    )


def _report_evaluation(evaluation: metrics.Evaluation, cfg: configuration.TrainingConfig) -> None:
    table = Table(header_style="bold")
    table.add_column("clase")
    table.add_column("apoyo", justify="right")
    table.add_column("precisión", justify="right")
    table.add_column("cobertura", justify="right")
    table.add_column("F1", justify="right")
    for row in evaluation.per_class:
        table.add_row(
            row.class_id,
            f"{row.support:,}",
            f"{row.precision:.4f}",
            f"{row.recall:.4f}",
            f"{row.f1:.4f}",
        )
    console.print(table)
    console.print(
        f"  exactitud {evaluation.accuracy:.4f} · "
        f"[bold]F1 macro {evaluation.macro_f1:.4f}[/]"
    )


def _report_gates(distribution: dict[str, float]) -> None:
    console.print("\n  De cada 100 fotos de plagas conocidas, la app respondería:")
    console.print(f"    diagnóstico          {distribution['identified'] * 100:5.1f}")
    console.print(f"    «no estoy seguro»    {distribution['low_confidence'] * 100:5.1f}")
    console.print(f"    «podría ser otra»    {distribution['ambiguous'] * 100:5.1f}")
    console.print(f"    «no la reconozco»    {distribution['ood'] * 100:5.1f}")

    to_review = distribution["low_confidence"] + distribution["ambiguous"] + distribution["ood"]
    if to_review > 0.25:
        console.print(
            f"  [yellow]! el {to_review:.0%} acabaría en revisión. Un modelo así satura al "
            "agrónomo y el productor deja de usar la app: sube `min_confidence` o "
            "revisa el entrenamiento antes de publicar.[/]"
        )


if __name__ == "__main__":
    app()
