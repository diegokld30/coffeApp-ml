"""El repositorio publicado tiene que contener el paquete completo.

Esta prueba existe por un fallo real: `.gitignore` traía el patrón `data/` para
excluir el dataset descargado, pero un patrón sin barra inicial coincide a
**cualquier profundidad**, así que se llevó también `src/agrovision_ml/data/`
—el descargador y la canalización tf.data— y el repositorio se publicó sin él.

En local no se notaba: los archivos estaban en disco y todo importaba. Solo
reventaba al clonar, con `ModuleNotFoundError: No module named
'agrovision_ml.data'` en la primera ejecución, después de haber montado Drive.

Es la peor forma de este error: no falla donde se comete.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
FUENTE = RAIZ / "src" / "agrovision_ml"


def _archivos_versionados() -> set[Path]:
    salida = subprocess.run(
        ["git", "ls-files"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {Path(linea) for linea in salida.splitlines() if linea}


@pytest.mark.skipif(
    not (RAIZ / ".git").exists(), reason="no es una copia de trabajo de git"
)
def test_todo_el_codigo_esta_versionado():
    versionados = _archivos_versionados()
    faltantes = sorted(
        modulo.relative_to(RAIZ)
        for modulo in FUENTE.rglob("*.py")
        if "__pycache__" not in modulo.parts
        and modulo.relative_to(RAIZ) not in versionados
    )

    assert not faltantes, (
        "Estos módulos existen en disco pero NO están en el repositorio, así que "
        "quien lo clone recibirá un paquete incompleto:\n  "
        + "\n  ".join(str(f) for f in faltantes)
        + "\n\nCausa habitual: un patrón de .gitignore sin barra inicial, que "
        "coincide a cualquier profundidad. Ánclalo con `/` al principio."
    )


def test_cada_paquete_tiene_init():
    """Sin `__init__.py`, el directorio no se instala como parte del paquete."""
    sin_init = sorted(
        carpeta.relative_to(RAIZ)
        for carpeta in FUENTE.rglob("*")
        if carpeta.is_dir()
        and "__pycache__" not in carpeta.parts
        and any(carpeta.glob("*.py"))
        and not (carpeta / "__init__.py").exists()
    )
    assert not sin_init, f"Carpetas con código pero sin __init__.py: {sin_init}"


def test_los_submodulos_importan():
    """Recorre el paquete e importa todo lo que no dependa de TensorFlow.

    Los módulos que sí lo usan se saltan: estas pruebas están pensadas para
    correr sin TensorFlow instalado, que es lo que permite verificarlas en
    cualquier máquina en menos de dos segundos.
    """
    import importlib

    for modulo in (
        "agrovision_ml.config",
        "agrovision_ml.calibration.temperature",
        "agrovision_ml.calibration.mahalanobis",
        "agrovision_ml.calibration.thresholds",
        "agrovision_ml.evaluation.metrics",
        "agrovision_ml.signing.sign_artifact",
        "agrovision_ml.data.download",
    ):
        importlib.import_module(modulo)
