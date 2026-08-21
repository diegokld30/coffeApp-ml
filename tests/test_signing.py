"""Pruebas de la firma Ed25519.

El contrato que se verifica aquí está escrito en `ArtifactInstaller.kt`:

```kotlin
val setDigest = MessageDigest.getInstance("SHA-256").apply {
    update(computed.lowercase().toByteArray())
    update(labelsRaw.toByteArray())
    update(calibrationRaw.toByteArray())
}.digest()
```

Si el pipeline construye ese mensaje de otra forma, la firma no verifica en el
teléfono. Y el modo de fallo es cruel: no hay error, no hay aviso, simplemente
**ningún dispositivo instala la actualización** y todos quedan congelados en la
versión anterior. El único síntoma es un contador de instalaciones que no sube.
"""

from __future__ import annotations

import base64
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from agrovision_ml.signing import sign_artifact


@pytest.fixture
def artefacto(tmp_path):
    """Un conjunto mínimo pero con la forma real."""
    directory = tmp_path / "v1.0.0"
    directory.mkdir()
    (directory / "model.tflite").write_bytes(b"\x1c\x00\x00\x00TFL3" + bytes(range(256)) * 40)
    (directory / "labels.json").write_text(
        json.dumps({"version": "1.0.0", "class_count": 2, "classes": []}, indent=2),
        encoding="utf-8",
    )
    (directory / "calibration.json").write_text(
        json.dumps({"version": "1.0.0", "temperature": 1.23}, indent=2),
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def clave(tmp_path):
    path = tmp_path / "private.pem"
    key = Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return path


def _kotlin_set_digest(directory) -> bytes:
    """Reproduce el encadenado del instalador de Android, paso a paso."""
    model_bytes = (directory / "model.tflite").read_bytes()
    model_hash_hex = hashlib.sha256(model_bytes).hexdigest().lower()

    digest = hashlib.sha256()
    digest.update(model_hash_hex.encode("utf-8"))  # update(computed.lowercase().toByteArray())
    digest.update((directory / "labels.json").read_bytes())  # update(labelsRaw.toByteArray())
    digest.update((directory / "calibration.json").read_bytes())
    return digest.digest()


def test_digest_coincide_con_el_de_android(artefacto):
    _, obtenido = sign_artifact.build_set_digest(artefacto)
    assert obtenido == _kotlin_set_digest(artefacto)


def test_firma_verifica(artefacto, clave):
    firmado = sign_artifact.sign(artefacto, clave, version="1.0.0", channel="draft")
    manifiesto = json.loads((artefacto / "manifest.json").read_text(encoding="utf-8"))

    assert sign_artifact.verify(artefacto, manifiesto["signing_public_key"])
    assert len((artefacto / "signature.bin").read_bytes()) == 64, "Ed25519 firma en 64 bytes"
    assert firmado.model_sha256 == hashlib.sha256(
        (artefacto / "model.tflite").read_bytes()
    ).hexdigest()


def test_clave_publica_son_32_bytes_crudos(clave):
    """`Ed25519SignatureVerifier` construye los parámetros sobre bytes crudos.

    Enviar la pública en PEM o DER produciría 44 u 88 bytes, BouncyCastle
    lanzaría, y `verify` devolvería `false` para artefactos perfectamente válidos.
    """
    key = sign_artifact.load_private_key(clave)
    raw = base64.b64decode(sign_artifact.public_key_base64(key))
    assert len(raw) == 32


@pytest.mark.parametrize(
    "archivo",
    ["model.tflite", "labels.json", "calibration.json"],
)
def test_manipular_cualquier_archivo_invalida_la_firma(artefacto, clave, archivo):
    """Los tres archivos están cubiertos. Ese es el punto de firmar el conjunto.

    Firmar solo el `.tflite` dejaría sustituir `calibration.json` —y con él los
    umbrales de las tres compuertas— sin romper la firma. Un atacante pondría el
    umbral OOD en infinito y la app dejaría de decir «no sé» para siempre.
    """
    sign_artifact.sign(artefacto, clave, version="1.0.0", channel="draft")
    manifiesto = json.loads((artefacto / "manifest.json").read_text(encoding="utf-8"))
    assert sign_artifact.verify(artefacto, manifiesto["signing_public_key"])

    objetivo = artefacto / archivo
    objetivo.write_bytes(objetivo.read_bytes() + b" ")

    assert not sign_artifact.verify(artefacto, manifiesto["signing_public_key"]), (
        f"Alterar {archivo} pasó desapercibido: la firma no cubre el conjunto."
    )


def test_otra_clave_no_verifica(artefacto, clave, tmp_path):
    sign_artifact.sign(artefacto, clave, version="1.0.0", channel="draft")

    impostora = tmp_path / "impostora.pem"
    otra = Ed25519PrivateKey.generate()
    impostora.write_bytes(
        otra.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    publica_ajena = sign_artifact.public_key_base64(sign_artifact.load_private_key(impostora))
    assert not sign_artifact.verify(artefacto, publica_ajena)


def test_falta_un_archivo(artefacto, clave):
    (artefacto / "calibration.json").unlink()
    with pytest.raises(sign_artifact.SigningError, match="indivisibles"):
        sign_artifact.sign(artefacto, clave, version="1.0.0", channel="draft")


def test_reformatear_el_json_rompe_la_firma(artefacto, clave):
    """Documenta por qué se hashean los BYTES en disco y no las estructuras.

    Los mismos datos con otra indentación son otro archivo para SHA-256. Si el
    pipeline firmara `json.dumps(estructura)` y el servidor sirviera el archivo
    tal cual, cualquier reformateo por el camino invalidaría un artefacto bueno.
    """
    sign_artifact.sign(artefacto, clave, version="1.0.0", channel="draft")
    manifiesto = json.loads((artefacto / "manifest.json").read_text(encoding="utf-8"))

    datos = json.loads((artefacto / "calibration.json").read_text(encoding="utf-8"))
    (artefacto / "calibration.json").write_text(json.dumps(datos), encoding="utf-8")  # sin indent

    assert not sign_artifact.verify(artefacto, manifiesto["signing_public_key"])
