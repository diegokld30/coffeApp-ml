"""Firma Ed25519 del conjunto de artefactos.

## Qué se firma, exactamente

`ArtifactInstaller.kt` construye el mensaje así, y esto es un contrato literal:

```kotlin
val setDigest = MessageDigest.getInstance("SHA-256").apply {
    update(computed.lowercase().toByteArray())   // hex del SHA-256 del .tflite
    update(labelsRaw.toByteArray())              // bytes exactos de labels.json
    update(calibrationRaw.toByteArray())         // bytes exactos de calibration.json
}.digest()
```

Es decir: se concatenan el **hash hexadecimal en minúsculas** del `.tflite`, el
texto de `labels.json` y el de `calibration.json`; se aplica SHA-256 al resultado
y **eso** es lo que se firma.

Que la firma cubra el conjunto y no solo los pesos es deliberado. Firmar
únicamente el `.tflite` dejaría sustituir `calibration.json` —y con él los
umbrales de las tres compuertas— sin invalidar la firma. Un atacante pondría el
umbral OOD en infinito y la app dejaría de decir «no sé» para siempre,
diagnosticando con aplomo cualquier cosa que se le ponga delante.

Los bytes se leen **de los archivos ya escritos en disco**, no de las estructuras
en memoria. Un `json.dumps` con otra indentación produce los mismos datos y un
hash distinto, y la app rechazaría un artefacto perfectamente válido.

## Dónde vive la clave privada

En la estación de ML o en un HSM. **Nunca** en el repositorio, nunca en un
contenedor de producción, nunca en las variables de CI que puede leer un
compañero de equipo (§15.6). La pública se embebe en el APK y viaja también en
`BuildConfig.MODEL_SIGNING_PUBLIC_KEY`.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

MODEL_FILE = "model.tflite"
LABELS_FILE = "labels.json"
CALIBRATION_FILE = "calibration.json"
METRICS_FILE = "metrics.json"
SIGNATURE_FILE = "signature.bin"
MANIFEST_FILE = "manifest.json"


class SigningError(RuntimeError):
    pass


@dataclass(frozen=True)
class SignedArtifact:
    directory: Path
    version: str
    model_sha256: str
    set_digest_hex: str
    signature_base64: str
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1_048_576


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def build_set_digest(directory: Path) -> tuple[str, bytes]:
    """Reproduce el mensaje que firma el pipeline y verifica el teléfono.

    Devuelve `(hash_hex_del_tflite, digest_del_conjunto)`.
    """
    model = directory / MODEL_FILE
    labels = directory / LABELS_FILE
    calibration = directory / CALIBRATION_FILE

    for path in (model, labels, calibration):
        if not path.exists():
            raise SigningError(f"Falta {path.name}: los cinco artefactos son indivisibles (§13.4)")

    model_hash = _sha256_file(model)

    digest = hashlib.sha256()
    digest.update(model_hash.encode("utf-8"))
    # Bytes tal cual están en disco. Ver el docstring del módulo.
    digest.update(labels.read_bytes())
    digest.update(calibration.read_bytes())

    return model_hash, digest.digest()


def load_private_key(path: Path) -> Ed25519PrivateKey:
    if not path.exists():
        raise SigningError(
            f"No existe la clave privada {path}.\n"
            "Genérala con `scripts/generate-signing-keys.sh` y guárdala FUERA del "
            "repositorio. Si se filtra, cualquiera puede publicar un modelo que la "
            "app instalará como legítimo."
        )

    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SigningError(
            f"{path} no es una clave Ed25519 (llegó {type(key).__name__}). "
            "La app verifica con Ed25519 y solo con eso."
        )
    return key


def public_key_base64(private_key: Ed25519PrivateKey) -> str:
    """Clave pública en el formato que espera `Ed25519SignatureVerifier`.

    Son los 32 bytes crudos en Base64 — no PEM, no DER. BouncyCastle construye
    `Ed25519PublicKeyParameters` directamente sobre ellos.
    """
    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def sign(directory: Path, private_key_path: Path, *, version: str, channel: str) -> SignedArtifact:
    """Firma el conjunto y escribe `signature.bin` y `manifest.json`."""
    private_key = load_private_key(private_key_path)
    model_hash, digest = build_set_digest(directory)

    signature = private_key.sign(digest)
    signature_base64 = base64.b64encode(signature).decode("ascii")

    (directory / SIGNATURE_FILE).write_bytes(signature)

    manifest = {
        "version": version,
        "channel": channel,
        "sha256": model_hash,
        "signature": signature_base64,
        "set_digest_sha256": digest.hex(),
        "artifacts": {
            "model": MODEL_FILE,
            "labels": LABELS_FILE,
            "calibration": CALIBRATION_FILE,
            "metrics": METRICS_FILE,
        },
        "signing_public_key": public_key_base64(private_key),
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = sum(
        (directory / name).stat().st_size
        for name in (MODEL_FILE, LABELS_FILE, CALIBRATION_FILE)
    )

    return SignedArtifact(
        directory=directory,
        version=version,
        model_sha256=model_hash,
        set_digest_hex=digest.hex(),
        signature_base64=signature_base64,
        size_bytes=total,
    )


def verify(directory: Path, public_key_b64: str) -> bool:
    """Verifica la firma igual que lo hará el teléfono.

    Se ejecuta **siempre** al terminar de firmar. Descubrir en campo que la firma
    no valida significa que ningún teléfono aceptó la actualización y todos
    quedaron congelados en la versión anterior, sin más síntoma que un contador
    de instalaciones que no sube.
    """
    signature_path = directory / SIGNATURE_FILE
    if not signature_path.exists():
        raise SigningError(f"No hay {SIGNATURE_FILE} en {directory}")

    _, digest = build_set_digest(directory)
    public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))

    try:
        public_key.verify(signature_path.read_bytes(), digest)
    except Exception:  # noqa: BLE001 — `InvalidSignature` y cualquier error de formato
        return False
    return True
