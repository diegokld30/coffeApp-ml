"""Descarga de JMuBEN / JMuBEN2 desde Mendeley Data.

Los dos conjuntos se publican bajo CC BY 4.0 y se sirven **sin credenciales**,
lo que permite que cualquiera del equipo reproduzca el entrenamiento desde cero
sin pedirle una cuenta a nadie.

    Jepkoech, J.; Kenduiywo, B.; Mugo, D.; Chebet, E. (2021)
    «Arabica coffee leaf images dataset for coffee leaf disease detection and
    classification». Data in Brief, 36, 107142.
    DOI 10.17632/t2r6rszp5c.1 · DOI 10.17632/tgv3zb82nd.1

Reanudable y verificable: son ~1,75 GB en cinco archivos, y en una conexión
rural la descarga se corta. Cada archivo se baja a `.parcial`, se reanuda con
`Range` si el servidor lo admite, y solo se renombra al terminar. Un `.zip`
presente es un `.zip` íntegro.
"""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import requests
from rich.console import Console
from tqdm import tqdm

console = Console()

_API = "https://data.mendeley.com/public-api/datasets/{dataset_id}/files"
_TIMEOUT = (15, 120)  # (conexión, lectura entre bytes)
_CHUNK = 1 << 20  # 1 MiB


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteFile:
    dataset_id: str
    filename: str
    size_bytes: int
    url: str

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1_048_576


def list_files(dataset_id: str) -> list[RemoteFile]:
    """Consulta el índice público del dataset."""
    response = requests.get(
        _API.format(dataset_id=dataset_id),
        params={"folder_id": "root", "version": "1"},
        timeout=_TIMEOUT,
    )
    if response.status_code != 200:
        raise DownloadError(
            f"Mendeley respondió {response.status_code} para {dataset_id}. "
            "Si persiste, descarga los zip a mano desde "
            f"https://data.mendeley.com/datasets/{dataset_id}/1 y colócalos en data/raw/."
        )

    files: list[RemoteFile] = []
    for entry in response.json():
        url = (entry.get("content_details") or {}).get("download_url")
        if not url:
            continue
        files.append(
            RemoteFile(
                dataset_id=dataset_id,
                filename=entry["filename"],
                size_bytes=int(entry.get("size", 0)),
                url=url,
            )
        )

    if not files:
        raise DownloadError(f"El dataset {dataset_id} no expuso ningún archivo descargable")
    return files


def _download_one(remote: RemoteFile, destination: Path) -> Path:
    final = destination / remote.filename
    if final.exists() and final.stat().st_size > 0:
        console.print(f"  [dim]ya estaba:[/] {remote.filename}")
        return final

    partial = final.with_suffix(final.suffix + ".parcial")
    downloaded = partial.stat().st_size if partial.exists() else 0

    headers = {}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"

    with requests.get(remote.url, stream=True, timeout=_TIMEOUT, headers=headers) as response:
        # 206 = reanudó donde se quedó. 200 con Range pedido = el servidor lo
        # ignoró y manda el archivo entero, así que hay que empezar de nuevo.
        if downloaded and response.status_code == 200:
            downloaded = 0
            partial.unlink(missing_ok=True)
        elif response.status_code not in (200, 206):
            raise DownloadError(f"{remote.filename}: HTTP {response.status_code}")

        total = remote.size_bytes or None
        mode = "ab" if downloaded else "wb"
        with (
            partial.open(mode) as handle,
            tqdm(
                total=total,
                initial=downloaded,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"  {remote.filename[:34]:34}",
                leave=False,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=_CHUNK):
                handle.write(chunk)
                bar.update(len(chunk))

    if remote.size_bytes and partial.stat().st_size != remote.size_bytes:
        raise DownloadError(
            f"{remote.filename}: se esperaban {remote.size_bytes} bytes y llegaron "
            f"{partial.stat().st_size}. Vuelve a ejecutar para reanudar."
        )

    partial.rename(final)
    return final


def _extract(archive: Path, destination: Path) -> None:
    """Descomprime aplanando la jerarquía a `<destino>/<carpeta_de_clase>/`.

    Los zip de Mendeley traen anidamientos distintos entre sí —unos con carpeta
    raíz, otros sin ella—, y algunos incluyen basura de macOS (`__MACOSX`,
    `.DS_Store`). Fijar aquí una estructura única evita que el cargador de datos
    tenga que adivinar.
    """
    with zipfile.ZipFile(archive) as zf:
        members = [
            m
            for m in zf.infolist()
            if not m.is_dir()
            and "__MACOSX" not in m.filename
            and not Path(m.filename).name.startswith(".")
            and Path(m.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not members:
            raise DownloadError(f"{archive.name} no contiene imágenes")

        # La carpeta de clase es el primer componente con nombre del interior.
        for member in tqdm(
            members, desc=f"  {archive.stem[:34]:34}", unit="img", leave=False
        ):
            parts = Path(member.filename).parts
            class_dir = parts[0] if len(parts) > 1 else archive.stem.split("-")[0]
            target = destination / class_dir / Path(member.filename).name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                continue
            with zf.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)


def ensure(dataset_ids: list[str], root: Path) -> Path:
    """Descarga y descomprime lo que falte. Devuelve la carpeta de imágenes.

    Idempotente: correrlo dos veces no vuelve a bajar ni a descomprimir nada.
    """
    raw = root / "raw"
    images = root / "images"
    raw.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    for dataset_id in dataset_ids:
        console.print(f"[cyan]▸[/] Índice de [bold]{dataset_id}[/]")
        remotes = list_files(dataset_id)
        total_mb = sum(r.size_mb for r in remotes)
        console.print(f"  {len(remotes)} archivo(s) · {total_mb:.0f} MB")

        for remote in remotes:
            archive = _download_one(remote, raw)
            _extract(archive, images)

    found = sorted(p.name for p in images.iterdir() if p.is_dir())
    console.print(f"[green]  ✓[/] clases en disco: {', '.join(found)}")
    return images


def count_by_class(images: Path) -> dict[str, int]:
    return {
        directory.name: sum(1 for _ in _iter_images(directory))
        for directory in sorted(images.iterdir())
        if directory.is_dir()
    }


def _iter_images(directory: Path) -> Iterator[Path]:
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        yield from directory.glob(suffix)
