# `ml/` — entrenamiento y publicación de modelos

Produce los cinco artefactos indivisibles que instala la app (§13.4), a partir de
un dataset público de café.

```
agrovision-ml train --config configs/coffee_v1.yaml
```

Un comando recorre todo: descarga → entrenamiento → calibración → evaluación →
exportación INT8 → verificación de paridad → criterios de aceptación → firma.
**Si el modelo no cumple los mínimos, se detiene antes de firmar**, y sin firma
ningún teléfono lo instala.

---

## Dónde entrenar

| | Recomendado | |
|---|---|---|
| **Google Colab, GPU T4 gratis** | ✅ | ~40 min. Abre `notebooks/AgroVision_Entrenamiento.ipynb` |
| Mac con 8 GB | ⚠️ | Corre, pero horas y con riesgo de quedarse sin memoria |
| Máquina con GPU NVIDIA | ✅ | `docker compose --profile training run ml train` |

## Los cinco artefactos

```
artifacts/v1.0.0/
├── model.tflite        ~5 MB · INT8 · entrada uint8 224×224 · DOS salidas
├── labels.json         orden canónico de clases ↔ catálogo
├── calibration.json    T, τ, δ, umbral OOD, centroides, matriz de precisión
├── metrics.json        F1 por clase, confusión, paridad, cita del dataset
└── signature.bin       Ed25519 sobre el conjunto entero
```

Viajan y se versionan **como una unidad**. La firma cubre los tres primeros
encadenados, no solo los pesos: si cubriera solo el `.tflite`, alguien podría
sustituir `calibration.json` —y con él los umbrales de las tres compuertas— sin
romper nada. Pondría el umbral OOD en infinito y la app dejaría de decir «no sé»
para siempre.

## La doble salida

**Es requisito de arquitectura, no una comodidad.**

```
imagen → tronco → GAP → Dense(128) ─┬─→ embedding
                                    └─→ Dense(5) → logits
```

El embedding alimenta dos subsistemas que sin él no existen: la compuerta OOD en
el teléfono y la agrupación por similitud en el servidor. Un modelo exportado
solo con logits compila, corre, predice — y deja muertos los dos. Por eso
`LiteRtPestClassifier` rechaza cualquier `.tflite` con menos de dos salidas, y
por eso `to_tflite.py` lo verifica antes de escribir el archivo.

## Las tres compuertas

```
FOTO
 ├─ C1 calidad ────── borrosa / oscura / sin hoja ──► «acércate», sin gastar cómputo
 ▼
INFERENCIA ──► logits + embedding
 ├─ C2 OOD ────────── energía + Mahalanobis ────────► «no la reconozco» ──► agrónomo
 ├─ C3 calibración ── p_max < τ ó margen < δ ───────► «no estoy seguro» ──► agrónomo
 ▼
DIAGNÓSTICO + ficha + opción de corregir
```

La compuerta 2 opera sobre el **embedding**, nunca sobre el softmax. Un umbral de
confianza detecta la *duda* del modelo; no detecta su *ignorancia*. Una plaga
jamás vista produce softmax de 0,95 sobre la clase visualmente parecida — es el
error prohibido nº 4 del `CLAUDE.md`.

`calibration/mahalanobis.py` es un **espejo exacto** de `EnergyMahalanobisGate.kt`.
Cuando toques una fórmula de un lado, toca la del otro en el mismo commit:
`tests/test_calibration.py::test_espejo_kotlin` reimplementa el Kotlin línea a
línea y falla si divergen.

## Datos

**JMuBEN + JMuBEN2** · 58 555 imágenes · CC BY 4.0

> Jepkoech, J.; Kenduiywo, B.; Mugo, D.; Chebet, E. (2021). *Arabica coffee leaf
> images dataset for coffee leaf disease detection and classification.*
> Data in Brief 36, 107142. DOI [10.17632/t2r6rszp5c.1](https://doi.org/10.17632/t2r6rszp5c.1)
> y [10.17632/tgv3zb82nd.1](https://doi.org/10.17632/tgv3zb82nd.1)

| clase | imágenes | ficha |
|---|---|---|
| Roya (`Leaf rust`) | 8 337 | existía |
| Mancha de hierro (`Cerscospora`) | 7 682 | existía |
| Quema del cogollo (`Phoma`) | 6 572 | **nueva** |
| Minador (`Miner`) | 16 979 | **nueva** |
| Sana (`Healthy`) | 18 985 | sin ficha |

**Dos limitaciones que hay que tener presentes** (ADR-016):

- Las imágenes vienen **recortadas a 128×128 pegadas a la lesión**. En campo el
  teléfono manda un recorte central de una foto de 1280 px. Se mitiga con zoom
  agresivo en la aumentación; no se cura sin fotos propias.
- Se tomaron en Kirinyaga, Kenia. No es el Macizo Colombiano: otra altura, otro
  suelo, otras variedades, otra luz.

## Reentrenar

Es el punto de todo el montaje.

```bash
# 1. Copia las fotos nuevas a la carpeta de su clase
cp fotos_piloto/roya/*.jpg data/images/"Leaf rust"/

# 2. Sube la versión en configs/coffee_v1.yaml
# 3. Vuelve a ejecutar
agrovision-ml train --config configs/coffee_v1.yaml
```

El reparto entrenamiento/validación/prueba **no se recalcula al azar**: se deriva
del hash BLAKE2b del nombre del archivo. Una imagen que estaba en prueba sigue en
prueba aunque el dataset crezca, así que las métricas de la versión nueva son
comparables con las de la anterior. Con un barajado normal no lo serían, y nadie
se daría cuenta.

### Para añadir una clase

Añade su entrada a `classes:` **al final de la lista**. Insertarla en medio
recorre los índices de los logits y el modelo diría «roya» donde antes decía
«minador», sin ningún error visible. Cambiar el conjunto de clases es un cambio
**mayor** de versión (§17.2) y exige subir `minAppVersion`.

## Comandos

```bash
agrovision-ml download    # solo el dataset (~1,75 GB, reanudable)
agrovision-ml train       # el ciclo completo
agrovision-ml keys        # genera el par Ed25519 (una sola vez, ver aviso)
agrovision-ml verify <dir> --public-key <base64>
```

## Desarrollo

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

Las pruebas de `tests/` **no necesitan TensorFlow**: cubren la matemática de
calibración y el contrato de firma, que es donde un error pasa desapercibido
hasta llegar a campo.

## Avisos

> ⚠️ **La clave privada de firma nunca entra al repositorio**, ni a un contenedor
> de producción, ni a los secretos de CI que puede leer cualquiera del equipo
> (§15.6). Vive en la estación de ML o en un HSM.
>
> Y **no la regeneres** si ya hay modelos publicados: los teléfonos verifican
> contra la clave pública embebida en el APK, así que una clave nueva obliga a
> publicar una versión nueva de la app y deja sin actualizaciones a quien no la
> instale.

> ⚠️ **El AUROC exige muestras fuera de dominio.** Sin `--ood`, `metrics.json`
> registra `null` y el modelo puede ir a `draft` o `internal`, **no a
> `production`**. Las mejores muestras llegarán del piloto: son exactamente las
> fotos que la app manda a revisión con «no la reconozco».

> ⚠️ **Las fichas nuevas no llevan recomendaciones de manejo.** Están vacías a
> propósito hasta que un ingeniero agrónomo con tarjeta profesional vigente las
> avale (§13.6). Publicar dosis sin respaldo expone ante el ICA.
