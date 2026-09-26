<p align="center">
  <img src="assets/brand-icon.png" width="176" alt="Logotipo de podcast-align-video">
</p>

<h1 align="center">podcast-align-video</h1>

<p align="center">
  <a href="../README.md" lang="en">English</a> · <a href="README.ja.md" lang="ja">日本語</a> · <a href="README.zh-CN.md" lang="zh-CN">简体中文</a> · <a href="README.ko.md" lang="ko">한국어</a> · Español
</p>

<p align="center">
  <strong>Sigue el inglés con los oídos y con los ojos.</strong><br>
  Convierte audio en inglés o un vídeo público de YouTube en un vídeo con subtítulos que resaltan cada palabra al pronunciarse.
</p>

<p align="center">
  <img src="assets/demo.gif" width="960" alt="Demostración de subtítulos que resaltan en dorado cada palabra al escucharse">
</p>

La palabra que se está pronunciando se ilumina en dorado, para que puedas seguir un podcast o una conversación en inglés mientras lees. Introduce un archivo de audio en inglés o la URL de un vídeo de YouTube que no requiera iniciar sesión y obtendrás un MP4 que puedes reproducir en tu reproductor habitual.

La versión 0.1 genera únicamente subtítulos en inglés. No traduce al español, no muestra definiciones y no transcribe audio en español. Esta página explica la herramienta y su uso en español; el programa se ejecuta desde la terminal.

[Ver la demo con sonido](https://github.com/alxs000000/podcast-align-video/releases/download/v0.1.0/podcast-align-video-demo.mp4) · [Descargar v0.1.0](https://github.com/alxs000000/podcast-align-video/releases/tag/v0.1.0)

La demo usa un fragmento de [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/), ES2002a, hablante A, segundos 77,0–81,4, bajo CC BY 4.0. Se recortó el fragmento, se añadieron subtítulos y se convirtió a vídeo. El GIF no tiene sonido. [Procedencia y mediciones, en inglés](DEMO.md).

## Funciones y requisitos

- Admite audio local que FFmpeg pueda decodificar, o un único vídeo público de YouTube sin autenticación.
- Genera el vídeo completo y, cuando hay silencios que cumplen el criterio, una versión sin esos silencios. El umbral predeterminado es de 5 segundos.
- Guarda el audio original, la transcripción en inglés, los tiempos de cada palabra y el registro de ejecución.
- Reutiliza resultados intermedios verificados cuando la entrada, los ajustes y las versiones de los modelos coinciden, para reanudar trabajos largos.

Necesitas Linux o WSL2 en Windows, una GPU NVIDIA compatible con CUDA, FFmpeg／FFprobe con libass y libx264, el controlador NVIDIA, `curl`, `tar`, `bzip2` y las bibliotecas compartidas que requiere Playwright Chromium. v0.1 no admite macOS, Windows sin WSL2 ni equipos sin GPU, y no dispone de interfaz gráfica. La transcripción automática y los tiempos de las palabras pueden contener errores.

## Preparación inicial

Ejecuta estos comandos en una terminal Bash de Linux／WSL2:

```bash
git clone https://github.com/alxs000000/podcast-align-video.git
cd podcast-align-video
./scripts/setup.sh
export PATH="$HOME/.local/bin:$PATH"
cp config/default.toml config/local.toml
```

El script instala Python 3.12, entornos separados y Chromium en el espacio del usuario, sin ejecutar `sudo` ni `apt`. Los datos se guardan por defecto en `~/.local/share/podcast-align-video`. Si una nueva terminal no encuentra el comando, repite la línea `export PATH`.

Solicita acceso y acepta las condiciones en la [página del modelo de Cohere](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026). Después de obtener aprobación, usa un token de lectura de tu propia cuenta de Hugging Face. El token introducido no se muestra en pantalla:

```bash
read -r -s -p 'Hugging Face token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
podcast-align-video models fetch --config config/local.toml
unset HF_TOKEN
podcast-align-video doctor --config config/local.toml
```

`models fetch` muestra los modelos, sus versiones fijas, las páginas de licencia y los tamaños aproximados antes de descargarlos; no acepta condiciones en tu nombre. `doctor` comprueba el entorno. `run` no descarga modelos y termina antes de empezar el procesamiento costoso si falta algún requisito.

## Crear un vídeo

```bash
# Audio local en inglés
podcast-align-video run ./episode.flac --config config/local.toml

# Un vídeo público de YouTube: sustituye VIDEO_ID
podcast-align-video run 'https://www.youtube.com/watch?v=VIDEO_ID' --config config/local.toml

# Carpeta de salida y silencios de al menos 7,5 segundos
podcast-align-video run ./episode.wav --config config/local.toml \
  --output-dir ./my-output --silence-threshold 7.5 --device cuda:0
```

No se admiten listas de reproducción, vídeos privados ni vídeos que requieran cookies o iniciar sesión. El audio de YouTube se descarga de nuevo en cada ejecución y conserva el formato descargado. El audio local se copia sin alterar sus bytes.

Sin `--output-dir`, los resultados se guardan en `./outputs/<título-normalizado>-<primeros12caracteres-de-la-huella>/`. `video.mp4` es el vídeo completo y `video-speech-cut.mp4` solo se crea si se eliminó algún silencio. También se guardan `source.<extensión-original>`, `transcript.txt`, `word-timings.json`, `run-manifest.json` y `run.log`. Un fallo limitado al recorte de silencios no afecta al vídeo completo ya generado.

## Cómo se generan vídeos largos más rápido

Chromium mide el tamaño del texto, los saltos de línea y la posición de cada palabra. Después, ASS／libass dibuja el vídeo a partir de esas medidas, sin grabar el navegador en tiempo real. La salida predeterminada es H.264 a 1920×1080 y 30 fps, con audio AAC a 48 kHz y una tasa de bits solicitada de 192 kbps.

El proceso usa Silero para detectar voz, Cohere para transcribir, Qwen para alinear palabras y MFA para corregir sus límites. MFA es obligatorio. Si termina correctamente pero algunas correcciones locales no son válidas, solo esos intervalos conservan los tiempos de Qwen.

Vuelve a ejecutar la misma entrada con los mismos ajustes para reanudar desde resultados válidos. Si la carpeta de salida explícita contiene otro trabajo, no se sobrescribe. `podcast-align-video clean JOB_ID` muestra qué archivos de trabajo de un job terminado se eliminarían; añade `--yes` para eliminarlos. Conserva resultados y modelos. No hay protección contra ejecuciones simultáneas del mismo trabajo o en la misma GPU.

## Licencias y documentación técnica

El código usa [Apache-2.0](../LICENSE) y la fuente Geist usa [OFL-1.1](../LICENSES/OFL-1.1.txt). Los modelos y la demo tienen sus propias condiciones; consulta los [avisos de terceros](../THIRD_PARTY_NOTICES.md). Debes tener derecho a descargar y transformar el material de entrada.

Consulta el [README en inglés](../README.md#python-api) para la API de Python, las pruebas y los detalles técnicos.
