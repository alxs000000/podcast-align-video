# podcast-align-video

`podcast-align-video` turns an English audio file or one public YouTube video into a 1920×1080 word-highlight video. It measures the subtitle layout in Chromium, then redraws that layout with ASS/libass so long recordings render much faster than real-time browser capture.

## v0.1 scope

- Linux or WSL2, Python 3.12, and an NVIDIA CUDA GPU.
- Any local audio format FFmpeg can decode, or one public YouTube video that does not require login or cookies.
- English transcription only.
- A synchronous CLI and small Python `run()` API.
- No HTTP service, database, queue, scheduler, Docker image, job lock, or concurrent-run guarantee.

You are responsible for rights to download and transform source media. YouTube playlists, private/authenticated videos, cookies, and live streams are intentionally unsupported. YouTube sources are fetched again on every run; the program never silently substitutes an old cached download.

## Install

Install system prerequisites yourself: FFmpeg/FFprobe built with libass and libx264, a working NVIDIA driver exposed to Linux/WSL2, `curl`, `tar`, `bzip2`, and the shared libraries required by Playwright Chromium. The setup script does not invoke `sudo` or a package manager. `doctor` performs a real headless Chromium launch and reports missing runtime libraries before media processing.

```bash
./scripts/setup.sh
```

The script uses the committed Linux/Python 3.12 lock files to create separate core, Cohere, and Qwen `uv` environments, a dedicated MFA 3.4.1 micromamba environment, and a Playwright Chromium installation under `~/.local/share/podcast-align-video`. The bootstrap tools themselves are pinned to uv 0.12.9 and micromamba 2.9.0. If a custom WSL data root is on a DrvFS mount that rejects Unix symlinks or Unix lock semantics, setup places the MFA and browser executable runtimes in a keyed user-local Linux directory and records their locations under the data root; models, checkpoints, source media, and outputs stay in the configured data root.

Copy `config/default.toml` when you need a different data root, model path, GPU device, or encoder. Paths are expanded locally and are never embedded in the repository.

## Models

Cohere Transcribe is gated. First request/accept access on the [official model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026), then authenticate with your own Hugging Face token. The tool cannot accept model terms for you.

```bash
podcast-align-video models fetch --config config/local.toml
podcast-align-video doctor --config config/local.toml
```

`models fetch` displays every model ID, immutable revision, license/model-card URL, and approximate size before downloading. `run` never downloads a missing model and fails during preflight with recovery instructions.

## Run

```bash
podcast-align-video run ./episode.flac --config config/local.toml
podcast-align-video run 'https://www.youtube.com/watch?v=VIDEO_ID' --config config/local.toml
podcast-align-video run ./episode.wav --output-dir ./my-output --silence-threshold 7.5 --device cuda:0
```

When `--output-dir` is omitted, output goes to `./outputs/<sanitized-title>-<fingerprint12>/`.

```text
video.mp4
video-speech-cut.mp4       # only when at least one qualifying silence was removed
source.<original-extension>
transcript.txt
word-timings.json
run-manifest.json
run.log
```

The full video is canonical. Silence removal defaults to 5 seconds and protects aligned word-focus intervals. No-cut is recorded as `no_cuts`; failure of only the optional speech-cut render produces a warning and exit status 0 after the full video is validated.

Stages use content hashes, effective settings, immutable model revisions, schema versions, atomic writes, and validated checkpoints. Re-running the same input and configuration resumes safe work. An explicit output directory with a different fingerprint is never overwritten.

Work is retained for manual resume. To inspect or remove only one completed job's work:

```bash
podcast-align-video clean JOB_ID
podcast-align-video clean JOB_ID --yes
```

The first command is a dry run showing the size. Neither command removes published artifacts or models.

Do not run the same job or share one GPU between concurrent invocations. v0.1 deliberately has no process or GPU lock; concurrent execution is the operator's responsibility.

## Python API

```python
from pathlib import Path
from podcast_align_video import RunConfig, run

result = run("episode.wav", output_dir=Path("output"), config=Path("config/local.toml"))
print(result.full_video)
```

`run`, `RunConfig`, and `RunResult` are the stable v0.1 Python surface. The small public `Transcriber` and `Aligner` protocols are available for code organization, but there is no dynamic plugin discovery and only the bundled Cohere → Qwen → MFA pipeline is supported.

## Rendering and codecs

Chromium measures every sentence after `document.fonts.ready`; the measurements are resumable JSONL checkpoints. ASS/libass reproduces the centered Geist typography, 76 px ceiling, wrapping, dark background, and gold rounded active-word box. Video-only H.264 segments are rendered in 30-minute units, validated, concatenated without re-encoding, and muxed once with source audio encoded as 48 kHz AAC at a requested 192 kbps.

The default encoder is `libx264`, preset `veryfast`, CRF 20. Set `renderer.encoder = "h264_nvenc"` and an appropriate NVENC preset such as `p4` in TOML if desired.

## Tests

The default suite uses fake model workers and a mocked YouTube download, so it does not require model files or a GPU:

```bash
python -m pytest
```

Set `PAV_RUN_RENDER_INTEGRATION=1` to include the real Chromium → ASS → FFmpeg test, and `PAV_RUN_SPEECH_CUT_INTEGRATION=1` to include the real speech-cut media test. Hardware/model acceptance tests are intentionally manual because they require gated weights and an NVIDIA host.

Maintainers can compare an acceptance result with a licensed word-boundary reference using `scripts/quality-report.py PREDICTION GOLD --output REPORT`. WER and boundary statistics are reported for observation and are not v0.1 hard gates.

## License

Source code is Apache-2.0. The bundled Geist font is licensed under SIL Open Font License 1.1; see `LICENSES/OFL-1.1.txt` and `THIRD_PARTY_NOTICES.md`.
