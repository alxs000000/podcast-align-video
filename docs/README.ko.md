<p align="center">
  <img src="assets/brand-icon.png" width="176" alt="podcast-align-video 로고">
</p>

<h1 align="center">podcast-align-video</h1>

<p align="center">
  <a href="../README.md" lang="en">English</a> · <a href="README.ja.md" lang="ja">日本語</a> · <a href="README.zh-CN.md" lang="zh-CN">简体中文</a> · 한국어 · <a href="README.es.md" lang="es">Español</a>
</p>

<p align="center">
  <strong>영어를 들으면서, 지금 말하는 단어를 눈으로 따라가세요.</strong><br>
  영어 음성이나 공개 YouTube 영상 하나로 단어별 강조 자막 영상을 만듭니다.
</p>

<p align="center">
  <img src="assets/demo.gif" width="960" alt="음성에 맞춰 영어 단어를 하나씩 금색으로 강조하는 자막 영상 데모">
</p>

말하고 있는 단어가 금색으로 강조되어, 영어 팟캐스트나 대화의 음성과 자막을 함께 따라갈 수 있습니다. 로컬 영어 음성 파일 또는 로그인 없이 볼 수 있는 YouTube 영상 하나를 입력하면 일반 영상 플레이어에서 재생할 수 있는 MP4를 생성합니다.

v0.1은 영어 자막만 생성합니다. 한국어 번역, 단어 뜻 표시, 한국어 음성 인식 기능은 없습니다. 이 페이지는 한국어 소개와 사용 안내이며, 프로그램은 터미널에서 실행합니다.

[소리가 있는 데모 보기](https://github.com/alxs000000/podcast-align-video/releases/download/v0.1.0/podcast-align-video-demo.mp4) · [v0.1.0 다운로드](https://github.com/alxs000000/podcast-align-video/releases/tag/v0.1.0)

데모는 [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)의 ES2002a, 화자 A, 77.0–81.4초 구간입니다. CC BY 4.0에 따라 구간 추출, 자막 추가, 영상 변환을 했습니다. 위 GIF에는 소리가 없습니다. [출처와 측정 결과(영어)](DEMO.md).

## 기능과 실행 환경

- FFmpeg로 읽을 수 있는 로컬 음성 파일 또는 공개 YouTube 영상 하나를 처리합니다.
- 전체 영상을 생성하며, 조건에 맞는 긴 무음이 있으면 무음을 제거한 영상도 생성합니다. 기본 기준은 5초입니다.
- 원본 음성, 영어 전사 텍스트, 단어별 시간 정보, 실행 기록을 저장합니다.
- 입력·설정·모델 버전이 같으면 검증된 중간 결과를 재사용하여 긴 작업을 이어갈 수 있습니다.

Linux 또는 Windows의 WSL2와 CUDA를 지원하는 NVIDIA GPU가 필요합니다. FFmpeg／FFprobe(libass·libx264 포함), NVIDIA 드라이버, `curl`, `tar`, `bzip2`, Playwright Chromium에 필요한 공유 라이브러리를 먼저 준비하세요. macOS, WSL2 없는 Windows, GPU 없는 환경은 지원하지 않으며 GUI도 없습니다. 자동 생성 자막과 단어 시간에는 오류가 남을 수 있습니다.

## 처음 설치하기

Linux／WSL2의 Bash 터미널에서 실행하세요.

```bash
git clone https://github.com/alxs000000/podcast-align-video.git
cd podcast-align-video
./scripts/setup.sh
export PATH="$HOME/.local/bin:$PATH"
cp config/default.toml config/local.toml
```

setup은 Python 3.12, 전용 실행 환경, Chromium을 사용자 영역에 설치합니다. `sudo`나 `apt`는 실행하지 않습니다. 기본 데이터 위치는 `~/.local/share/podcast-align-video`입니다. 새 터미널에서 명령을 찾지 못하면 위의 `export PATH`를 다시 실행하세요.

[Cohere 모델 페이지](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)에서 접근을 신청하고 이용 약관에 동의해야 합니다. 승인된 계정의 Hugging Face 읽기 토큰을 준비한 뒤 다음을 실행하세요. 입력한 토큰은 화면에 표시되지 않습니다.

```bash
read -r -s -p 'Hugging Face token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
podcast-align-video models fetch --config config/local.toml
unset HF_TOKEN
podcast-align-video doctor --config config/local.toml
```

`models fetch`는 모델 이름·고정 버전·라이선스 페이지·예상 용량을 표시한 뒤 다운로드합니다. 이용 약관에 대신 동의하지는 않습니다. `doctor`는 환경을 확인합니다. `run`은 모델을 다운로드하지 않으며, 필수 항목이 없으면 시간이 오래 걸리는 처리를 시작하기 전에 종료합니다.

## 영상 만들기

```bash
# 로컬 영어 음성
podcast-align-video run ./episode.flac --config config/local.toml

# 공개 YouTube 영상 하나: VIDEO_ID를 실제 ID로 바꾸세요
podcast-align-video run 'https://www.youtube.com/watch?v=VIDEO_ID' --config config/local.toml

# 저장 위치 지정, 7.5초 이상의 무음 제거
podcast-align-video run ./episode.wav --config config/local.toml \
  --output-dir ./my-output --silence-threshold 7.5 --device cuda:0
```

재생목록, cookie나 로그인이 필요한 영상, 비공개 영상은 지원하지 않습니다. YouTube 음성은 실행할 때마다 다시 다운로드하고 받은 형식을 유지합니다. 로컬 음성은 바이트를 바꾸지 않고 복사합니다.

기본 출력 위치는 `./outputs/<정리한 제목>-<입력 및 설정 지문 앞12자리>/`입니다. `video.mp4`가 전체 영상이며, 실제 무음을 제거한 경우에만 `video-speech-cut.mp4`가 생성됩니다. `source.<원본 확장자>`, `transcript.txt`, `word-timings.json`, `run-manifest.json`, `run.log`도 저장합니다. 무음 제거만 실패해도 이미 완성된 전체 영상은 유지됩니다.

## 긴 음성도 빠르게 영상으로 만드는 방식

브라우저로 자막의 글자 크기·줄바꿈·단어 위치를 측정한 후 ASS／libass로 영상을 그립니다. 브라우저를 음성 길이만큼 실시간으로 녹화할 필요가 없습니다. 기본 출력은 1920×1080, 30fps, H.264이며 음성은 48 kHz AAC, 지정 비트레이트 192 kbps입니다.

Silero 발화 감지 → Cohere 전사 → Qwen 시간 정렬 → MFA 보정 → 브라우저 측정 → 영상 생성 순서로 처리합니다. MFA는 필수입니다. 정상 실행 후 일부 구간의 보정 결과를 사용할 수 없을 때만 해당 구간을 Qwen 시간으로 되돌립니다.

같은 입력과 설정으로 다시 실행하면 유효한 중간 결과를 재사용합니다. 지정한 출력 폴더에 다른 작업의 결과가 있으면 덮어쓰지 않습니다. `podcast-align-video clean JOB_ID`는 완료된 작업의 정리 계획만 표시하며, `--yes`를 추가하면 작업 파일만 삭제합니다. 완성 결과와 모델은 유지됩니다. 같은 작업이나 같은 GPU의 동시 실행을 막는 기능은 없습니다.

## 라이선스와 기술 문서

코드는 [Apache-2.0](../LICENSE), Geist 폰트는 [OFL-1.1](../LICENSES/OFL-1.1.txt)입니다. 모델과 데모에는 각각의 이용 조건이 적용되며 [제3자 고지](../THIRD_PARTY_NOTICES.md)에 설명되어 있습니다. 입력 자료를 다운로드하고 가공할 권한은 사용자가 확인해야 합니다.

Python API, 테스트, 기술 세부 사항은 [영어 README](../README.md#python-api)를 참고하세요.
