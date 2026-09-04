#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if git grep -I -n -i -E 'PodcastPipeline|youtube-cohere|ryouta?|alxs([^0-9]|$)|70o4qscwhjo|Users[/\\][^/\\]+|/home/[^/]+|/mnt/[a-z]/|hf_[A-Za-z0-9]{20,}' -- ':!scripts/audit-release.sh'; then
  echo "Release audit failed: personal path, legacy name, or token-shaped text is tracked." >&2
  exit 1
fi

forbidden="$(git ls-files | grep -E '(^|/)(outputs|jobs|models|browser-cache|work)/|\.(wav|mp3|m4a|aac|flac|ogg|opus|mp4|mkv|webm|safetensors|bin|pt|pth|onnx|log)$' || true)"
if [[ -n "${forbidden}" ]]; then
  echo "Release audit failed: generated media, runtime state, model, or log is tracked:" >&2
  echo "${forbidden}" >&2
  exit 1
fi

test -f LICENSE
test -f LICENSES/OFL-1.1.txt
test -f THIRD_PARTY_NOTICES.md
test -f src/podcast_align_video/render/assets/fonts/Geist-590.ttf

if git ls-files --error-unmatch config/local.toml >/dev/null 2>&1; then
  echo "Release audit failed: config/local.toml is tracked." >&2
  exit 1
fi

echo "Release audit passed."
