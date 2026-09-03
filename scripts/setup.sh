#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAV_DATA_ROOT="${PODCAST_ALIGN_VIDEO_DATA_ROOT:-${HOME}/.local/share/podcast-align-video}"
UV_VERSION="0.12.9"
MICROMAMBA_VERSION="2.9.0"
MICROMAMBA_SHA256="8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      PAV_DATA_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "podcast-align-video v0.1 setup supports Linux and WSL2 only." >&2
  exit 1
fi

mkdir -p "${PAV_DATA_ROOT}/envs" "${PAV_DATA_ROOT}/tools"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

supports_symlinks() {
  local probe_dir="${PAV_DATA_ROOT}/.symlink-probe-$$"
  mkdir -p "${probe_dir}"
  : > "${probe_dir}/target"
  if ln -s target "${probe_dir}/link" 2>/dev/null; then
    rm -f "${probe_dir}/link" "${probe_dir}/target"
    rmdir "${probe_dir}"
    return 0
  fi
  rm -f "${probe_dir}/target"
  rmdir "${probe_dir}"
  return 1
}

if supports_symlinks; then
  DATA_ROOT_SUPPORTS_SYMLINKS=1
  UV_CACHE_DIR="${PAV_DATA_ROOT}/uv-cache"
  UV_PYTHON_INSTALL_DIR="${PAV_DATA_ROOT}/tools/python"
  BROWSER_CACHE="${PAV_DATA_ROOT}/browser-cache"
  rm -f "${PAV_DATA_ROOT}/browser-cache.location"
else
  DATA_ROOT_SUPPORTS_SYMLINKS=0
  DATA_KEY="$(printf '%s' "${PAV_DATA_ROOT}" | sha256sum | cut -c1-12)"
  WSL_RUNTIME_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}/podcast-align-video/runtimes/${DATA_KEY}"
  UV_CACHE_DIR="${WSL_RUNTIME_ROOT}/uv-cache"
  UV_PYTHON_INSTALL_DIR="${WSL_RUNTIME_ROOT}/python"
  BROWSER_CACHE="${WSL_RUNTIME_ROOT}/browser-cache"
  mkdir -p "${WSL_RUNTIME_ROOT}"
  printf '%s\n' "${BROWSER_CACHE}" > "${PAV_DATA_ROOT}/browser-cache.location.tmp"
  mv "${PAV_DATA_ROOT}/browser-cache.location.tmp" "${PAV_DATA_ROOT}/browser-cache.location"
fi
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR
mkdir -p "${BROWSER_CACHE}"

UV_BIN="${PAV_DATA_ROOT}/tools/uv/uv"
if [[ ! -x "${UV_BIN}" ]] || [[ "$("${UV_BIN}" --version | awk '{print $2}')" != "${UV_VERSION}" ]]; then
  UV_INSTALL_DIR="${PAV_DATA_ROOT}/tools/uv"
  curl --proto '=https' --tlsv1.2 -LsSf \
    "https://astral.sh/uv/${UV_VERSION}/install.sh" \
    -o "${TEMP_DIR}/uv-installer.sh"
  env UV_INSTALL_DIR="${UV_INSTALL_DIR}" UV_NO_MODIFY_PATH=1 sh "${TEMP_DIR}/uv-installer.sh"
fi
[[ "$("${UV_BIN}" --version | awk '{print $2}')" == "${UV_VERSION}" ]] || {
  echo "uv version verification failed" >&2
  exit 1
}

"${UV_BIN}" python install 3.12 --no-bin
PYTHON_BIN="$("${UV_BIN}" python find 3.12)"

# CPython normally creates a lib64 -> lib symlink. Some WSL-mounted Windows
# volumes disallow symlinks, so pre-creating lib64 keeps venv fully copy-based.
create_venv() {
  local target="$1"
  mkdir -p "${target}/lib64"
  "${PYTHON_BIN}" -m venv --copies "${target}"
}

create_venv "${PAV_DATA_ROOT}/envs/core"
"${UV_BIN}" pip sync \
  --python "${PAV_DATA_ROOT}/envs/core/bin/python" \
  --link-mode copy "${PROJECT_ROOT}/envs/core-lock.txt"
"${UV_BIN}" pip install \
  --python "${PAV_DATA_ROOT}/envs/core/bin/python" \
  --link-mode copy --no-deps -e "${PROJECT_ROOT}"

create_venv "${PAV_DATA_ROOT}/envs/cohere"
"${UV_BIN}" pip sync \
  --python "${PAV_DATA_ROOT}/envs/cohere/bin/python" \
  --link-mode copy "${PROJECT_ROOT}/envs/cohere-lock.txt"

create_venv "${PAV_DATA_ROOT}/envs/qwen"
"${UV_BIN}" pip sync \
  --python "${PAV_DATA_ROOT}/envs/qwen/bin/python" \
  --link-mode copy "${PROJECT_ROOT}/envs/qwen-lock.txt"

MICROMAMBA_BIN="${PAV_DATA_ROOT}/tools/micromamba/bin/micromamba"
if [[ ! -x "${MICROMAMBA_BIN}" ]] || [[ "$("${MICROMAMBA_BIN}" --version)" != "${MICROMAMBA_VERSION}" ]]; then
  mkdir -p "$(dirname "${MICROMAMBA_BIN}")"
  curl --proto '=https' --tlsv1.2 -LsSf \
    "https://micro.mamba.pm/api/micromamba/linux-64/${MICROMAMBA_VERSION}" \
    -o "${TEMP_DIR}/micromamba.tar.bz2"
  printf '%s  %s\n' "${MICROMAMBA_SHA256}" "${TEMP_DIR}/micromamba.tar.bz2" | sha256sum --check --status
  tar -xjf "${TEMP_DIR}/micromamba.tar.bz2" -C "${TEMP_DIR}" bin/micromamba
  install -m 0755 "${TEMP_DIR}/bin/micromamba" "${MICROMAMBA_BIN}"
fi
[[ "$("${MICROMAMBA_BIN}" --version)" == "${MICROMAMBA_VERSION}" ]] || {
  echo "micromamba version verification failed" >&2
  exit 1
}

MFA_LOCATION_FILE="${PAV_DATA_ROOT}/envs/mfa.location"
if [[ "${DATA_ROOT_SUPPORTS_SYMLINKS}" == "1" ]]; then
  MAMBA_ROOT="${PAV_DATA_ROOT}/micromamba-root"
  MFA_ENV_PREFIX="${PAV_DATA_ROOT}/envs/mfa"
  rm -f "${MFA_LOCATION_FILE}"
else
  # Conda packages contain real Unix symlinks even with --always-copy. DrvFS
  # mounts without metadata reject them, so keep only this executable runtime
  # on WSL's Linux filesystem. Models, checkpoints, caches, and outputs remain
  # under PAV_DATA_ROOT. The location file is local operational state.
  MAMBA_ROOT="${WSL_RUNTIME_ROOT}/micromamba-root"
  MFA_ENV_PREFIX="${WSL_RUNTIME_ROOT}/mfa"
  mkdir -p "${WSL_RUNTIME_ROOT}"
  printf '%s\n' "${MFA_ENV_PREFIX}" > "${MFA_LOCATION_FILE}.tmp"
  mv "${MFA_LOCATION_FILE}.tmp" "${MFA_LOCATION_FILE}"
  echo "PAV_DATA_ROOT does not support Unix symlinks; MFA runtime: ${MFA_ENV_PREFIX}" >&2
fi

MFA_ENV_FINGERPRINT="$(
  {
    printf 'micromamba=%s\n' "${MICROMAMBA_VERSION}"
    sha256sum "${PROJECT_ROOT}/envs/mfa-environment.yml"
  } | sha256sum | cut -d' ' -f1
)"
MFA_ENV_MARKER="${MFA_ENV_PREFIX}/.podcast-align-video-environment"
if [[ ! -x "${MFA_ENV_PREFIX}/bin/mfa" ]] \
  || [[ ! -f "${MFA_ENV_MARKER}" ]] \
  || [[ "$(<"${MFA_ENV_MARKER}")" != "${MFA_ENV_FINGERPRINT}" ]]; then
  MAMBA_ROOT_PREFIX="${MAMBA_ROOT}" \
    "${MICROMAMBA_BIN}" --no-rc create -y --always-copy --override-channels -c conda-forge \
    -p "${MFA_ENV_PREFIX}" \
    -f "${PROJECT_ROOT}/envs/mfa-environment.yml"
  printf '%s\n' "${MFA_ENV_FINGERPRINT}" > "${MFA_ENV_MARKER}.tmp"
  mv "${MFA_ENV_MARKER}.tmp" "${MFA_ENV_MARKER}"
else
  echo "Reusing MFA ${MFA_ENV_PREFIX}"
fi

PLAYWRIGHT_BROWSERS_PATH="${BROWSER_CACHE}" \
  "${PAV_DATA_ROOT}/envs/core/bin/python" -m playwright install chromium

mkdir -p "${HOME}/.local/bin"
ln -sfn "${PAV_DATA_ROOT}/envs/core/bin/podcast-align-video" "${HOME}/.local/bin/podcast-align-video"

echo
echo "User-space environments are ready under: ${PAV_DATA_ROOT}"
echo "No sudo or apt command was run."
echo "Next: podcast-align-video models fetch --config /path/to/config.toml"
echo "Then: podcast-align-video doctor --config /path/to/config.toml"
