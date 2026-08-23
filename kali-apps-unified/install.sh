#!/usr/bin/env bash
# One-command installer for the kali-apps CLI toolkit.
#
# Prefers pipx: installs each command globally (pykali_hash, KALI_non-hash,
# KALI_hash, kali-tree, kali-tensor, kali-downloader, kali-prepare-metadata,
# kali-rf-distance) with no venv activation ever required.
#
# Falls back to a plain venv in ./.venv if pipx isn't available and can't be
# installed automatically.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: no python3/python found. Install Python 3.9+ and re-run." >&2
    exit 1
fi

COMMAND_LIST="    pykali_hash, KALI_non-hash, KALI_hash, kali-tree, kali-tensor,
    kali-downloader, kali-prepare-metadata, kali-rf-distance"

install_with_pipx() {
    echo "==> Found pipx — installing kali-apps globally (recommended)"
    pipx install . --force
    echo
    echo "==> Done. Commands installed globally, available in any new terminal:"
    echo "$COMMAND_LIST"
    echo
    echo "Try:"
    echo "    KALI_non-hash --help"
}

install_with_venv() {
    echo "==> pipx not available — falling back to a local virtual environment (.venv)"
    "$PYTHON_BIN" -m venv .venv

    # shellcheck disable=SC1091
    source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

    echo "==> Upgrading pip"
    pip install --upgrade pip >/dev/null

    echo "==> Installing kali-apps and dependencies"
    pip install .

    echo
    echo "==> Done. Commands installed in this environment:"
    echo "$COMMAND_LIST"
    echo
    echo "This is a venv install, so each new terminal session needs:"
    echo "    source \"$HERE/.venv/bin/activate\"    (Windows: .venv\\Scripts\\activate)"
    echo
    echo "Then run any command with --help, e.g.:"
    echo "    KALI_non-hash --help"
    echo
    echo "Tip: install pipx instead to skip the activation step entirely —"
    echo "     see README.md for platform-specific pipx install instructions."
}

if command -v pipx >/dev/null 2>&1; then
    install_with_pipx
else
    echo "==> pipx not found. Attempting to install it automatically..."
    if "$PYTHON_BIN" -m pip install --user pipx >/dev/null 2>&1 \
        && "$PYTHON_BIN" -m pipx ensurepath >/dev/null 2>&1; then
        echo "==> pipx installed. NOTE: you must open a new terminal for it to"
        echo "    be on your PATH, then re-run this script (./install.sh)."
        exit 0
    else
        echo "==> Could not auto-install pipx (no permissions, or offline)."
        install_with_venv
    fi
fi
