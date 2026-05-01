#!/usr/bin/env bash
set -euo pipefail

repo="${PAWAHARA_REPO:-mizuamedesu/PawaharaHarness}"
version="${PAWAHARA_VERSION:-latest}"
install_dir="${PAWAHARA_INSTALL_DIR:-$HOME/.local/bin}"
download_url="${PAWAHARA_DOWNLOAD_URL:-}"
binary_name="pawahara-harness"

usage() {
  cat <<'EOF'
Install Pawahara Harness on Linux.

Usage:
  ./install.sh [--version v0.1.0] [--install-dir ~/.local/bin]

Environment:
  PAWAHARA_REPO          GitHub repo in owner/name form. Default: mizuamedesu/PawaharaHarness
  PAWAHARA_VERSION       Release tag to install. Default: latest
  PAWAHARA_INSTALL_DIR   Directory for the binary. Default: ~/.local/bin
  PAWAHARA_DOWNLOAD_URL  Override archive URL.

Examples:
  curl -fsSL https://raw.githubusercontent.com/mizuamedesu/PawaharaHarness/main/install.sh | bash
  PAWAHARA_VERSION=v0.1.0 ./install.sh
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

expand_path() {
  case "$1" in
    "~")
      printf '%s\n' "$HOME"
      ;;
    "~/"*)
      printf '%s/%s\n' "$HOME" "${1#~/}"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

download_file() {
  local url="$1"
  local output="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --connect-timeout 20 --output "$output" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
  else
    die "curl or wget is required"
  fi
}

path_entry_for_shell() {
  case "$install_dir" in
    "$HOME")
      printf '$HOME'
      ;;
    "$HOME"/*)
      printf '$HOME/%s' "${install_dir#"$HOME"/}"
      ;;
    *)
      printf '%s' "$install_dir"
      ;;
  esac
}

append_path_to_rc() {
  local rc_file="$1"
  local path_entry="$2"
  local marker_begin="# >>> pawahara-harness installer >>>"
  local marker_end="# <<< pawahara-harness installer <<<"

  mkdir -p "$(dirname "$rc_file")"
  touch "$rc_file"
  if grep -Fq "$marker_begin" "$rc_file"; then
    return
  fi

  {
    printf '\n%s\n' "$marker_begin"
    printf 'case ":$PATH:" in\n'
    printf '  *":%s:"*) ;;\n' "$path_entry"
    printf '  *) export PATH="%s:$PATH" ;;\n' "$path_entry"
    printf 'esac\n'
    printf '%s\n' "$marker_end"
  } >>"$rc_file"
}

ensure_path() {
  case ":$PATH:" in
    *":$install_dir:"*)
      return
      ;;
  esac

  local path_entry
  path_entry="$(path_entry_for_shell)"
  append_path_to_rc "$HOME/.profile" "$path_entry"

  if [ -f "$HOME/.bashrc" ] || [ "${SHELL##*/}" = "bash" ]; then
    append_path_to_rc "$HOME/.bashrc" "$path_entry"
  fi
  if [ -f "$HOME/.zshrc" ] || [ "${SHELL##*/}" = "zsh" ]; then
    append_path_to_rc "$HOME/.zshrc" "$path_entry"
  fi

  printf 'Added %s to PATH in your shell profile.\n' "$install_dir"
  printf 'Restart your shell or run: export PATH="%s:$PATH"\n' "$install_dir"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --version)
      [ "$#" -ge 2 ] || die "--version requires a value"
      version="$2"
      shift 2
      ;;
    --install-dir)
      [ "$#" -ge 2 ] || die "--install-dir requires a value"
      install_dir="$2"
      shift 2
      ;;
    --repo)
      [ "$#" -ge 2 ] || die "--repo requires a value"
      repo="$2"
      shift 2
      ;;
    --download-url)
      [ "$#" -ge 2 ] || die "--download-url requires a value"
      download_url="$2"
      shift 2
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[ "$(uname -s)" = "Linux" ] || die "this installer currently supports Linux only"

case "$(uname -m)" in
  x86_64|amd64)
    archive_name="pawahara-harness-linux-x86_64.tar.gz"
    ;;
  *)
    die "no prebuilt Linux binary is published for $(uname -m)"
    ;;
esac

install_dir="$(expand_path "$install_dir")"
mkdir -p "$install_dir"

if [ -z "$download_url" ]; then
  if [ "$version" = "latest" ]; then
    download_url="https://github.com/$repo/releases/latest/download/$archive_name"
  else
    download_url="https://github.com/$repo/releases/download/$version/$archive_name"
  fi
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

archive_path="$tmpdir/$archive_name"
printf 'Downloading %s\n' "$download_url"
download_file "$download_url" "$archive_path"

tar -xzf "$archive_path" -C "$tmpdir"
[ -f "$tmpdir/$binary_name" ] || die "archive did not contain $binary_name"

install -m 755 "$tmpdir/$binary_name" "$install_dir/$binary_name"
"$install_dir/$binary_name" --help >/dev/null

ensure_path
printf 'Installed %s to %s\n' "$binary_name" "$install_dir/$binary_name"
