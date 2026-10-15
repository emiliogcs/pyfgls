#!/usr/bin/env bash
# Update https://emiliogcs.com/code/ from this repository's README, docs and docstrings.
#
#   scripts/sync_site.sh path/to/website-checkout
#
# Installs this checkout with the `site` extra into a throwaway virtual environment,
# builds the page with scripts/build_site_page.py (which runs the Quickstart examples),
# writes it into the website's code/ folder and lists what changed there. It commits
# and pushes nothing: review the diff in the website checkout, commit it on a branch and
# open a pull request. Merging into the website's main branch publishes the page.
#
# Needs Python >= 3.10 (set PYTHON=/path/to/python3.x if `python3` is older).
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
site="${1:-$here/../emiliogcs.github.io}"
py="${PYTHON:-python3}"

if [ ! -f "$site/code/index.html" ]; then
  echo "No code/index.html in $site: pass the website checkout as the first argument." >&2
  exit 1
fi
if ! "$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  echo "$py is older than Python 3.10: set PYTHON=/path/to/python3.x" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
"$py" -m venv "$tmp/venv"
"$tmp/venv/bin/python" -m pip install --quiet --upgrade pip
"$tmp/venv/bin/python" -m pip install --quiet -e "$here[site]"
"$tmp/venv/bin/python" "$here/scripts/build_site_page.py" --site "$site"

echo
git -C "$site" status --short -- code
