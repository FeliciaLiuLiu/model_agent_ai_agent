#!/usr/bin/env bash
set -euo pipefail

if ! command -v mmdc >/dev/null 2>&1; then
  echo "mmdc not found. Install @mermaid-js/mermaid-cli first." >&2
  echo "Example: npm install -g @mermaid-js/mermaid-cli" >&2
  exit 1
fi

for f in ./*.mmd; do
  base="${f%.mmd}"
  mmdc -i "$f" -o "${base}.svg"
  mmdc -i "$f" -o "${base}.png"
  echo "Rendered ${base}.svg and ${base}.png"
done
