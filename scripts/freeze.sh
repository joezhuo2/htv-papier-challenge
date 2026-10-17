#!/usr/bin/env bash
# Freeze a finished recording for the demo: scripts/freeze.sh <session-folder> [--as <file>]
# Default copies events.jsonl to demo/replay.jsonl and the exports to docs/examples/exports.
# --as <file> copies only the events to demo/<file> (a fallback) and leaves replay.jsonl and the exports alone.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
name=${1:?usage: scripts/freeze.sh <session-folder> [--as <file>]}
dest=replay.jsonl
if [[ ${2:-} == --as ]]; then dest=${3:?--as needs a file name}; fi
src=$root/runs/sessions/$name
log=$src/events.jsonl

[[ -f $log ]] || { echo "no $log" >&2; exit 1; }
last=$(grep -v '^\s*$' "$log" | tail -1)
[[ $last == *'"type": "session_finished"'* ]] || { echo "$log does not end with session_finished" >&2; exit 1; }
grep -q '"type": "final_report"' "$log" || { echo "$log has no final_report event" >&2; exit 1; }

mkdir -p "$root/demo"
cp "$log" "$root/demo/$dest"
if [[ $dest == replay.jsonl ]]; then
  out=$root/docs/examples/exports
  rm -rf "$out"
  mkdir -p "$out"
  shopt -s nullglob
  files=("$src"/exports/*.md)
  (( ${#files[@]} )) && cp "${files[@]}" "$out/"
fi

if grep -rEil 'nvapi-|AIza|sk-|PAPIERS_TOKEN|api_key|api-key' "$root/demo" "$root/docs/examples"; then
  echo "secret pattern found in the files above" >&2
  exit 1
fi
echo clean

python3 - "$log" "$dest" <<'PY'
import json, sys
events = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
report = next(e for e in reversed(events) if e["type"] == "final_report")["report"]
best = report["best"]
t = events[-1]["t"]
print(f"frozen to demo/{sys.argv[2]}")
print(f"events: {len(events)}")
print(f"length: {t} ms ({t // 60000}m {t // 1000 % 60:02d}s)")
print(f"best: {best['run']} ({best.get('hypothesis')}, {best.get('branch')}), {best['vs_naive']}x vs naive, {best['vs_eager']}x vs eager")
print(f"verdict: {best.get('verdict')}")
PY
