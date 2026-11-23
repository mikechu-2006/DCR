#!/usr/bin/env bash
# Parallel ranged download of a data.ppy.sh dump with integrity check.
#
#   scripts/fetch_dump_parallel.sh <url> <out.tar.bz2> [connections]
#
# Why not one curl: the host serves ~0.25 MB/s per connection but scales to
# ~1.5 MB/s over 8 connections.  Ranges are inclusive and sized with ceil() so
# every byte is covered exactly once; the total size and the bzip2 CRC are both
# verified before the file is accepted.
set -euo pipefail

URL="${1:?usage: fetch_dump_parallel.sh <url> <out.tar.bz2> [connections]}"
OUT="${2:?usage: fetch_dump_parallel.sh <url> <out.tar.bz2> [connections]}"
N="${3:-8}"
TMP="${OUT}.parts"
RETRIES=4

SIZE=$(curl -sSI -m 30 "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
[ -n "$SIZE" ] || { echo "no content-length for $URL" >&2; exit 1; }

mkdir -p "$TMP"   # keep completed parts so a retry only refetches what is missing
CHUNK=$(( (SIZE + N - 1) / N ))

fetch_part() {
  local i=$1 s=$2 e=$3 exp=$(( $3 - $2 + 1 )) p="$TMP/part_$1" try=0 got=0
  got=$(stat -c%s "$p" 2>/dev/null || echo 0)
  [ "$got" -eq "$exp" ] && return 0
  while [ "$try" -lt "$RETRIES" ]; do
    try=$((try+1))
    # --speed-limit/--speed-time: a stalled stream otherwise sits until -m 900
    curl -sSL -m 900 --speed-limit 50000 --speed-time 45 -r "$s-$e" -o "$p" "$URL" || true
    got=$(stat -c%s "$p" 2>/dev/null || echo 0)
    [ "$got" -eq "$exp" ] && return 0
    echo "part $i: got $got / $exp (try $try)" >&2
  done
  return 1
}

pids=()
for ((i=0; i<N; i++)); do
  s=$(( i * CHUNK )); e=$(( s + CHUNK - 1 )); [ "$e" -ge "$SIZE" ] && e=$(( SIZE - 1 ))
  [ "$s" -ge "$SIZE" ] && continue
  fetch_part "$i" "$s" "$e" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "a part failed; keeping $TMP for resume" >&2; exit 1; }

for ((i=0; i<N; i++)); do [ -f "$TMP/part_$i" ] && cat "$TMP/part_$i"; done > "$OUT"
GOT=$(stat -c%s "$OUT")
[ "$GOT" -eq "$SIZE" ] || { echo "size mismatch: $GOT != $SIZE" >&2; exit 1; }
if [ "${SKIP_BZ2_TEST:-0}" != "1" ]; then
  bzip2 -t "$OUT" || { echo "bzip2 integrity check failed: $OUT" >&2; exit 1; }
else
  echo "SKIP_BZ2_TEST=1: relying on per-part size checks and tar's own CRC"
fi
rm -rf "$TMP"
echo "OK $OUT ($SIZE bytes)"
