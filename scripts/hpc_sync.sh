#!/bin/bash
# hpc_sync.sh - deploy the D2V attribute-channel harness to the HKUST-GZ HPC and
# pull results back.  The HPC checkout (~/DCR) is NOT a git repo; this is the transport.
#
#   scripts/hpc_sync.sh code [--dry-run]   # local -> HPC: python code, configs, job scripts
#   scripts/hpc_sync.sh data [--dry-run]   # local -> HPC: the 1k / P4 / attr input tables
#   scripts/hpc_sync.sh pull [--dry-run]   # HPC -> local: attr content tables + probe results
#
# Every subcommand accepts --dry-run (or DRY_RUN=1); dry-run still ssh-lists the
# remote side so you can see exactly what would move.
# Endpoint overrides: HPC_USER (default aiaa4051_083), HPC_HOST
# (default hpc2login.hpc.hkust-gz.edu.cn), HPC_DIR (default DCR, relative to remote $HOME).
#
# Deliberate non-goals:
#   * no --delete: the HPC side holds checkpoints (data2vec_main/runs/) and probe
#     results that do not exist locally; mirroring would silently destroy them.
#   * 'code' never ships runs/ cache/ splits/ data/ *.pt *.npz *.parquet, so the
#     remote 737 MB 8ch checkpoint and the processed tables are never disturbed.
#     New split files under data2vec_main/splits/ must be shipped explicitly.
#
# Typical workflow:
#   scripts/hpc_sync.sh code                     # deploy train/extract/probe scripts
#   scripts/hpc_sync.sh data                     # ship 1k pairbest / P4 seeds / signal table
#   ssh ... cd ~/DCR && export PATH=/opt/slurm/bin:$PATH && sbatch data2vec_main/scripts/attr_train.sbatch
#   scripts/hpc_sync.sh pull                     # bring results/d2v_attr*/ home
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HPC_USER="${HPC_USER:-aiaa4051_083}"
HPC_HOST="${HPC_HOST:-hpc2login.hpc.hkust-gz.edu.cn}"
HPC_DIR="${HPC_DIR:-DCR}"          # repo path relative to the remote $HOME
HPC="${HPC_USER}@${HPC_HOST}"

DRY_RUN="${DRY_RUN:-0}"
RSYNC_OPTS=(-az --info=progress2)

say() { printf '[hpc_sync] %s\n' "$*"; }
die() { printf '[hpc_sync] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

remote_preflight() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$HPC" true \
    || die "ssh $HPC failed (check the key / VPN)"
}

remote_mkdirs() {
  local dirs=("data2vec_main/d2v_main" "data2vec_main/scripts" "data2vec_main/configs"
              "dcr" "scripts/slurm" "data/processed" "results")
  if [ "$DRY_RUN" = 1 ]; then
    say "[dry-run] would ssh $HPC mkdir -p $HPC_DIR/{$(IFS=,; echo "${dirs[*]}")}"
    return 0
  fi
  local -a mk=()
  local d
  for d in "${dirs[@]}"; do mk+=("'$HPC_DIR/$d'"); done
  ssh "$HPC" "mkdir -p ${mk[*]}" || die "remote mkdir failed"
}

# rsync helper: forwards the shared options; extra args (sources + dest) follow.
# Dry runs add -v so every would-transfer path is printed by name.
xfer() {
  if [ "$DRY_RUN" = 1 ]; then
    rsync "${RSYNC_OPTS[@]}" -v --dry-run "$@"
  else
    rsync "${RSYNC_OPTS[@]}" "$@"
  fi
}

# ---------------------------------------------------------------------------
# code: data2vec_main/{d2v_main,scripts,configs} + dcr/ + scripts/*.py + scripts/slurm/*.sbatch
# ---------------------------------------------------------------------------
sync_code() {
  say "== code -> $HPC:$HPC_DIR =="
  remote_preflight
  remote_mkdirs

  local -a excl=(
    --exclude='__pycache__/'
    --exclude='runs/'
    --exclude='cache/'
    --exclude='splits/'
    --exclude='data/'
    --exclude='*.pt'
    --exclude='*.npz'
    --exclude='*.parquet'
  )

  xfer "${excl[@]}" data2vec_main/d2v_main/  "$HPC:$HPC_DIR/data2vec_main/d2v_main/"
  xfer "${excl[@]}" data2vec_main/scripts/  "$HPC:$HPC_DIR/data2vec_main/scripts/"
  xfer "${excl[@]}" data2vec_main/configs/  "$HPC:$HPC_DIR/data2vec_main/configs/"
  xfer "${excl[@]}" dcr/                    "$HPC:$HPC_DIR/dcr/"

  local -a py=(scripts/*.py)
  if [ -e "${py[0]}" ]; then
    xfer "${excl[@]}" "${py[@]}" "$HPC:$HPC_DIR/scripts/"
  else
    say "WARNING: no scripts/*.py found"
  fi

  local -a sb=(scripts/slurm/*.sbatch)
  if [ -e "${sb[0]}" ]; then
    xfer "${excl[@]}" "${sb[@]}" "$HPC:$HPC_DIR/scripts/slurm/"
  else
    say "WARNING: no scripts/slurm/*.sbatch found"
  fi

  say "code sync done ($(date +%H:%M:%S)); local sources:"
  local code_dirs=(data2vec_main/d2v_main data2vec_main/scripts data2vec_main/configs dcr)
  du -sh "${code_dirs[@]}" 2>/dev/null | sed 's/^/[hpc_sync]   /' || true
  say "  scripts/*.py: ${#py[@]} files | scripts/slurm/*.sbatch: ${#sb[@]} files"
  if [ "$DRY_RUN" = 1 ]; then
    say "dry-run: nothing was written (rsync --dry-run)"
  fi
}

# ---------------------------------------------------------------------------
# data: the 1k pairbest / P4 seed splits / content+signal tables the attr probes need.
# Only files that exist locally are shipped; the rest are reported.
# ---------------------------------------------------------------------------
DATA_FILES=(
  step1_1k_pairbest.parquet
  step1_1k_pairbest_d2v_holdout.parquet
  rank_signals_1k.parquet
  chart_content_cm3p_local.parquet
  chart_content_d2v_holdout.parquet
  step1_mclean25_pairbest.parquet
  step1_mclean25_p4sets_s1.parquet
  charts_1k.txt
)
for _s in 1 2 3 4 5; do
  DATA_FILES+=("step1_1k_pairbest_p4_s${_s}.parquet")
  DATA_FILES+=("step1_1k_pairbest_p4_d2vhold_s${_s}.parquet")
done
unset _s

sync_data() {
  say "== data -> $HPC:$HPC_DIR/data/processed/ =="
  remote_preflight

  local -a present=() missing=()
  local f
  for f in "${DATA_FILES[@]}"; do
    if [ -f "data/processed/$f" ]; then
      present+=("data/processed/$f")
    else
      missing+=("$f")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    say "missing locally (skipped): ${missing[*]}"
  fi
  [ "${#present[@]}" -gt 0 ] || die "no data files to sync under data/processed/"

  remote_mkdirs
  local total
  total="$(du -ch "${present[@]}" | tail -1 | cut -f1)"
  say "shipping ${#present[@]} file(s), $total:"
  printf '[hpc_sync]   %s\n' "${present[@]}"

  xfer "${present[@]}" "$HPC:$HPC_DIR/data/processed/"

  say "data sync done ($(date +%H:%M:%S)); total $total"
  if [ "$DRY_RUN" = 1 ]; then
    say "dry-run: nothing was written (rsync --dry-run)"
  fi
}

# ---------------------------------------------------------------------------
# pull: attribute content tables + probe results that exist on the HPC side.
# ---------------------------------------------------------------------------
list_remote_attr_dirs() {
  ssh "$HPC" "cd '$HPC_DIR' 2>/dev/null || exit 0; for d in results/d2v_attr*/; do [ -d "\$d" ] && printf '%s\\n' "\${d%/}"; done; true" \
    || die "ssh $HPC failed while listing results/d2v_attr*/"
}

list_remote_content_tables() {
  ssh "$HPC" "cd '$HPC_DIR/data/processed' 2>/dev/null || exit 0; for f in chart_content_d2v_a*-*.parquet; do [ -f "\$f" ] && printf '%s\\n' "\$f"; done; true" \
    || die "ssh $HPC failed while listing chart_content_d2v_a*-*.parquet"
}

sync_pull() {
  say "== pull: $HPC:$HPC_DIR -> $REPO_ROOT =="
  remote_preflight

  local dirs tables ndir=0 ntab=0
  dirs="$(list_remote_attr_dirs)"
  tables="$(list_remote_content_tables)"

  if [ -n "$dirs" ]; then
    local d
    while IFS= read -r d; do
      [ -n "$d" ] || continue
      say "  results: $d/"
      [ "$DRY_RUN" = 1 ] || mkdir -p "$d"
      xfer "$HPC:$HPC_DIR/$d/" "$d/"
      ndir=$((ndir + 1))
    done <<< "$dirs"
  else
    say "  results: no results/d2v_attr*/ on the HPC side"
  fi

  if [ -n "$tables" ]; then
    [ "$DRY_RUN" = 1 ] || mkdir -p data/processed
    local f
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      say "  content table: data/processed/$f"
      xfer "$HPC:$HPC_DIR/data/processed/$f" "data/processed/"
      ntab=$((ntab + 1))
    done <<< "$tables"
  else
    say "  content tables: no data/processed/chart_content_d2v_a*-*.parquet on the HPC side"
  fi

  say "pull done: $ndir result dir(s), $ntab content table(s)"
  [ $((ndir + ntab)) -gt 0 ] || say "nothing to pull"
  if [ "$DRY_RUN" = 1 ]; then
    say "dry-run: nothing was written (rsync --dry-run)"
  fi
}

# ---------------------------------------------------------------------------
main() {
  command -v rsync >/dev/null || die "rsync not installed locally"
  [ "$#" -ge 1 ] || { usage; exit 2; }
  local cmd="$1"; shift
  local arg
  for arg in "$@"; do
    case "$arg" in
      --dry-run|-n) DRY_RUN=1 ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown option: $arg" ;;
    esac
  done
  if [ "$DRY_RUN" = 1 ]; then
    say "DRY RUN: no files will be written"
  fi
  case "$cmd" in
    code) sync_code ;;
    data) sync_data ;;
    pull) sync_pull ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 2 ;;
  esac
}

main "$@"
