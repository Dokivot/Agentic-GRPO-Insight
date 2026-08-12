#!/usr/bin/env bash
# Sync results/ directory from cloud server to local repo via rsync.
# Ensures cloud+local dual-copy of all experiment metrics.
#
# Usage:
#   bash scripts/sync_results.sh user@cloud-host
#   bash scripts/sync_results.sh user@cloud-host --include-trajectories
#
# By default, trajectory JSON files are excluded to save bandwidth.
# Use --include-trajectories to sync everything.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOCAL_RESULTS="$PROJECT_ROOT/results"

CLOUD_HOST="${1:?Usage: bash scripts/sync_results.sh user@cloud-host [--include-trajectories]}"
shift

INCLUDE_TRAJ=false
if [ "${1:-}" = "--include-trajectories" ]; then
    INCLUDE_TRAJ=true
fi

CLOUD_RESULTS="${CLOUD_RESULTS_DIR:-~/DoProj/results}"

echo "Syncing results from $CLOUD_HOST:$CLOUD_RESULTS to $LOCAL_RESULTS"
echo "Include trajectories: $INCLUDE_TRAJ"
echo ""

EXCLUDE_ARGS=""
if [ "$INCLUDE_TRAJ" = false ]; then
    EXCLUDE_ARGS="--exclude=trajectories/"
    echo "Note: Trajectory files excluded. Use --include-trajectories to include."
fi

mkdir -p "$LOCAL_RESULTS"

rsync -avz --checksum \
    $EXCLUDE_ARGS \
    "$CLOUD_HOST:$CLOUD_RESULTS/" \
    "$LOCAL_RESULTS/"

echo ""
echo "Sync complete."
echo "Local results directory:"
du -sh "$LOCAL_RESULTS" 2>/dev/null || true
find "$LOCAL_RESULTS" -type f | wc -l | xargs echo "Total files:"
