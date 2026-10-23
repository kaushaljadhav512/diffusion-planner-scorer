#!/usr/bin/env bash
# Download and lay out the nuPlan splits needed for the baselines.
#
# nuPlan links are personal and expire: log in at https://www.nuscenes.org/nuplan#download,
# copy the links for the files below into a text file (one URL per line), then:
#   bash setup/download_nuplan.sh urls.txt
#
# Needed for baseline reproduction:   maps, val (Val14), test (Test14-hard)
# Needed later for training / labels: one or more train_* splits (use a subset), mini for debugging
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.env"

URLS="${1:?usage: download_nuplan.sh <file-with-one-url-per-line>}"
DL="$NUPLAN_HOME/downloads"
mkdir -p "$DL"

echo "[download] $(grep -c . "$URLS") files -> $DL"
aria2c -c -x 8 -s 8 -j 2 --auto-file-renaming=false -d "$DL" -i "$URLS"

# Map each archive to its destination. Archives nest DBs at varying depths, so we
# extract to a staging dir and move every *.db to the layout nuplan-devkit expects.
dest_for() {
  case "$1" in
    *maps*)  echo "$NUPLAN_MAPS_ROOT" ;;
    *mini*)  echo "$NUPLAN_DATA_ROOT/nuplan-v1.1/mini" ;;
    *test*)  echo "$NUPLAN_DATA_ROOT/nuplan-v1.1/test" ;;
    *val*|*train*) echo "$NUPLAN_DATA_ROOT/nuplan-v1.1/trainval" ;;
    *) echo "" ;;
  esac
}

for zip in "$DL"/*.zip; do
  name="$(basename "$zip" .zip)"
  marker="$DL/.$name.extracted"
  [ -f "$marker" ] && { echo "[skip] $name already extracted"; continue; }
  dest="$(dest_for "$name")"
  [ -n "$dest" ] || { echo "[warn] don't know where $name goes, skipping"; continue; }
  echo "[extract] $name -> $dest"
  if [[ "$name" == *maps* ]]; then
    stage="$DL/stage_$name"; unzip -q -o "$zip" -d "$stage"
    # maps zip contains a top-level "maps/" dir holding nuplan-maps-v1.0.json and one dir per city
    src="$(dirname "$(find "$stage" -name 'nuplan-maps-v1.0.json' | head -1)")"
    cp -rn "$src"/. "$NUPLAN_MAPS_ROOT"/ && rm -rf "$stage"
  else
    stage="$DL/stage_$name"; unzip -q -o "$zip" -d "$stage"
    find "$stage" -name '*.db' -exec mv -n {} "$dest"/ \;
    rm -rf "$stage"
  fi
  touch "$marker"
done

echo
echo "[layout]"
for d in trainval test mini; do
  printf "  nuplan-v1.1/%-9s %5s DBs\n" "$d" "$(find "$NUPLAN_DATA_ROOT/nuplan-v1.1/$d" -name '*.db' | wc -l)"
done
echo "  maps: $(ls "$NUPLAN_MAPS_ROOT" | tr '\n' ' ')"
df -h "$NUPLAN_HOME" | tail -1
echo "Zips kept in $DL; delete them once the layout above looks right to free disk."
