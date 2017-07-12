#!/bin/bash

DATA_DIR="$1"
DEST_DIR="$2"

for DATASET in "chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs"
do
    /usr/bin/env python3 link_7scenes.py \
		 --data_dir="${DATA_DIR}/${DATASET}" \
		 --dest_dir="${DEST_DIR}/7scenes_${DATASET}" \
		 --dry_run=False
done
