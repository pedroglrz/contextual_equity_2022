#!/bin/bash
DATA_DIR='/scratch/ppg228/did_project/data'
DATA='2022-05-01_did_processed_data_2021.csv'
GRID_SIZE='medium_grid2'

python ./magicloop_did.py \
    --data_dir $DATA_DIR \
    --data $DATA \
    --grid_size $GRID_SIZE