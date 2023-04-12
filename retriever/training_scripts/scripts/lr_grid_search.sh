#!/usr/bin/bash

for i in {1..10}; do
    echo "LR: " ${i}e-6
    python splade_md2d.py \
        --model_name splade_weights/splade_max \
        --use_all_queries \
        --data_path ../../data/mdd_dpr/beir_format \
        --lr ${i}e-6 \
        --epochs 10 \
        --use_all_queries
done