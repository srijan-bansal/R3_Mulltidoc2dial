#!/usr/bin/bash

python retriever/splade_md2d.py \
    --model_name retriever/splade_weights/splade_max \
    --use_all_queries \
    --data_path data/mdd_dpr/beir_format \
    --lr 1e-6 \
    --epochs 20 \
    --use_all_queries