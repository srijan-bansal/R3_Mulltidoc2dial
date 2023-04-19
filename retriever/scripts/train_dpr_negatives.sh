#!/usr/bin/bash

python retriever/splade_md2d.py \
    --model_name naver/splade-cocondenser-selfdistil \
    --use_all_queries \
    --data_path data/dpr_negatives_beir_format \
    --lr 1e-6 \
    --epochs 20 