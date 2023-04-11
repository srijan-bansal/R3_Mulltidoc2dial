#!/usr/bin/bash

export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=1 

python splade_md2d.py \
    --model_name splade_weights/distilsplade_max \
    --use_all_queries \
    --data_path ../../data/mdd_dpr/dpr_negatives_beir_format \
    --lr 1e-6 \
    --epochs 20 