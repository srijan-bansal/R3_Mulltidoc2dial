#!/usr/bin/bash

python retriever/run_inference.py \
--model_name distilsplade \
--model_path /projects/tir6/general/adityasv/0_MLMTransformer \
--data_dir data/beir_format \
--dump_dir retrieved_results \
--split validation

python retriever/run_eval.py \
--data_dir data/beir_format \
--retrieved_results_path retrieved_results/distilsplade-results.tsv \
--split validation

