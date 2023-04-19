#!/bin/sh

python retriever/convert_md2d_train_data_to_splade.py
python retriever/dpr_negatives_for_training.py