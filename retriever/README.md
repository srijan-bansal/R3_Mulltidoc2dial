# Data setup

1. convert_md2d_train_data_to_splade.py
2. dpr_negatives_for_training.py

# Checkpoint setup
download weights from https://github.com/naver/splade/tree/main/weights and paste the weights directory to splade_weights

# Training scripts
```
# Fine tuning
bash scripts/train.sh

# DPR negatives
bash scripts/train_dpr_negatives.sh
```

# Inference and Evaluation

For inference, use
```
run_inference.py
```

For computing metrics, use
```
run_eval.py
```



## These scripts are based on https://github.com/naver/splade/tree/main