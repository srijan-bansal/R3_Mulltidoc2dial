from train_reranker import train

from utils import setup_logging
    
config = {
    "encoder": "roberta-base",
    "train": "../data/mdd_all/dd-generation-structure/train.source",
    "val": "../data/mdd_all/dd-generation-structure/val.source",
    "train_psg": "../data/retrieved_results/splade-results.tsv",
    "val_psg": "../data/retrieved_results/splade-results.tsv",
    "train_gold_pids": "../data/mdd_all/dd-generation-structure/train.pids",
    "val_gold_pids": "../data/mdd_all/dd-generation-structure/val.pids",
    "hard_negatives": None,
    "checkpoint_dir": "checkpoints",
    "cache_dir": "cache",
    "max_length": 300,
    "train_batch_size": 24,
    "eval_batch_size": 100,
    "lr": 2e-05,
    "num_epoch": 5,
    "iter_size": 8,
    "criterion": "CE",
    "no_gpu": False,
    "fp16": False
}

if __name__ == "__main__":
    setup_logging(logpath="logs/")
    train(config)