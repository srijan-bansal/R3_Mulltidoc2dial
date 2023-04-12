from train_reranker import train

from utils import setup_logging
    
config = {
    "encoder": "roberta-base",
    "train_source": "../data/mdd_all/dd-generation-structure/train.source",
    "val_source": "../data/mdd_all/dd-generation-structure/val.source", # replace with test for predictions
    "train_qid": "../data/mdd_all/dd-generation-structure/train.qids",
    "val_qid": "../data/mdd_all/dd-generation-structure/val.qids", # replace with test for predictions
    "train_psg": "../data/retrieved_results/splade-results.tsv", # tsv output of retriever for train
    "val_psg": "../data/retrieved_results/splade-results.tsv", # tsv output of retriever for val, replace with test for predictions
    "train_gold_pids": "../data/mdd_all/dd-generation-structure/train.pids",
    "val_gold_pids": "../data/mdd_all/dd-generation-structure/val.pids", # no need for test
    "checkpoint_dir": "checkpoints",
    "cache_dir": "cache",
    "max_length": 300,
    "train_batch_size": 24, # number of passages thats considered for training
    "no_passages" : 100, # number of passages that needs to be reranked per query
    "eval_batch_size": 32, # number of passages for evaluation split (because 100 passages cant be passed at once)
    "lr": 2e-05,
    "num_epoch": 5,
    "iter_size": 8, # effective batch size (accumulated gradients)
    "criterion": "CE",
    "no_gpu": False,
    "fp16": False,
    "output_dev_file" : "rerank_dev_predictions.tsv",
    "output_train_file" : "rerank_train_predictions.tsv",
    "evaluate" : False # make this true for just test predictions 
}

if __name__ == "__main__":
    setup_logging(logpath="logs/")
    train(config)