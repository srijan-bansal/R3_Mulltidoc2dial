from train_reranker import train

from utils import setup_logging
    
config = {
    "encoder": "roberta-base",
    "all_doc": "../data/multidoc2dial/multidoc2dial_doc.json",
    "train_source": "../data/mdd_all/dd-generation-structure/val.source",
    "val_source": "../data/mdd_all/dd-generation-structure/val.source", # replace with test for predictions
    "train_qid": "../data/mdd_all/dd-generation-structure/val.qids",
    "val_qid": "../data/mdd_all/dd-generation-structure/val.qids", # replace with test for predictions
    "train_psg": "../data/retrieved_results/splade-results.tsv", # tsv output of retriever for train
    "val_psg": "../data/retrieved_results/splade-results.tsv", # tsv output of retriever for val, replace with test for predictions
    "train_gold_pids": "../data/mdd_all/dd-generation-structure/val.pids",
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
    "evaluate" : True, # make this true for just test predictions
    "ckpt_path" : "checkpoints/reranker_1.ckpt", # set to checkpoint path for prediction
    "eval_freq" : 2500 # almost equivalent to len(train) / iter_size (cover 1 epoch)
}

if __name__ == "__main__":
    setup_logging(logpath="logs/")
    train(config)