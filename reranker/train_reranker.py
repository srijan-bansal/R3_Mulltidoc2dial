import argparse
import logging
import os
import sys
import socket
import json
import pickle
import torch
import torch.nn as nn

from datetime import datetime
from transformers import AutoConfig, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, RandomSampler

from framework import RerankerFramework
from reranker_datasets import (EfficientQARerankerDatasetForBaselineReranker_TRAIN,
                        EfficientQARerankerDatasetForBaselineReranker, 
                        BaselineRerankerQueryBuilder)
from model import BaselineReranker
from utils import setup_logging


LOGGER = logging.getLogger(__name__)

from torch.nn import DataParallel as DataParallel_raw
import numpy as np

class DataParallel(DataParallel_raw):
    """
    we do the scatter outside of the DataPrallel.
    input: Scattered Inputs without kwargs.
    """

    def __init__(self, module):
        # Disable all the other parameters
        super(DataParallel, self).__init__(module)

    def custom_scatter(self, inputs):
        batch_size = inputs[0]['input_ids'].size(0)
        if len(self.device_ids) == 2:
            if batch_size % 2 == 0:
                per_gpu_size = [batch_size // 2 for _ in self.device_ids]
            else:
                per_gpu_size = [batch_size // 2, batch_size - (batch_size // 2)]
        elif len(self.device_ids) == 3:
            if batch_size % 3 == 0:
                per_gpu_size = [batch_size // 3 for _ in self.device_ids]
            elif batch_size % 3 == 1:
                per_gpu_size = [batch_size // 3, batch_size // 3, batch_size - (2 * batch_size // 3) ]
            else:
                per_gpu_size = [batch_size // 3, batch_size - (2 * batch_size // 3), batch_size - (2 * batch_size // 3) ]
        
        assert np.sum(per_gpu_size) == batch_size
        new_inputs = [{} for _ in self.device_ids]
        for key in inputs[0]:
            s = 0
            for i, _ in enumerate(self.device_ids):
                e = s + per_gpu_size[i]
                new_inputs[i][key] = inputs[0][key][s:e].to(self.device_ids[i])
                s = e
        new_inputs = [(x,) for x in new_inputs]
        return new_inputs

    def forward(self, *inputs, **kwargs):
        none_kwargs = [{} for _ in self.device_ids]
        new_inputs = self.custom_scatter(inputs)
        replicas = self.replicate(self.module, self.device_ids)
        outputs = self.parallel_apply(replicas, new_inputs, none_kwargs)
        return self.gather(outputs, self.output_device)

def build_parser():
    parser = argparse.ArgumentParser(description='Passages Reranker training process.')
    parser.add_argument("--config", default=None, help="")
    parser.add_argument("--train", default="../../data/mdd_all/train.source", help="train dataset")
    parser.add_argument("--val", default="../../data/mdd_all/val.source", help="validation dataset")
    parser.add_argument("--encoder", default="xlm-roberta-base", help="name or path to encoder")
    parser.add_argument("--cache_dir", default=None, help="cache directory")
    parser.add_argument("--max_length", default=512, type=int, help="maximum length of the input sequence")
    parser.add_argument("--checkpoint_dir", default=".checkpoints", help="directory to saving checkpoints")
    parser.add_argument("--no_gpu", action="store_true", help="no use GPU")
    parser.add_argument("--train_batch_size", default=20, type=int, help="mini-batch size")
    parser.add_argument("--eval_batch_size", default=100, type=int, help="mini-batch size")
    parser.add_argument("--iter_size", default=8, type=int, help="accumulated gradient")
    parser.add_argument("--num_epoch", default=5, type=int, help="number of epochs")
    parser.add_argument("--lr", default=1, type=int, help="learning rate")
    parser.add_argument("--fp16", action="store_true", help="train with fp16")
    parser.add_argument("--criterion", default=None, help="loss function (CE/BCE)")
    parser.add_argument("--ckpt_path", default=None, help="checkpoint to load for evaluation")
    parser.add_argument("--evaluate", action="store_true", help="evaluate the model")
    parser.add_argument("--output_file", default=None, help="output file name")
    parser.add_argument("--eval_freq", default=5000, help="(eval freq * accum_step) = no of training examples after which evaluation needs to be performed")
    return parser


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def binary_cross_entropy(pos_weight, device):
    def inner(logits):
        batch_size = logits.shape[0]
        one_hots = torch.zeros(batch_size, device=logits.get_device())
        one_hots[0] = 1.
        one_hots = one_hots.unsqueeze(1)
        return criterion(logits, one_hots)
    pos_weight = torch.tensor([pos_weight], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
    return inner


def get_dataloader_for_baseline_reranker(dataset, random_sampler=False):
    if random_sampler:
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            collate_fn=lambda batch: batch[0]
        )
    else:        
        dataloader = DataLoader(
            dataset,
            collate_fn=lambda batch: batch[0]
        )

    return dataloader

    
def train(args):

    # seed_everything(12345)
    LOGGER.info("Config: " + json.dumps(args, sort_keys=True, indent=2))
    evaluate = args["evaluate"]
    config = AutoConfig.from_pretrained(args["encoder"], cache_dir=args["cache_dir"])
    tokenizer = AutoTokenizer.from_pretrained(args["encoder"], cache_dir=args["cache_dir"], use_fast=False)

    LOGGER.info("Load datasets.")

    model_config = {
        "reranker_model_type": "baseline",
        "encoder": args["encoder"],
        "encoder_config": config,
        "max_length": args["max_length"]
    }

    query_builder = BaselineRerankerQueryBuilder(tokenizer, args["max_length"])
    
    train_dataloader = None
    if not evaluate:
        train_dataset = EfficientQARerankerDatasetForBaselineReranker_TRAIN(args["train"], tokenizer, query_builder, args["train_batch_size"])
        train_dataloader = get_dataloader_for_baseline_reranker(train_dataset, random_sampler=True)
        
    val_dataset = EfficientQARerankerDatasetForBaselineReranker(args["val"], query_builder, args["eval_batch_size"])
    val_dataloader = get_dataloader_for_baseline_reranker(val_dataset, random_sampler=False)

    LOGGER.info("Reranker training configuration: " + json.dumps(args, indent=4, sort_keys=True))
    LOGGER.info("Model initialization.")
    LOGGER.info(f"Cuda is available: {torch.cuda.is_available()}")

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args["no_gpu"] else "cpu")
    framework = RerankerFramework(device, model_config, train_dataloader, val_dataloader, args["output_file"])
    
    encoder = AutoModel.from_pretrained(args["encoder"], cache_dir=args["cache_dir"])
    
    
    
    model = BaselineReranker(config, encoder)
    # model = nn.DataParallel(model)
    if evaluate:
        # model_state_dict = framework.load_model(args['ckpt_path'])
        model_state_dict = framework.load_model(args['ckpt_path']).state_dict()
        #if hasattr(model, 'module') :
        #    model_state_dict = model_state_dict.module
        model.load_state_dict(model_state_dict)
    # model = nn.DataParallel(model)
    model = model.to(device)
    
    if not evaluate:
        save_ckpt = None

        checkpoint_name = "reranker_"
        checkpoint_name += args["encoder"].split('/')[-1]
        checkpoint_name += "_" + datetime.today().strftime('%Y-%m-%d-%H-%M')
        checkpoint_name += "_" + socket.gethostname()

        if args["checkpoint_dir"]:
            if not os.path.isdir(args["checkpoint_dir"]):
                os.mkdir(args["checkpoint_dir"])
            save_ckpt = os.path.join(args["checkpoint_dir"], checkpoint_name)

        LOGGER.info("Training started.")
    

        if args["criterion"] == "CE":
            LOGGER.info(f"Cross entropy is used.")
            criterion = torch.nn.CrossEntropyLoss()
        elif args["criterion"] == "BCE":
            LOGGER.info(f"Binary cross entropy is used.")
            checkpoint_name+= "_" + "BCE-loss"
            criterion = binary_cross_entropy(pos_weight = args["train_batch_size"]-1, device = device)
        else:
            LOGGER.warn(f'Unknown \'{args["criterion"]}\' loss function. Default loss function is used.')
            criterion = None
        
        framework.train(model,
                        learning_rate=args["lr"],
                        batch_size=args["train_batch_size"],
                        iter_size=args["iter_size"],
                        num_epoch=args["num_epoch"],
                        save_ckpt=save_ckpt,
                        fp16=args["fp16"],
                        criterion=criterion,
                        eval_freq=args["eval_freq"])
        LOGGER.info("Training completed.")
        
    LOGGER.info("Evaluation started")
    framework.validate(model,val_dataloader)

    LOGGER.info("Validation completed.")


if __name__ == "__main__":
    setup_logging(logpath="logs/")
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        if not os.path.exists(args.config):
            LOGGER.error("Config file does not found.")
            sys.exit(1)
        with open(args.config) as file_:
            jsons = json.load(file_)

        args.__dict__.update(jsons)

    train(var(args))
