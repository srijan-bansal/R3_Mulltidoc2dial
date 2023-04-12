from transformers import AutoConfig, AutoTokenizer, AutoModel
from .framework import RerankerFramework
from .reranker_datasets import BaselineRerankerQueryBuilder, SingleQuery
from .model import BaselineReranker
import torch

def get_reranker_model():
    cache_dir = 'cache/'
    config = AutoConfig.from_pretrained("roberta-base", cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", cache_dir=cache_dir, use_fast=False)
    encoder = AutoModel.from_pretrained("roberta-base", cache_dir=cache_dir)

    max_length = 300
    query_builder = BaselineRerankerQueryBuilder(tokenizer, max_length)
    model_config = {
        "reranker_model_type": "baseline",
        "encoder": "roberta-base",
        "encoder_config": config,
        "max_length": max_length
    }
    device = torch.device("cuda:0")
    framework = RerankerFramework(device, model_config)
    model = BaselineReranker(config, encoder)
    model = torch.load('checkpoints/reranker_1.ckpt')
    model = model.to(device)
    model = model.module
    return model, query_builder, framework

def rerank(query, pids, model, query_builder, framework):
    query_batch = SingleQuery(query, pids, query_builder)
    pids = framework.rerank(model, query_batch)
    return pids