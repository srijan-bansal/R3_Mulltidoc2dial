import os
import sys
import pytrec_eval
from beir import util
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from typing import Type, List, Dict, Union, Tuple
from models import Splade, BEIRSpladeModel, BEIRDPR
from beir.retrieval.custom_metrics import mrr
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder

import argparse

import pdb

def evaluate(qrels: Dict[str, Dict[str, int]],
             results: Dict[str, Dict[str, float]],
             k_values: List[int]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    _mrr = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)

    _mrr = mrr(qrels, results, k_values)

    for eval in [ndcg, _map, recall, precision, _mrr]:
        for k in eval.keys():
            print("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision, _mrr

def recall_at_k(qrels, results, k=10):
    not_retreived = []
    recall = 0
    counts = 0
    for query_id in qrels.keys():
        results_at_k = [doc_id for (doc_id, score) in sorted(results[query_id].items(), key=lambda x: x[1], reverse=True)][:k]
        results_at_k = set(results_at_k)
        for doc_id in qrels[query_id].keys():
            if doc_id in results_at_k:
                recall += 1
            else:
                not_retreived.append((query_id, doc_id))
            counts += 1

    return recall/counts, not_retreived

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=['dpr-ft', 'dpr-nq', 'tas-b', 'distilsplade', 'splade'])
    parser.add_argument("--model_path", default=None, help='Checkpoint directory. If not mentioned, default models are loaded')
    parser.add_argument("--data_dir", default='../data/beir_format')
    parser.add_argument("--dump_dir", default='../data/retrieved_results')
    parser.add_argument("--split", default='validation', type=str)
    parser.add_argument("--k_values", default='10,100', type=str)

    args = parser.parse_args()
    k_values = [int(i) for i in args.k_values.strip().split(',')]
    args.k_values = k_values
    return args

def main():
    args = get_args()
    model_name = args.model_name
    model_path = args.model_path
    data_dir = args.data_dir
    dump_dir = args.dump_dir
    split = args.split

    corpus, queries, qrels = GenericDataLoader(
        data_dir).load(split=split)

    print("lengths of queries, corpus, qrels:", len(queries), len(corpus), len(qrels))

    if "dpr-ft" in model_name:
        if model_path is None:
            query_encoder = DPRQuestionEncoder.from_pretrained(
                "sivasankalpp/dpr-multidoc2dial-structure-question-encoder")
            query_tokenizer = AutoTokenizer.from_pretrained(
                "sivasankalpp/dpr-multidoc2dial-structure-question-encoder")

            doc_encoder = DPRContextEncoder.from_pretrained(
                "sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")
            doc_tokenizer = AutoTokenizer.from_pretrained(
                "sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")
        else:
            query_encoder = DPRQuestionEncoder.from_pretrained(
                model_path)
            query_tokenizer = AutoTokenizer.from_pretrained(
                model_path)

            doc_encoder = DPRContextEncoder.from_pretrained(
                model_path)
            doc_tokenizer = AutoTokenizer.from_pretrained(
                model_path)

        beir_model = BEIRDPR(query_encoder, doc_encoder,
                            query_tokenizer, doc_tokenizer)
        model = DRES(beir_model, batch_size=128)

    elif "dpr-nq" in model_name:
        if model_path is None:
            model = DRES(models.SentenceBERT(
                "sentence-transformers/facebook-dpr-question_encoder-multiset-base"), batch_size=128)
        else:
            model = DRES(models.SentenceBERT(
                model_path), batch_size=128)

    elif "tas-b" in model_name:
        if model_path is None:
            model = DRES(models.SentenceBERT(
                "sentence-transformers/msmarco-distilbert-base-tas-b"), batch_size=128)
        else:
            model = DRES(models.SentenceBERT(
                model_path), batch_size=128)

    elif "distilsplade" in model_name:
        if model_path is None:
            model_type_or_dir = "splade_weights/distilsplade_max"
            model = Splade(model_type_or_dir)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
            beir_splade = BEIRSpladeModel(model, tokenizer)
            model = DRES(beir_splade)
        else:
            model_type_or_dir = model_path
            model = Splade(model_type_or_dir)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
            beir_splade = BEIRSpladeModel(model, tokenizer)
            model = DRES(beir_splade)

    elif "splade" in model_name:
        if model_path is None:
            model_type_or_dir = "splade_weights/splade_max"
            model = Splade(model_type_or_dir)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
            beir_splade = BEIRSpladeModel(model, tokenizer)
            model = DRES(beir_splade)
        else:
            model_type_or_dir = model_path
            model = Splade(model_type_or_dir)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
            beir_splade = BEIRSpladeModel(model, tokenizer)
            model = DRES(beir_splade)

    retriever = EvaluateRetrieval(model, score_function="dot", k_values=[i-1 for i in args.k_values]) # retriever retrieves topk +1 for some reason
    results = retriever.retrieve(corpus, queries)

    assert len(results) == len(queries)
    for query_id in results.keys():
        assert query_id in qrels
        for doc_id,score in results[query_id].items():
            assert doc_id in corpus

    os.makedirs(dump_dir, exist_ok=True)
    with open(os.path.join(dump_dir, f'{model_name}-results.tsv'), 'w') as fo:
        fo.write('\t'.join(["qid","pid","score"])+'\n')
        for query_id in results.keys():
            for doc_id, score in sorted(results[query_id].items(), key=lambda x: x[1], reverse=True):
                fo.write('\t'.join(list(map(str, [query_id, doc_id, score]))) + '\n')

if __name__ == "__main__":
    main()