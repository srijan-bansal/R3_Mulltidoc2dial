import os
import sys
import pytrec_eval
from collections import defaultdict

from beir import util
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from typing import Type, List, Dict, Union, Tuple
from beir.retrieval.custom_metrics import mrr

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help='path containing the queries corpus and qrels')
    parser.add_argument("--retrieved_results_path", required=True, help='path containing the retrieved result TSV')
    parser.add_argument("--split", default='validation', type=str)
    parser.add_argument("--k_values", default='10,100', type=str)

    args = parser.parse_args()
    k_values = [int(i) for i in args.k_values.strip().split(',')]
    args.k_values = k_values
    return args

def main():
    args = get_args()
    retrieved_results_path = args.retrieved_results_path
    data_dir = args.data_dir
    split = args.split
    k_values = args.k_values

    corpus, queries, qrels = GenericDataLoader(
        data_dir).load(split=split)

    print("lengths of queries, corpus, qrels:", len(queries), len(corpus), len(qrels))

    results = defaultdict(dict)
    with open(retrieved_results_path) as fi:
        fi.readline() # header
        for i in fi:
            qid, pid, score = i.strip().split('\t')
            score = float(score)
            results[qid][pid] = score

    assert len(results) == len(queries)
    for query_id in results.keys():
        assert query_id in qrels
        for doc_id,score in results[query_id].items():
            assert doc_id in corpus

    ndcg, _map, recall, precision, _mrr = evaluate(qrels, results, k_values)
    print(ndcg, _map, recall, precision, _mrr)

if __name__ == "__main__":
    main()