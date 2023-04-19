import os
import argparse
from typing import Type, List, Dict, Union, Tuple
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder

from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from models import BEIRDPR

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="data/beir_format/", help='path to the directory containing the data. We assume the presence of mdd_dpr in this directory')
    parser.add_argument('--dump_dir', default='data/dpr_negatives_beir_format/', help='path to the directory containing the data. We assume the presence of mdd_dpr in this directory')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for inference')
    parser.add_argument('--k_values', type=int, default=20, help='number of retrieved documents per example')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    print(args)

    query_encoder = DPRQuestionEncoder.from_pretrained(
                "sivasankalpp/dpr-multidoc2dial-structure-question-encoder")
    query_tokenizer = AutoTokenizer.from_pretrained(
        "sivasankalpp/dpr-multidoc2dial-structure-question-encoder")

    doc_encoder = DPRContextEncoder.from_pretrained(
        "sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")
    doc_tokenizer = AutoTokenizer.from_pretrained(
        "sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")

    beir_model = BEIRDPR(query_encoder, doc_encoder,
                            query_tokenizer, doc_tokenizer)
    model = DRES(beir_model, batch_size=128)

    corpus, queries, qrels = GenericDataLoader(
        args.base_dir).load(split="train")
        
    retriever = EvaluateRetrieval(model, score_function="dot", k_values=[20])
    results = retriever.retrieve(corpus, queries)

    os.makedirs(args.dump_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dump_dir,'qrels'), exist_ok=True)

    with open(os.path.join(args.base_dir, 'corpus.jsonl'), 'r') as fi:
        with open(os.path.join(args.dump_dir, 'corpus.jsonl'), 'w') as fo:
            for line in fi:
                fo.write(line)

    with open(os.path.join(args.base_dir,"queries.jsonl"), 'r') as fi:
        with open(os.path.join(args.dump_dir, "queries.jsonl"), 'w') as fo:
            for line in fi:
                fo.write(line)

    with open(os.path.join(args.dump_dir,"qrels", "train.tsv"), 'w') as fo:
        fo.write('\t'.join(['query_id', 'doc_id', 'label']) + "\n")
        for query_id in results.keys():
            for doc_id, rel in qrels[query_id].items():
                if rel == 1:
                    fo.write('\t'.join([query_id, doc_id, str(rel)]) + '\n')
            for doc_id, score in sorted(results[query_id].items(), key=lambda x: x[1], reverse=True):
                if qrels[query_id].get(doc_id, 0) == 0: # write only irrelevant results as negatives
                    fo.write('\t'.join([query_id, doc_id, str(rel)]) + '\n')

if __name__ == "__main__":
    main()