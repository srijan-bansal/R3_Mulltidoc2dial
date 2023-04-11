import os
import json
import argparse
from tqdm.auto import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True, help='path to the directory containing the data. We assume the presence of mdd_dpr in this directory')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    passage_path = os.path.join(args.base_dir, 'mdd_dpr', 'dpr.psg.multidoc2dial_all.structure.json')
    with open(passage_path, 'r') as fpassage:
        passages = json.load(fpassage)

    train_data_path = os.path.join(args.base_dir, 'mdd_dpr', 'dpr.multidoc2dial_all.structure.train.json')
    with open(train_data_path, 'r') as ftrain:
        train_data = json.load(ftrain)

    os.makedirs(os.path.join(args.base_dir, 'beir_format'), exist_ok=True)

    corpus_path = os.path.join(args.base_dir, 'beir_format', 'corpus.jsonl')
    with open(corpus_path, 'w') as fcorpus:
        for passage in tqdm(passages):
            id = passage['id']
            passage['_id'] = str(id)
            del passage['id']
            fcorpus.write(json.dumps(passage) + "\n")

    queries_path = os.path.join(args.base_dir, 'beir_format', 'queries.jsonl')
    with open(queries_path, 'w') as fqueries:
        for dial in tqdm(train_data):
            query = {}
            query['_id'] = dial['qid']
            query['text'] = dial['question']

            fqueries.write(json.dumps(query) + '\n')


    qrels_dir = os.path.join(args.base_dir, 'beir_format', 'qrels')
    os.makedirs(qrels_dir, exist_ok=True)
    qrels_path = os.path.join(qrels_dir, 'train.tsv')

    with open(qrels_path, 'w') as fqrels:

        fqrels.write("\t".join(["query_id", "doc_id", "label"]) + "\n")

        for dial in tqdm(train_data):
            query_id = dial['qid']

            for pos in dial['positive_ctxs']:
                psg_id = pos['psg_id']
                psg_id = str(psg_id)
                fqrels.write("\t".join([query_id, psg_id, "1"]) + "\n")

            for pos in dial['negative_ctxs']:
                psg_id = pos['psg_id']
                psg_id = str(psg_id)
                fqrels.write("\t".join([query_id, psg_id, "0"]) + "\n")

            for pos in dial['hard_negative_ctxs']:
                psg_id = pos['psg_id']
                psg_id = str(psg_id)
                fqrels.write("\t".join([query_id, psg_id, "0"]) + "\n")

if __name__ == "__main__":
    main()