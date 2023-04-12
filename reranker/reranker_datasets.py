import logging
import os
import sys
import itertools
import json
import random
import pickle

import torch
import torch.utils.data as data
import pandas as pd
from collections import defaultdict

LOGGER = logging.getLogger(__name__)



class BaselineRerankerQueryBuilder(object):

    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.start_context_token_id = self.tokenizer.convert_tokens_to_ids("muw00")
        self.start_title_token_id = self.tokenizer.convert_tokens_to_ids("muw01")

    def tokenize_and_convert_to_ids(self, text):
        tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def tokenize_question_and_convert_to_ids(self, text):
        tokens = self.tokenizer.tokenize(text)[:128]
        return self.tokenizer.convert_tokens_to_ids(tokens)

    @property
    def num_special_tokens_to_add(self):
        return self.tokenizer.num_special_tokens_to_add(pair=True)

    def __call__(self, question, passages, numerized=False):
        if not numerized:
            question = self.tokenize_question_and_convert_to_ids(question)
            passages = [(self.tokenize_and_convert_to_ids(item[0]), self.tokenize_and_convert_to_ids(item[1])) for item in passages]

        cls = self.tokenizer.convert_tokens_to_ids([self.tokenizer.bos_token])
        sep = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])
        eos = self.tokenizer.convert_tokens_to_ids([self.tokenizer.eos_token])

        input_ids_list = []

        for passage in passages:
            input_ids = cls + question + sep + sep
            input_ids.extend([self.start_title_token_id] + passage[0])
            input_ids.extend([self.start_context_token_id] + passage[1] + eos)

            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length-1] + eos

            input_ids_list.append(input_ids)
    
        seq_len = max(map(len, input_ids_list))

        input_ids_tensor = torch.ones(len(input_ids_list), seq_len).long()
        attention_mask_tensor = torch.zeros(len(input_ids_list), seq_len).long()

        for batch_index, input_ids in enumerate(input_ids_list):

            for sequence_index, value in enumerate(input_ids):
                input_ids_tensor[batch_index][sequence_index] = value
                attention_mask_tensor[batch_index][sequence_index] = 1.

        features = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor
        }

        return features

def text2line(text):
    return text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()

def split_text_section(spans, title):
    def get_text(buff, title, span):
        text = " ".join(buff).replace("\n", " ")
        parent_titles = [title.replace("/", "-").rsplit("#")[0]]
        if len(span["parent_titles"]) > 1:
            parent_titles = [ele['text'].replace("/", "-").rsplit("#")[0] for ele in span["parent_titles"]]
        text = " / ".join(parent_titles) + " // " + text
        return text2line(text)

    buff = []
    pre_sec, pre_title, pre_span = None, None, None
    passages = []
    subtitles = []
        
    for span_id in spans:
        span = spans[span_id]
        parent_titles = title
        if len(span["parent_titles"]) > 1:                        
            parent_titles = [ele['text'].replace("/", "-").rsplit("#")[0] for ele in span["parent_titles"]]
            parent_titles = " / ".join(parent_titles)
        if pre_sec == span["id_sec"] or pre_title == span["title"].strip():
            buff.append(span["text_sp"])
        elif buff:
            text = get_text(buff, title, pre_span)
            passages.append(text)
            subtitles.append(parent_titles)
            buff = [span["text_sp"]]
        else:
            buff.append(span["text_sp"])
        pre_sec = span["id_sec"]
        pre_span = span
        pre_title = span["title"].strip()
    if buff:
        text = get_text(buff, title, span)
        passages.append(text)
        subtitles.append(parent_titles)
    return passages, subtitles

def get_predicted_passages(pred_file, qids):
    df = pd.read_table(pred_file)
    pred_pids = list(df['pid'])
    pred_qids = list(df['qid'])
    assert (len(qids) == len(set(pred_qids)))
    qid_map = defaultdict(list)
    for qid, pid in zip(pred_qids, pred_pids):
        qid_map[qid].append(pid)
    return [qid_map[qid] for qid in qids]
            


class EfficientQARerankerDatasetForBaselineReranker_TRAIN(data.Dataset):
    def __init__(self, all_doc_filename, query_filename, query_id_filename, pred_filename, gold_pid_filename, tokenizer, query_builder, batch_size, shuffle_predicted_indices=False):
        self.doc_data = json.load(open(all_doc_filename))
        self.queries = [line.strip() for line in open(query_filename, "r").readlines()]
        self.qids = [line.strip() for line in open(query_id_filename, "r").readlines()]
        self.psg_predictions = get_predicted_passages(pred_filename, self.qids)
        self.pids = [line.strip() for line in open(gold_pid_filename, "r").readlines()]
        self.tokenizer = tokenizer
        self.shuffle_predicted_indices = shuffle_predicted_indices
        self.query_builder = query_builder
        self.batch_size = batch_size
        doc_passages = {}
        all_passages = []
        start_idx = 0
        for domain in self.doc_data['doc_data']:
            for doc_id in self.doc_data['doc_data'][domain].keys():
                ex = self.doc_data['doc_data'][domain][doc_id]
                passages, subtitles = split_text_section(ex["spans"], ex["title"])
                all_passages.extend(passages)
                doc_passages[ex["doc_id"]] = (start_idx, len(passages))
                start_idx += len(passages)
                
        self.passage_map = {}
        for title in doc_passages:
            psg_start_ix = doc_passages[title][0]
            n_psgs = doc_passages[title][1]
            for i in range(n_psgs):
                self.passage_map[psg_start_ix + i] = {"text": all_passages[psg_start_ix + i], "title": title}


    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]

        support_list = []
        ground_truth_doc = self.pids[idx]
        shuffle_indices = self.psg_predictions[idx]
        if ground_truth_doc in shuffle_indices:
            shuffle_indices.remove(ground_truth_doc)

        if self.shuffle_predicted_indices:
            random.shuffle(shuffle_indices)
        shuffle_indices = [ground_truth_doc] + shuffle_indices
        for i in shuffle_indices:
            candidate = self._get_raw_doc(i)
            support_list.append(candidate)
            if len(support_list) >= self.batch_size:
                break

        if (self.batch_size != len(support_list)):
            LOGGER.warn(f"{self.batch_size} != {len(support_list)} (batch_size != len(support_list))")

        features = self.query_builder(query, support_list, numerized=False)
        #assert support_list[0] == ground_truth_doc
        features["labels"] = torch.tensor([0])
        features["qid"] = self.qids[idx]
        return features

    def _get_raw_doc(self, doc_id):
        doc_text = self.passage_map[int(doc_id)]['text']
        index = doc_text.find('//')
        title = doc_text[:index]
        context = doc_text[index+2:]
        title = title.strip()
        context = context.strip()
        return (title, context)

    def _get_tokenize_doc(self, idx):
        title, context = self._get_raw_doc(idx)
        title = self.query_builder.tokenize_and_convert_to_ids(title)
        context = self.query_builder.tokenize_and_convert_to_ids(context)
        return (title, context)

class EfficientQARerankerDatasetForBaselineReranker_VAL(data.Dataset):

    def __init__(self, all_doc_filename, query_filename, query_id_filename, pred_filename, query_builder, batch_size, gold_pid_filename=None):
        self.doc_data = json.load(open(all_doc_filename))
        self.queries = [line.strip() for line in open(query_filename, "r").readlines()]
        self.qids = [line.strip() for line in open(query_id_filename, "r").readlines()]
        self.psg_predictions = get_predicted_passages(pred_filename, self.qids)
        self.pids = None
        if gold_pid_filename is not None:
            self.pids = [line.strip() for line in open(gold_pid_filename, "r").readlines()]
        self.query_builder = query_builder
        self.batch_size = batch_size
        doc_passages = {}
        all_passages = []
        start_idx = 0
        for domain in self.doc_data['doc_data']:
            for doc_id in self.doc_data['doc_data'][domain].keys():
                ex = self.doc_data['doc_data'][domain][doc_id]
                passages, subtitles = split_text_section(ex["spans"], ex["title"])
                all_passages.extend(passages)
                doc_passages[ex["doc_id"]] = (start_idx, len(passages))
                start_idx += len(passages)
                
        self.passage_map = {}
        for title in doc_passages:
            psg_start_ix = doc_passages[title][0]
            n_psgs = doc_passages[title][1]
            for i in range(n_psgs):
                self.passage_map[psg_start_ix + i] = {"text": all_passages[psg_start_ix + i], "title": title}

    @property
    def passages_in_batch(self):
        return self.batch_size

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        psg_indices = self.psg_predictions[idx]
        passages = [self._get_raw_doc(idx) for idx in psg_indices[:self.batch_size]]

        batch = self.query_builder(query, passages, False)
        if self.pids is not None:
            batch["hits"] = [self.pids[idx]]
        batch["psg_predictions"] = psg_indices[:self.batch_size]
        batch["qid"] = self.qids[idx]
        return batch

    def _get_raw_doc(self, doc_id):
        doc_text = self.passage_map[int(doc_id)]['text']
        index = doc_text.find('//')
        title = doc_text[:index]
        context = doc_text[index+2:]
        title = title.strip()
        context = context.strip()
        return (title, context)
    
class SingleQuery():
    def __init__(self, all_doc_filename, query, pids, query_builder, batch_size=100):
        self.doc_data = json.load(open(all_doc_filename))
        self.query = query
        self.query_builder = query_builder
        self.batch_size = batch_size
        doc_passages = {}
        all_passages = []
        start_idx = 0
        for domain in self.doc_data['doc_data']:
            for doc_id in self.doc_data['doc_data'][domain].keys():
                ex = self.doc_data['doc_data'][domain][doc_id]
                passages, subtitles = split_text_section(ex["spans"], ex["title"])
                all_passages.extend(passages)
                doc_passages[ex["doc_id"]] = (start_idx, len(passages))
                start_idx += len(passages)
                
        self.passage_map = {}
        for title in doc_passages:
            psg_start_ix = doc_passages[title][0]
            n_psgs = doc_passages[title][1]
            for i in range(n_psgs):
                self.passage_map[psg_start_ix + i] = {"text": all_passages[psg_start_ix + i], "title": title}
        
        self.pids = pids

    def get_batch(self):
        query = self.query
        psg_indices = self.pids
        passages = [self._get_raw_doc(idx) for idx in psg_indices[:self.batch_size]]
        batch = self.query_builder(query, passages, False)
        batch["psg_predictions"] = psg_indices[:self.batch_size]
        return batch

    def _get_raw_doc(self, doc_id):
        doc_text = self.passage_map[int(doc_id)]['text']
        index = doc_text.find('//')
        title = doc_text[:index]
        context = doc_text[index+2:]
        title = title.strip()
        context = context.strip()
        return (title, context)