# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
import sacrebleu
from tqdm import tqdm
from datasets import load_metric
import string
import re
from collections import Counter

metric_meteor = load_metric("meteor")
metric_rouge = load_metric("rouge")

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def get_match_scores(hypos, answers):
    f1 = em = total = 0
    for prediction, ground_truths in zip(hypos, answers):
        total += 1
        em += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    
    em = 100.0 * em / total
    f1 = 100.0 * f1 / total
    return em, f1


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, best_dev_bleu,checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')
    torch.manual_seed(opt.seed)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=1,
        collate_fn=collator
    )
    targets, answers, best_answers = [], [], []
    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    start = step % len(train_dataloader)
    while step < opt.total_steps:
        epoch += 1
        ''' this is to resume training '''
        for i, batch in enumerate(tqdm(train_dataloader)):
            if i < start : continue
            start = 0
            step += 1
            (idx, labels, _, context_ids, context_mask) = batch
            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]
            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()
            if step % opt.eval_freq == 0:
                dev_em, dev_bleu, answers, golds = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM | {dev_bleu:.2f}BLEU | "
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)    
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation EM", dev_em, step)
                        tb_logger.add_scalar("Evaluation BLEU", dev_bleu, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em, max(dev_bleu, best_dev_bleu),
                                  opt, checkpoint_path, 'best_dev_em')
                    if dev_bleu > best_dev_bleu:
                        best_dev_bleu = dev_bleu
                        src.util.save(model, optimizer, scheduler, step, best_dev_em, best_dev_bleu,
                                  opt, checkpoint_path, 'best_dev_bleu')
                        best_answers = answers
                        targets = golds
                    curr_loss = 0.

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em, best_dev_bleu,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break
    return best_answers, targets

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=1,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    em_score = 0.0
    bleu_score = 0.0
    model = model.module if hasattr(model, "module") else model
    answers = []
    golds = []
    preds = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            (idx, _, _, context_ids, context_mask) = batch
            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=opt.output_maxlength,
                num_beams=opt.num_beams,
                num_return_sequences=opt.num_return_sequences
            )
            batch_answers = []
            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                batch_answers.append(ans)
            answers.extend(batch_answers)
            num_sequence = 1 if opt.num_return_sequences is None else opt.num_return_sequences
            if len(dataset.get_example(idx[0])['answers']) > 0 and len(dataset.get_example(idx[0])['answers'][0]) > 0:
                batch_preds = []
                for k in range(len(outputs)//num_sequence):
                    gold = dataset.get_example(idx[k])['answers']
                    pred = batch_answers[k * num_sequence] # in case of multiple sequence take the output of just first prediction
                    score = src.evaluation.ems(pred, gold)
                    batch_preds.append(pred)
                    golds.append(gold[0])
                    total += 1
                    exactmatch.append(score)
                preds.extend(batch_preds)
    if len(golds) > 0:
        bleu_score = sacrebleu.corpus_bleu(preds, [golds]).score
        em_score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return em_score, bleu_score, answers, golds


def write_to_file(answers, opt):
    with open(opt.output_file, 'w+') as fp:
        for ans in answers:
            fp.write(ans + '\n')
            fp.flush()

def calculate_metrics(answers, golds, opt):
    num_sequence = 1 if opt.num_return_sequences is None else opt.num_return_sequences
    preds = []
    total = 0
    exactmatch = []
    for k in range(len(golds)):
        gold = golds[k]
        pred = answers[k*num_sequence] # in case of multiple sequence take the output of just first prediction
        score = src.evaluation.ems(pred, [gold])
        preds.append(pred)
        total += 1
        exactmatch.append(score)
    bleu_score = round(sacrebleu.corpus_bleu(preds, [golds]).score, 4)
    em_score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    refs = [[reference] for reference in golds]
    _, f1_score = get_match_scores(preds, refs)
    rouge_score = round(metric_rouge.compute(predictions=preds, references=golds)["rougeL"].mid.fmeasure * 100, 4)
    meteor_score = round(metric_meteor.compute(predictions=preds, references=golds)["meteor"] * 100, 4)
    ### we are using the EM from FiD implementation and BLEU from sacrebleu 
    ### these are different from the one in actual evaluation
    logger.info(f"EM : {em_score*100:.2f} | BLEU : {bleu_score:.2f} | F1 : {f1_score:.2f} | RougeL : {rouge_score:.2f} | Meteor : {meteor_score:.2f}")

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    options.add_eval_options()
    opt = options.parse()
    print (opt)
    opt.is_main = True
    opt.device = torch.device("cuda")
    torch.manual_seed(opt.seed)
    if opt.output_file is None:
        opt.output_file = f'output-{opt.name}.txt'
    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    options.print_options(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )
    
    model_name = 't5-' + opt.model_size
    model_class = src.model.FiDT5
    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength, question_maxlength=opt.question_maxlength)
    if opt.train :
        train_examples = src.data.load_data(opt.train_data)
        train_dataset = src.data.Dataset(train_examples, opt.n_context)

    eval_examples = src.data.load_data(opt.eval_data)
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)
    if not checkpoint_exists and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.device)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em, best_dev_bleu = 0, 0.0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        if opt.evaluate :
            load_path = checkpoint_path / 'checkpoint' / 'best_dev_bleu'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em, best_dev_bleu = \
            src.util.load(model_class, load_path, opt, reset_params=False, load_optim=True)
        logger.info(f"Model loaded from {load_path}")
        ## hardcoding for continuing my_experiment_q_100_n_10_l_512_wh experiment
    elif opt.ispretrained:
        model, optimizer, scheduler = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True, load_optim=False)
        logger.info(f"Model loaded from {opt.model_path}")
        step, best_dev_em, best_dev_bleu = 0, 0.0, 0.0
    else:
        # model, optimizer, scheduler, opt_checkpoint, step, best_dev_em, best_dev_bleu = \
        #     src.util.load(model_class, opt.model_path, opt, reset_params=True, load_optim=True)
        # logger.info(f"Model loaded from {opt.model_path}")
        model, optimizer, scheduler = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True, load_optim=False)
        logger.info(f"Model loaded from {opt.model_path}")
        step, best_dev_em, best_dev_bleu = 0, 0.0, 0.0

    model.set_checkpoint(opt.use_checkpoint)

    logger.info("Start training")
    
    if opt.train:
        answers, targets = train(
            model,
            optimizer,
            scheduler,
            step,
            train_dataset,
            eval_dataset,
            opt,
            collator,
            best_dev_em,
            best_dev_bleu,
            checkpoint_path
        )
        write_to_file(answers, opt)
        calculate_metrics(answers, targets, opt)
        
    if opt.evaluate:
        dev_em, dev_bleu, answers, golds = evaluate(model, eval_dataset, tokenizer, collator, opt)
        write_to_file(answers, opt)
        if len(golds) > 0:
            calculate_metrics(answers, golds, opt)