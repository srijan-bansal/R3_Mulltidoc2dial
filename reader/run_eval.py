import sys
import torch
import transformers
import numpy as np
from types import SimpleNamespace

import src.util
import src.model


def get_options():
    opt = SimpleNamespace()
    opt.n_context = 10
    opt.is_main = True
    opt.device = torch.device("cuda")
    opt.model_path = "/home/srijanb/Sumit/FiD/checkpoint/nq_experiment_phase_4/checkpoint/best_dev_bleu"
    opt.text_maxlength = 512
    opt.question_maxlength = 100
    opt.answer_maxlength = -1
    opt.output_maxlength = 50
    opt.seed = 0
    return opt


def get_reader_model():
    opt = get_options()
    torch.manual_seed(opt.seed)
    model_name = 't5-base'
    model_class = src.model.FiDT5
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    model = src.util.load(model_class, opt.model_path , opt, reset_params=True, load_optim=False, create_optim=False)
    return model, tokenizer, opt

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

def get_response(opt, model, tokenizer, example):
    n_context = opt.n_context
    question_prefix = 'question:'
    title_prefix = 'title:'
    passage_prefix = 'context:'
    question = question_prefix + " " + example['question']
    
    if 'ctxs' in example and n_context is not None:
        f = title_prefix + " {} " + passage_prefix + " {}"
        contexts = example['ctxs'][:n_context]
        passages = [f.format(c['text'].split('//')[0].strip(), c['text'].split('//')[1].strip()) for c in contexts]
    
    
    text_maxlength = opt.text_maxlength
    question_maxlength = opt.question_maxlength
    
    def append_question(question, passages):
        question = ' '.join(question.split(' ')[:question_maxlength])
        return [question + " " + t for t in passages]
    
    text_passages = [append_question(question, passages)]
    passage_ids, passage_masks = encode_passages(text_passages, tokenizer, text_maxlength)
    outputs = model.generate(
                input_ids=passage_ids.cuda(),
                attention_mask=passage_masks.cuda(),
                max_length=opt.output_maxlength,
                num_beams=4
             )
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return ans


example = {
    'question' : 'How is your day?',
    'ctxs' : [ {'text' : 'Random shit // Random shit'}, 
               {'text' : 'Random shit // Random shit'}, 
               {'text' : 'Random shit // Random shit'}, 
               {'text' : 'Random shit // Random shit'}, 
               {'text' : 'Random shit // Random shit'}, 
               {'text' : 'Random shit // Random shit'}, 
               {'text' : 'Random shit // Random shit'}, 
               {'text' : 'Random shit // Random shit'}, 
               {'text' : 'Random shit // Random shit'}, 
               {'text' : 'Random shit // Random shit'}
             ]
}
model, tokenizer, opt = get_reader_model()
ans = get_response(opt, model, tokenizer, example)