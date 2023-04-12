#!/bin/sh

HF_HOME=cache
DATA_DIR=data

if [ -d "$HF_HOME" ]; then
    echo "$HF_HOME exists."
else     
    mkdir $HF_HOME
fi


if [ -d $DATA_DIR ]; then 
    echo "$DATA_DIR exists."
else
    mkdir $DATA_DIR && \
    cd $DATA_DIR && \
    wget http://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip && \
    wget http://doc2dial.github.io/multidoc2dial/file/multidoc2dial_domain.zip && \
    unzip multidoc2dial.zip && \
    unzip multidoc2dial_domain.zip && \
    rm *.zip && \
    wget https://huggingface.co/facebook/rag-token-nq/raw/main/question_encoder_tokenizer/tokenizer_config.json && \
    wget https://huggingface.co/facebook/rag-token-nq/raw/main/question_encoder_tokenizer/vocab.txt
    cd ..
fi

task=grounding

python multidoc2dial/scripts/data_preprocessor.py \
--dataset_name multidoc2dial/scripts/hf_datasets/doc2dial/doc2dial_pub.py \
--dataset_config_name multidoc2dial \
--output_dir $DATA_DIR/mdd_all \
--kb_dir $DATA_DIR/mdd_kb \
--segmentation structure \
--task $task 


task=generation
python multidoc2dial/scripts/data_preprocessor.py \
--dataset_name multidoc2dial/scripts/hf_datasets/doc2dial/doc2dial_pub.py \
--dataset_config_name multidoc2dial \
--output_dir $DATA_DIR/mdd_all \
--kb_dir $DATA_DIR/mdd_kb \
--segmentation structure \
--task $task 



python multidoc2dial/scripts/data_preprocessor.py \
--dataset_name multidoc2dial/scripts/hf_datasets/doc2dial/doc2dial_pub.py \
--dataset_config_name multidoc2dial \
--output_dir $DATA_DIR/mdd_dpr \
--segmentation structure \
--split train \
--dpr

python multidoc2dial/scripts/data_preprocessor.py \
--dataset_name multidoc2dial/scripts/hf_datasets/doc2dial/doc2dial_pub.py \
--dataset_config_name multidoc2dial \
--output_dir $DATA_DIR/mdd_dpr \
--segmentation structure \
--split validation \
--dpr

python multidoc2dial/scripts/data_preprocessor.py \
--dataset_name multidoc2dial/scripts/hf_datasets/doc2dial/doc2dial_pub.py \
--dataset_config_name multidoc2dial \
--output_dir $DATA_DIR/mdd_dpr \
--segmentation structure \
--split test \
--dpr