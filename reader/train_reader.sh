train_data_path=$1
eval_data_path=$2
EXPT_NAME=$3
MODEL_NAME=$4
OUTDIR=$5

python reader/train_reader.py \
        --train_data $train_data_path \
        --eval_data $eval_data_path \
        --model_size base \
        --per_gpu_batch_size 2 \
        --accumulation_steps 8 \
        --n_context 10 \
        --name $EXPT_NAME \
        --checkpoint_dir $OUTDIR \
        --total_steps 50000 \
        --ispretrained \
        --model_path $MODEL_NAME \
        --train \
        --text_maxlength 512 \
        --question_maxlength 100 \
        --eval_freq 5000 \
        --save_freq 2500