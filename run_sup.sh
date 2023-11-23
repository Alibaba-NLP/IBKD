#!/bin/bash

NUM_GPU=1

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

for lr in 1e-6
do
for temp in 0.05
do
for bz in 128
do
for lamb in 1.0
do
MODEL_NAME=output_path
BASE_MODEL=stduent_model_path

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train_sup.py \
    --model_name_or_path  $BASE_MODEL \
    --train_file  data/sup_toy_data.csv \
    --output_dir $MODEL_NAME \
    --num_train_epochs 3 \
    --per_device_train_batch_size $bz \
    --learning_rate $lr \
    --max_seq_length 128 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type avg \
    --overwrite_output_dir \
    --temp $temp \
    --do_train \
    --do_eval \
    --fp16 \
    --lambdas $lamb \
    "$@"
echo $MODEL_NAME
CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_name_or_path $MODEL_NAME --pooler avg --task_set sts --mode test

done
done
done
done
