#!/bin/bash


NUM_GPU=1

PORT_ID=$(expr $RANDOM + 1000)

BASE_MODEL=stage1_model_path
MODEL_NAME_OR_PATH=output_model_path
export OMP_NUM_THREADS=8

for lamb in 0.2
do
for temp in 1.0
do
for gamma in 1.0
do
for lr in 1e-4
do
for layer in 0
do
bz=256
MODEL_NAME=output_model_name_or_path
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train_unsup.py \
    --model_name_or_path $BASE_MODEL \
    --train_file data/unsup_toy_sents.txt \
    --output_dir $MODEL_NAME \
    --num_train_epochs 10 \
    --per_device_train_batch_size $bz \
    --learning_rate $lr \
    --max_seq_length 128 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --eval_steps 2000 \
    --save_steps 2000 \
    --pooler_type avg \
    --mlp_only_train \
    --overwrite_output_dir \
    --do_train \
    --lambdas $lamb \
    --lambdas2 $layer \
    --gamma $gamma \
    --temp $temp \
    --do_eval \
    --teacher_emb_path data/toy_embs.pt \
    --teacher_emb_dim 1024 \
    --student_emb_dim 312 \
    "$@"
echo $MODEL_NAME
CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_name_or_path $MODEL_NAME --pooler avg --task_set sts --mode test
done
done
done
done
done
