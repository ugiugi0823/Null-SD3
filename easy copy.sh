#!/bin/bash

for lr in $(seq 0.1 0.1 0.9)
do
    CUDA_VISIBLE_DEVICES=1 python run.py \
    --learning_rate $lr \
    --optimizer "adam"
done
