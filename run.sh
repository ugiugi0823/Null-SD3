#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python run_sweep.py \
--learning_rate 0.1 \
--optimizer "adam" \