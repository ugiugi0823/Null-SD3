#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python run_sweep.py \
--learning_rate 0.0001 \
--optimizer "adam" \