#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--seed 0 \
--alpha_edl 1.3 \
--alpha_uct_guide 0.4 \
--amplitude 0.7 \
--alpha2 0.4 \
--interval 50 \
--max_seqlen 320 \
--lr 0.00005 \
--k 7 \
--dataset_name Thumos14reduced \
--path_dataset path/to/thumos \
--num_class 20 \
--use_model DELU \
--max_iter 5000 \
--dataset SampleDataset \
--weight_decay 0.001 \
--project TEST \
--action_threshold 0.5 \
--background_threshold 0.5 \
--top_k_rat 10 \
--similarity_threshold 0.8 \
--AWM DDG_Net \
--model_name Thumos14-DDG_Net \
--alpha6 1 \
--temperature 0.5 \
--weight 2 \
--alpha5 3.2
# We provide different "AWM" to compare, please refer to model.py.