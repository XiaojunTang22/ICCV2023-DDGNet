#!/usr/bin/env bash
python test.py --without_wandb \
--path_dataset path/to/thumos \
--action_threshold 0.5 \
--background_threshold 0.5 \
--similarity_threshold 0.8 \
--top_k_rat 10 \
--use_model DELU \
--AWM DDG_Net \
--model_name Thumos14-DDG_Net \
--pretrained_ckpt ./download_ckpt/best_Thumos14.pkl
