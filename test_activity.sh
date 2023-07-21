#!/usr/bin/env bash
python test.py --without_wandb \
--dataset_name ActivityNet1.2 \
--dataset AntSampleDataset \
--max_seqlen 60 \
--num_class 100 \
--proposal_method multiple_threshold_hamnet \
--path_dataset path/to/activity \
--action_threshold 0.5 \
--background_threshold 0.5 \
--similarity_threshold 0.8 \
--top_k_rat 10 \
--use_model DELU_ACT \
--AWM DDG_Net \
--model_name ActivityNet-DDG_Net \
--pretrained_ckpt ./download_ckpt/best_ActivityNet.pkl
