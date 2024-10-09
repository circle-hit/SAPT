#!/bin/bash
#SBATCH -J cl                           
#SBATCH -o cl-%j.out                       
#SBATCH -p compute 
#SBATCH -N 1                           
#SBATCH -t 5:00:00   
#SBATCH --mem 64G 
#SBATCH --gres=gpu:a100-sxm4-80gb:1        


export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)  

lr=0.001
topk=20


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
CUDA_VISIBLE_DEVICES=0 python  src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
   --data_dir CL_Benchmark \
   --task_config_dir configs/Ours_CL_configs/lookback_task181_outcome_extraction \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task181_outcome_extraction \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 1 \
   --learning_rate $lr \
   --max_steps  5000 \
   --run_name Ours_CL_round1 \
   --max_source_length 5 \
   --max_target_length 512 \
   --generation_max_length 512 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy epoch \
   --save_strategy epoch \
   --top_k $topk  \


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/Ours_CL_configs/lookback_task591_sciq_answer_generation \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task591_sciq_answer_generation \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 32 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate $lr \
#    --max_steps  5000 \
#    --deepspeed configs/ds_configs/stage2.config \
#    --run_name Ours_CL_round1 \
#    --max_source_length 5 \
#    --max_target_length 512 \
#    --generation_max_length 512 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy epoch \
#    --save_strategy epoch \
#    --top_k $topk  \


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/Ours_CL_configs/lookback_task1729_personachat_generate_next \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task1729_personachat_generate_next \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 32 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate $lr \
#    --max_steps  5000 \
#    --deepspeed configs/ds_configs/stage2.config \
#    --run_name Ours_CL_round1 \
#    --max_source_length 5 \
#    --max_target_length 512 \
#    --generation_max_length 512 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy epoch \
#    --save_strategy epoch \
#    --top_k $topk  \


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/Ours_CL_configs/lookback_task1510_evalution_relation_extraction \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task1510_evalution_relation_extraction \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 32 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate $lr \
#    --max_steps  5000 \
#    --deepspeed configs/ds_configs/stage2.config \
#    --run_name Ours_CL_round1 \
#    --max_source_length 5 \
#    --max_target_length 512 \
#    --generation_max_length 512 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy epoch \
#    --save_strategy epoch \
#    --top_k $topk  \


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/Ours_CL_configs/lookback_task748_glucose_reverse_cause_event_detection \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task748_glucose_reverse_cause_event_detection \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 32 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate $lr \
#    --max_steps  5000 \
#    --deepspeed configs/ds_configs/stage2.config \
#    --run_name Ours_CL_round1 \
#    --max_source_length 5 \
#    --max_target_length 512 \
#    --generation_max_length 512 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy epoch \
#    --save_strategy epoch \
#    --top_k $topk  \


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/Ours_CL_configs/lookback_task002_quoref_answer_generation \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task002_quoref_answer_generation \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 32 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate $lr \
#    --max_steps  5000 \
#    --deepspeed configs/ds_configs/stage2.config \
#    --run_name Ours_CL_round1 \
#    --max_source_length 5 \
#    --max_target_length 512 \
#    --generation_max_length 512 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy epoch \
#    --save_strategy epoch \
#    --top_k $topk  \


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/Ours_CL_configs/lookback_task1687_sentiment140_classification \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task1687_sentiment140_classification \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 32 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate $lr \
#    --max_steps  5000 \
#    --deepspeed configs/ds_configs/stage2.config \
#    --run_name Ours_CL_round1 \
#    --max_source_length 5 \
#    --max_target_length 512 \
#    --generation_max_length 512 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy epoch \
#    --save_strategy epoch \
#    --top_k $topk  \


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/Ours_CL_configs/lookback_task511_reddit_tifu_long_text_summarization \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task511_reddit_tifu_long_text_summarization \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 32 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate $lr \
#    --max_steps  5000 \
#    --deepspeed configs/ds_configs/stage2.config \
#    --run_name Ours_CL_round1 \
#    --max_source_length 5 \
#    --max_target_length 512 \
#    --generation_max_length 512 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy epoch \
#    --save_strategy epoch \
#    --top_k $topk  \


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/Ours_CL_configs/lookback_task875_emotion_classification \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task875_emotion_classification \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 32 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate $lr \
#    --max_steps  5000 \
#    --deepspeed configs/ds_configs/stage2.config \
#    --run_name Ours_CL_round1 \
#    --max_source_length 5 \
#    --max_target_length 512 \
#    --generation_max_length 512 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy epoch \
#    --save_strategy epoch \
#    --top_k $topk  \


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/Ours_CL_configs/lookback_task639_multi_woz_user_utterance_generation \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task639_multi_woz_user_utterance_generation \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 32 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate $lr \
#    --max_steps  5000 \
#    --deepspeed configs/ds_configs/stage2.config \
#    --run_name Ours_CL_round1 \
#    --max_source_length 5 \
#    --max_target_length 512 \
#    --generation_max_length 512 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy epoch \
#    --save_strategy epoch \
#    --top_k $topk  \


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/Ours_CL_configs/lookback_task1290_xsum_summarization \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task1290_xsum_summarization \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 32 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate $lr \
#    --max_steps  5000 \
#    --deepspeed configs/ds_configs/stage2.config \
#    --run_name Ours_CL_round1 \
#    --max_source_length 5 \
#    --max_target_length 512 \
#    --generation_max_length 512 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy epoch \
#    --save_strategy epoch \
#    --top_k $topk  \


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/Ours_CL_configs/lookback_task1590_diplomacy_text_generation \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task1590_diplomacy_text_generation \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 32 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate $lr \
#    --max_steps  5000 \
#    --deepspeed configs/ds_configs/stage2.config \
#    --run_name Ours_CL_round1 \
#    --max_source_length 5 \
#    --max_target_length 512 \
#    --generation_max_length 512 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy epoch \
#    --save_strategy epoch \
#    --top_k $topk  \


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/Ours_CL_configs/lookback_task1572_samsum_summary \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task1572_samsum_summary \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 32 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate $lr \
#    --max_steps  5000 \
#    --deepspeed configs/ds_configs/stage2.config \
#    --run_name Ours_CL_round1 \
#    --max_source_length 5 \
#    --max_target_length 512 \
#    --generation_max_length 512 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy epoch \
#    --save_strategy epoch \
#    --top_k $topk  \


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/Ours_CL_configs/lookback_task073_commonsenseqa_answer_generation \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task073_commonsenseqa_answer_generation \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 32 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate $lr \
#    --max_steps  5000 \
#    --deepspeed configs/ds_configs/stage2.config \
#    --run_name Ours_CL_round1 \
#    --max_source_length 5 \
#    --max_target_length 512 \
#    --generation_max_length 512 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy epoch \
#    --save_strategy epoch \
#    --top_k $topk  \


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/Ours_CL_configs/lookback_task363_sst2_polarity_classification \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir logs_and_outputs/Ours_CL/outputs_lr_0001_topk_${topk}/lookback_task363_sst2_polarity_classification \
#    --per_device_train_batch_size 16 \
#    --per_device_eval_batch_size 32 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate $lr \
#    --max_steps  5000 \
#    --deepspeed configs/ds_configs/stage2.config \
#    --run_name Ours_CL_round1 \
#    --max_source_length 5 \
#    --max_target_length 512 \
#    --generation_max_length 512 \
#    --add_task_name False \
#    --add_dataset_name False \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy epoch \
#    --save_strategy epoch \
#    --top_k $topk  \
