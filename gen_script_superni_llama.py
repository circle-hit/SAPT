import json
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False,indent=2)

def load_jsonline(path):
    with open(path, 'r', encoding='utf-8') as f:
        result=[]
        for line_s in f:
            line=json.loads(line_s)
            result.append(line)
    return result

def write_jsonline(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            line_s=json.dumps(line, ensure_ascii=False)
            f.write(line_s)
            f.write('\n')

order_idx = 1

if order_idx == 1:
    all_tasks = ["task1572_samsum_summary",
                 "task363_sst2_polarity_classification",
                 "task1290_xsum_summarization",
                 "task181_outcome_extraction",
                 "task002_quoref_answer_generation",
                 "task1510_evalution_relation_extraction",
                 "task639_multi_woz_user_utterance_generation",
                 "task1729_personachat_generate_next",
                 "task073_commonsenseqa_answer_generation",
                 "task1590_diplomacy_text_generation",
                 "task748_glucose_reverse_cause_event_detection",
                 "task511_reddit_tifu_long_text_summarization",
                 "task591_sciq_answer_generation",
                 "task1687_sentiment140_classification",
                 "task875_emotion_classification"] # Order 1
else:
    all_tasks = ['task748_glucose_reverse_cause_event_detection',
                 'task073_commonsenseqa_answer_generation',
                 'task1590_diplomacy_text_generation',
                 'task639_multi_woz_user_utterance_generation',
                 'task1572_samsum_summary',
                 'task1687_sentiment140_classification',
                 'task591_sciq_answer_generation',
                 'task363_sst2_polarity_classification',
                 'task1510_evalution_relation_extraction',
                 'task1729_personachat_generate_next',
                 'task181_outcome_extraction',
                 'task511_reddit_tifu_long_text_summarization',
                 'task002_quoref_answer_generation',
                 'task1290_xsum_summarization',
                 'task875_emotion_classification'] # Order 2
    
dataset_list = all_tasks
task_order = ','.join(all_tasks)

config_template={
    "SuperNI": [
    ],
}

import os
import pathlib
import numpy as np
from copy import deepcopy

lora_r = 4
lora_alpha = 32
lora_dropout = 0.
kl_ratio = 1
attn_temperature = 1
learning_rate = 5e-5
num_train_epochs = 50
attn_lr = 0.
trans_hidden_dim = 100
replay_after_n_epoch = 0

run_name = f"your_job_name"

history_config=[]
for one_data_name in dataset_list:

    pathlib.Path(f'./configs/{run_name}_configs/{one_data_name}').mkdir(parents=True, exist_ok=True)

    config={
        "sampling strategy": "full",
        "dataset name": f"{one_data_name}"
    } 
    history_config.append(config)

    dev_config=deepcopy(config_template)
    dev_config['SuperNI'].append(config)
    write_json(f'./configs/{run_name}_configs/{one_data_name}/dev_tasks.json', dev_config)
    
    train_config=deepcopy(config_template)
    train_config['SuperNI'].append(config)
    write_json(f'./configs/{run_name}_configs/{one_data_name}/train_tasks.json', train_config)

    test_config=deepcopy(config_template)
    test_config['SuperNI'].extend(history_config)
    write_json(f'./configs/{run_name}_configs/{one_data_name}/test_tasks.json', test_config)

sh_str=rf'''#!/bin/bash
#SBATCH -J cl                           
#SBATCH -o cl-%j.out                       
#SBATCH -p compute 
#SBATCH -N 1                           
#SBATCH -t 20:00:00   
#SBATCH --mem 128G 
#SBATCH --gres=gpu:a100-sxm4-80gb:1  

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)  

deepspeed --num_gpus=4 src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path your_llama_model_path \
   --data_dir CL_Benchmark \
   --task_order {task_order} \
   --task_config_dir configs/{run_name}_configs/{dataset_list[0]} \
   --output_dir logs_and_outputs/{run_name}/outputs/1-{dataset_list[0]} \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 4 \
   --learning_rate {learning_rate} \
   --attn_lr {attn_lr} \
   --num_train_epochs {num_train_epochs} \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name {run_name} \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r {lora_r} \
   --lora_alpha {lora_alpha} \
   --lora_dropout {lora_dropout} \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio {kl_ratio} \
   --attn_temperature {attn_temperature} \
   --trans_hidden_dim {trans_hidden_dim} \

rm -rf logs_and_outputs/{run_name}/outputs/1-{dataset_list[0]}/checkpoint*
   
sleep 5
'''

previous_lora_path_list = []
for idx in range(len(dataset_list)-1):

    previous_lora_path_list.append(f"logs_and_outputs/{run_name}/outputs/{idx+1}-{dataset_list[idx]}/saved_weights")
    previous_lora_path = ','.join(previous_lora_path_list)
    sh_str+=rf'''

deepspeed --num_gpus=4 src/run_llama.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path your_llama_model_path \
   --load_checkpoint_from logs_and_outputs/{run_name}/outputs/{idx+1}-{dataset_list[idx]}/saved_weights/trans_input.pt \
   --previous_lora_path {previous_lora_path} \
   --previous_prompt_key_path logs_and_outputs/{run_name}/outputs/{idx+1}-{dataset_list[idx]}/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order {task_order} \
   --gen_data_dir generated_data/lora_gen_superni_llama \
   --task_config_dir configs/{run_name}_configs/{dataset_list[idx+1]} \
   --output_dir logs_and_outputs/{run_name}/outputs/{idx+2}-{dataset_list[idx+1]} \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 4 \
   --learning_rate {learning_rate} \
   --attn_lr {attn_lr} \
   --num_train_epochs {num_train_epochs} \
   --bf16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name {run_name} \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_rougeL_for_{dataset_list[idx+1]} \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r {lora_r} \
   --lora_alpha {lora_alpha} \
   --lora_dropout {lora_dropout} \
   --add_instruction_replay \
   --data_replay_freq 1 \
   --replay_after_n_epoch {replay_after_n_epoch} \
   --kl_ratio {kl_ratio} \
   --attn_temperature {attn_temperature} \
   --trans_hidden_dim {trans_hidden_dim} \

rm -rf logs_and_outputs/{run_name}/outputs/{idx+2}-{dataset_list[idx+1]}/checkpoint*

sleep 5
'''
    
sh_str+=rf'''
python score.py {run_name} single_train_results_path
'''
    
with open(f'{run_name}.sh', 'w') as f:
    f.write(sh_str)