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


dataset_list=[
    "task181_outcome_extraction",
    "task591_sciq_answer_generation",
    "task1729_personachat_generate_next",
    "task1510_evalution_relation_extraction",
    "task748_glucose_reverse_cause_event_detection",
    "task002_quoref_answer_generation",
    "task1687_sentiment140_classification",
    "task511_reddit_tifu_long_text_summarization",
    "task875_emotion_classification",
    "task639_multi_woz_user_utterance_generation",
    "task1290_xsum_summarization",
    "task1590_diplomacy_text_generation",
    "task1572_samsum_summary",
    "task073_commonsenseqa_answer_generation",
    "task363_sst2_polarity_classification",
]

config_template={
    "Ours_CL": [
    ],
}

sh_str=rf'''#!/bin/bash
#SBATCH -J cl                           
#SBATCH -o cl-%j.out                       
#SBATCH -p compute 
#SBATCH -N 1                           
#SBATCH -t 5:00:00   
#SBATCH --mem 64G 
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1        


export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)  

lr=0.001
topk=20
'''
output_name=r'outputs_lr_0001_topk_${topk}'

for idx in range(len(dataset_list)):
    sh_str+=rf'''

CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path ~/workplace/A_pretrain_models/t5_large \
   --data_dir CL_Benchmark \
   --task_config_dir configs/Ours_CL_configs/lookback_{dataset_list[idx]} \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/Ours_CL/{output_name}/lookback_{dataset_list[idx]} \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 1 \
   --learning_rate $lr \
   --max_steps  5000 \
   --deepspeed configs/ds_configs/stage2.config \
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
'''
    
with open(f'./scripts/Ours_CL_lookback.sh', 'w') as f:
    f.write(sh_str)