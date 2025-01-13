import json
import random
import pathlib
import os
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

    "task1590_diplomacy_text_generation",
    "task1572_samsum_summary",
    "task639_multi_woz_user_utterance_generation",
    "task1290_xsum_summarization",

    "task073_commonsenseqa_answer_generation",
    "task363_sst2_polarity_classification",
]
# model='t5_xl'
model='t5_large'

for one_name in dataset_list:
    ckpt_name='lr_0001_topk_20'


    for rate in [0.05, 0.1, 0.5, 1]:
        origial_data=load_json(f'./generate_lookback/{model}_generate_instruction_input_lr_0001_topk_20/{one_name}/train.json')
        origial_len=len(origial_data['Instances'])

        output_root_path=f'./generate_lookback/{model}_generate_instruction_input_instances_{rate}_{ckpt_name}'
        pathlib.Path(f'{output_root_path}/{one_name}/').mkdir(parents=True, exist_ok=True)
        origial_data['Definition']=random.sample(origial_data['Definition'], 1)
        target_len=int(origial_len*rate)
        if target_len==0:
            target_len=1
        if len(origial_data['Instances'])>target_len:
            origial_data['Instances']=random.sample(origial_data['Instances'], target_len)
        write_json(f'{output_root_path}/{one_name}/train.json', origial_data)  
        write_json(f'{output_root_path}/{one_name}/test.json', origial_data)  
        write_json(f'{output_root_path}/{one_name}/dev.json', origial_data)  

