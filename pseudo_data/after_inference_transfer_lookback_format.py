import json
import random
import pathlib
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
    # "task181_outcome_extraction",
    # "task591_sciq_answer_generation",
    # "task1729_personachat_generate_next",
    # "task1510_evalution_relation_extraction",
    # "task748_glucose_reverse_cause_event_detection",
    # "task002_quoref_answer_generation",
    # "task1687_sentiment140_classification",
    # "task511_reddit_tifu_long_text_summarization",
    # "task875_emotion_classification",

    # "task1590_diplomacy_text_generation",
    # "task1572_samsum_summary",
    # "task639_multi_woz_user_utterance_generation",
    # "task1290_xsum_summarization",

    "task073_commonsenseqa_answer_generation",
    "task363_sst2_polarity_classification",
]


for one_name in dataset_list:
    print(one_name)
    origial_data=load_json(f'./CL_Benchmark/Ours_CL/lookback_{one_name}/train.json')   
    origial_len=len(origial_data['Instances'])
    ckpt_name='lr_0001_topk_20'
    lookback_data=load_json(f'./logs_and_outputs/Ours_CL/{ckpt_name}/lookback_{one_name}/lookback_inference_result.json')
    instruction=set()
    instences=[]
    positive=[]
    for one_item in lookback_data['Instances']:
        text=one_item['input']
        if '__inp__' in text and '__ans__' in text:
            temp=text.split('__inp__')
            one_instru=temp[0].strip()
            if '__ans__' in one_instru:
                continue
            one_input,one_output=temp[1].split('__ans__')
            instruction.add(one_instru)
            instences.append({
                'input':one_input.strip(),
                'output':one_output.strip(),
            })
            positive.append({
                'input':one_input.strip(),
                'output':one_output.strip(),
            })
        elif '__inp__' in text:
            temp=text.split('__inp__')
            one_instru=temp[0].strip()
            one_input=temp[1].strip()
            instruction.add(one_instru)
            instences.append({
                'input':one_input,
                'output':'',
            })
        else:
            continue

    if one_name in ['task1590_diplomacy_text_generation', 'task1572_samsum_summary']:
        for one_item in lookback_data['Instances']:
            original_instru_len=len(origial_data['Definition'][0])
            text=one_item['input']
            if '__inp__' not in text and '__ans__' not in text:
                one_instru=text[:original_instru_len]
                one_input=text[original_instru_len:]
                instruction.add(one_instru)
                instences.append({
                    'input':one_input,
                    'output':'',
                })
                positive.append({
                    'input':one_input,
                    'output':'',
                })
            
    if len(positive)>3:
        positive=random.sample(positive, 3)
    origial_data['Definition']=list(instruction)
    origial_data['Positive Examples']=positive
    origial_data['Negative Examples']=[]
    origial_data['Instances']=instences

    print(f'{int(len(instences)*100/origial_len)}%')
    print(len(instences))
    output_root_path=f'./generate_lookback/generate_instruction_input_{ckpt_name.replace("outputs_", "")}'
    pathlib.Path(f'{output_root_path}/{one_name}/').mkdir(parents=True, exist_ok=True)
    write_json(f'{output_root_path}/{one_name}/train.json', origial_data)  
    write_json(f'{output_root_path}/{one_name}/test.json', origial_data)  
    write_json(f'{output_root_path}/{one_name}/dev.json', origial_data)  
    


    output_root_path=f'./generate_lookback/generate_instruction_input_instances_0.02_{ckpt_name.replace("outputs_", "")}'
    pathlib.Path(f'{output_root_path}/{one_name}/').mkdir(parents=True, exist_ok=True)
    origial_data['Definition']=random.sample(origial_data['Definition'], 1)
    target_len=int(origial_len*0.02)
    if target_len==0:
        target_len=1
    if len(origial_data['Instances'])>target_len:
        origial_data['Instances']=random.sample(origial_data['Instances'], target_len)
    write_json(f'{output_root_path}/{one_name}/train.json', origial_data)  
    write_json(f'{output_root_path}/{one_name}/test.json', origial_data)  
    write_json(f'{output_root_path}/{one_name}/dev.json', origial_data)  

