import json
import os
import sys

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False,indent=2)

def cal_continue_learning_metrics(scores_array, individual_scores):
    task_num=len(scores_array)

    Cl=sum(scores_array[-1])/task_num

    fgt_list=[]
    for t_idx in range(task_num-1):
        history=[line[t_idx] for line in scores_array[:-1]]
        history_best=max(history)
        fgt_list.append(history_best-scores_array[-1][t_idx])
    Fgt=sum(fgt_list)/len(fgt_list)

    Fwt=sum([scores_array[i][i] for i in range(task_num)])/task_num - 50.94

    Bwt=sum([scores_array[-1][i] - scores_array[i][i] for i in range(task_num)])/task_num
    
    return {
        'Cl':Cl,
        'Fgt':Fgt,
        'Fwt':Fwt,
        'Bwt':Bwt,
    }

run_name = sys.argv[1]

single_path = sys.argv[2]

with open(f"logs_and_outputs/{run_name}/outputs/task_order.txt", 'r') as f:
    data_list = f.readlines()
data_list = data_list[0].split(',')

task_num=len(data_list)

result_root_path=f'logs_and_outputs/{run_name}/outputs'
single_root_path=f'logs_and_outputs/{single_path}/outputs'

scores=[]
for i in range(len(data_list)):
    score_line=[]
    print(data_list[i])
    inference_result=load_json(f'{result_root_path}/{i+1}-{data_list[i]}/all_results.json')
    for j in range(i+1):
        score=inference_result[f'predict_eval_rougeL_for_{data_list[j]}'] #  "predict_exact_match_for_" for Long Sequence
        score_line.append(score)
    score_line.extend([0]*(task_num-i-1))
    scores.append(score_line)

with open(os.path.join(single_root_path, "task_order.txt"), 'r') as f:
    single_task_order = f.readlines()
    single_task_list = single_task_order[0].split(',')
individual_scores=[]

for i in range(task_num):
    inference_result=load_json(f'{single_root_path}/{i+1}-{single_task_list[i]}/all_results.json')
    score=inference_result[f'predict_eval_rougeL_for_{single_task_list[i]}']
    individual_scores.append(score)

cl_scores=cal_continue_learning_metrics(scores, individual_scores)
print(json.dumps(cl_scores,indent=2))

from tabulate import tabulate
title=list(range(task_num))
print(tabulate([individual_scores], headers=title, tablefmt='fancy_grid'))
with open(os.path.join("results", run_name + '.txt'), 'w') as f:
    f.write(str(cl_scores))
    f.write('\n')

    f.write(tabulate([individual_scores], headers=title, tablefmt='fancy_grid'))
    f.write('\n')


    title=['']+title
    scores_line=[[i]+line for i,line in enumerate(scores)]
    print(tabulate(scores_line, headers=title, tablefmt='fancy_grid'))

    f.write(tabulate(scores_line, headers=title, tablefmt='fancy_grid'))

