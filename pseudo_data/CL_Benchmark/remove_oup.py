from glob import glob 
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

for path in glob('./Ours_CL/*'):
    for split in ['train', 'test']:
        data=load_json(f'{path}/{split}.json')
        if '__ans__' in data['Positive Examples'][0]['output']:
            for one in data['Instances']:
                one['output']=one['output'].split('__ans__')[0]
            for one in data['Positive Examples']:
                one['output']=one['output'].split('__ans__')[0]
            for one in data['Negative Examples']:
                one['output']=one['output'].split('__ans__')[0]
        write_json(f'{path}/{split}.json',data)
