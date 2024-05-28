# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""CL_Benchmark Dataset."""

import json
import os
import random
import datasets
from hashlib import md5

logger = datasets.logging.get_logger(__name__)
TASK_CONFIG_FILES = {"train": "train_tasks.json", "dev": "dev_tasks.json", "test": "test_tasks.json"}
INSTRUCTION_STRATEGIES = ['single', 'multiple']
ANSWER_PREFIX = "Output:"
SINGLE_QUOTES_SUBSTITUTE = "#$%#"
AUX_PROB = 0.3


def gen_cache_path(cache_dir, data_args):
    hash_str = data_args.data_dir + data_args.task_config_dir + \
               str(data_args.max_num_instances_per_task) + str(data_args.max_num_instances_per_eval_task)
    hash_obj = md5(hash_str.encode("utf-8"))
    hash_id = hash_obj.hexdigest()
    cache_path = os.path.join(cache_dir, str(hash_id))

    return cache_path


def check_path(path):
    if not path or not os.path.exists(path):
        raise ValueError('{} is not valid, please check the input path!'.format(path))


def save_ds(instances, file_name):
    with open(file_name, "w+", encoding='utf-8') as fi:
        json.dump(instances, fi, ensure_ascii=False, indent=2)


class CLConfig(datasets.BuilderConfig):
    """
    Config dataset load procedure.

    Args:
        data_dir: task data dir, which contains the corresponding dataset dirs
        prompt_path: prompt json file, which saves task and its prompts map
        task_file: task config file, save training and testing split config, and sampling strategies.
         Support two sampling strategies: 'random' indicates random sampling, while 'full' means to return all samples.
        max_num_instances_per_task: max training sample size of each task
        max_num_instances_per_eval_task: max eval sample size of each task
    """

    def __init__(
            self,
            *args,
            data_dir=None,
            task_config_dir=None,
            num_examples=None,
            max_num_instances_per_task=None,
            max_num_instances_per_eval_task=None,
            over_sampling=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_examples = num_examples
        self.over_sampling = over_sampling
        self.task_configs = self._parse_task_config(task_config_dir)
        self.max_num_instances_per_task = max_num_instances_per_task
        self.max_num_instances_per_eval_task = max_num_instances_per_eval_task

    def _parse_task_config(self, task_config_dir):
        if not task_config_dir:
            return None

        task_configs = {}
        for task, file_name in TASK_CONFIG_FILES.items():
            task_config_file = os.path.join(task_config_dir, file_name)

            if not os.path.exists(task_config_file):
                raise ValueError('Please check {} config, {} not exists!'.format(task, task_config_file))

            with open(task_config_file, 'r+') as f:
                task_configs[task] = json.loads(f.read())

        return task_configs


class CLInstructions(datasets.GeneratorBasedBuilder):
    """CL Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = CLConfig
    BUILDER_CONFIGS = [
        CLConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "Task": datasets.Value("string"),
                    "Dataset": datasets.Value("string"),
                    "subset": datasets.Value("string"),
                    "Samples": [{
                        "id": datasets.Value("string"),
                        "sentence": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "ground_truth": datasets.Value("string")
                    }],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "sentence": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "instruction": datasets.Value("string"),
                        "ground_truth": datasets.Value("string")
                    }
                }
            ),
            supervised_keys=None
        )


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_configs is None:
            logger.error("Please provide right input: data_dir or task_config_dir!")

        # split dir save datasets
        # task config to specify train,dev,test
        split_dir = self.config.data_dir
        task_configs = self.config.task_configs

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['train'],
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['dev'],
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "dev"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['test'],
                    "max_num_instances_per_task": None,  # default load total test samples to test
                    "subset": "test"
                }),
        ]


    def _load_dataset(self, dataset_path):
        with open(dataset_path, encoding="utf-8") as task_f:
            s = task_f.read()
            instances = json.loads(s)
        
        return instances
    
    def load_LongSeq_dataset(self, dataset_path, dataset_name, sampling_strategy, max_num_instances, subset):

        data = self._load_dataset(dataset_path)
        print(list(data.keys()))
        input_mode='zeroshot'
        definition = ""
        if len(data["Definition"]) > 0:
            if input_mode=='fewshot' or input_mode=='zeroshot':
                if isinstance(data["Definition"], list):
                    definition = data["Definition"][0].strip() # TODO: should we use <Definition>?
                else:
                    definition = data["Definition"].strip()
                definition += "\n"

        sample_template = {"Task": "CL", "Dataset": dataset_name, "Samples": [], "subset": subset}

        for idx, instance in enumerate(data['Instances']):
            example = sample_template.copy()
            instruction = ""
            # add the input first.
            instruction += "{0}"
            instruction += "\n"
            instruction += "Output: "
            pos_examples = []
            if input_mode=='fewshot':
                for idx, pos_example in enumerate(data["Positive Examples"][:1]):
                    pos_example_str = f"Positive Example {idx+1} -\n"
                    pos_example_str += f"Input: {pos_example['input'].strip()}"
                    pos_example_str += "\n"
                    pos_example_str += f"Output: {pos_example['output'].strip()}"
                    pos_example_str += "\n" 
                    pos_examples.append(pos_example_str)

            instruction = definition + "".join(pos_examples) + instruction

            # print('-------------------')
            # print(instruction)
            # print('-------------------')

            if isinstance(instance["output"], list):
                label=instance["output"][random.randint(0, len(instance["output"])-1)]
            else:
                label=instance["output"]

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['input'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example

    def load_SuperNI_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):

        data = self._load_dataset(dataset_path)
        print(list(data.keys()))
        input_mode='zeroshot'
        definition = ""
        if input_mode=='fewshot' or input_mode=='zeroshot':
            if isinstance(data["Definition"], list):
                definition = "Definition: " + data["Definition"][0].strip() # TODO: should we use <Definition>?
            else:
                definition = "Definition: " + data["Definition"].strip()
            definition += "\n\n"

        sample_template = {"Task": "CL", "Dataset": dataset_name, "Samples": [], "subset": subset}

        for idx, instance in enumerate(data['Instances']):
            example = sample_template.copy()
            instruction = ""
            # add the input first.
            if input_mode=='fewshot' or input_mode=='zeroshot':
                instruction += "Now complete the following example -\n"
            instruction += "Input: {0}"
            instruction += "\n"
            instruction += "Output: "
            pos_examples = []
            if input_mode=='fewshot':
                for idx, pos_example in enumerate(data["Positive Examples"][:1]):
                    pos_example_str = f"Positive Example {idx+1} -\n"
                    pos_example_str += f"Input: {pos_example['input'].strip()}"
                    pos_example_str += "\n"
                    pos_example_str += f"Output: {pos_example['output'].strip()}"
                    pos_example_str += "\n" 
                    pos_examples.append(pos_example_str)

            instruction = definition + "".join(pos_examples) + instruction

            # print('-------------------')
            # print(instruction)
            # print('-------------------')

            if isinstance(instance["output"], list):
                label=instance["output"][random.randint(0, len(instance["output"])-1)]
            else:
                label=instance["output"]

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['input'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example


    def _generate_examples(self, path=None, task_config=None, max_num_instances_per_task=None, subset=None):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")

        for task in task_config:
            if task == 'SuperNI':
                load_func = self.load_SuperNI_dataset
            elif task == "Long_Sequence":
                load_func = self.load_LongSeq_dataset
            else:
                raise ValueError("Unsupport {} task, plz check {} task config!".format(task, subset))

            # load dataset
            for dataset in task_config[task]:
                ds_name = dataset["dataset name"]
                sampling_strategy = dataset.get("sampling strategy", "random")
                ds_path = os.path.join(path, task, ds_name, subset + '.json')
                print(ds_path)
                labels_path = None
                assert os.path.exists(ds_path)

                idx = -1
                instances = []
                for sample in load_func(ds_path, labels_path, ds_name, sampling_strategy, max_num_instances_per_task,
                                        subset):
                    idx += 1
                    instances.append(sample)
                    yield f"{task}##{ds_path}##{idx}", sample
