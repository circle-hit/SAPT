import logging

import torch
from transformers.data.data_collator import *


logger = logging.getLogger(__name__)

SUPPORTED_DECODER_MODELS = ['llama']
SUPPORTED_SEQ2SEQ_MODELS = ['t5']

def check_model(model_name, supported_models):
    for sup_model in supported_models:
        if sup_model.lower() in model_name.lower():
            return True

    return False

def replace_sublist(lst, sublist, replacement):
    n = len(lst)
    m = len(sublist)
    
    for i in range(n - m + 1):
        if lst[i:i+m] == sublist:
            return lst[:i] + replacement + lst[i+m:]
    
    return lst

@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_dataset_name: bool = False
    add_instruction_replay: bool = True
    common_dataset_name: str = None
    text_only: bool = False
    num_examples: int = 0
    input_record_file: str = None

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        model_name = self.model.config._name_or_path
        # print(model_name)
        if check_model(model_name, SUPPORTED_DECODER_MODELS):
            model_inputs = self.decoder_call(batch, return_tensors)
        elif check_model(model_name, SUPPORTED_SEQ2SEQ_MODELS):
            model_inputs = self.seq2seq_call(batch, return_tensors)
        else:
            raise ValueError('Unsupport model {}!'.format(model_name))

        return model_inputs

    def get_instruction(self, instance):
        # "instructions \n options \n {0} \n Answer: "
        instruction = instance['Instance']["instruction"]
        content = instance['Instance']['sentence']

        # add task/ds prefix
        prefix = ''
        if self.add_task_name:
            prefix += "Task:" + instance['Task'] + '\n'
        if self.add_dataset_name:
            ds_name = self.common_dataset_name if self.common_dataset_name else instance['Dataset']
            prefix = prefix + "Dataset:"
            prefix = prefix + ds_name + '\n' if prefix else instance['Dataset'] + '\n'
        if prefix:
            instruction = prefix + instruction

        # TODO, support few shot
        # add few shot samples
        samples = ''
        if len(instance['Samples']) > 0:
            raise Exception('Few shot is coming soon...')
        if samples:
            content = samples + content
        # TODO, fix bug
        if self.add_instruction_replay:
            try:
                instruction = instruction.format(content)
            finally:
                return instruction
        else:
            return instruction


    def seq2seq_call(self, batch, return_tensors):
        sources = []
        labels = []

        for instance in batch:
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)

            source = instruction
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        # TODO, support online demo
        if self.text_only:
            model_inputs = {"inputs": sources, "labels": labels}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    labels,
                    max_length=self.max_target_length,
                    padding=self.padding,
                    return_tensors=return_tensors,
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of
                )
            label_mask = labels["attention_mask"].bool()
            model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

            # prepare decoder_input_ids
            if self.model is not None:
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
                model_inputs["decoder_input_ids"] = decoder_input_ids

            self._save_samples(model_inputs, sources, labels)

        return model_inputs

    def decoder_call(self, batch, return_tensors):
        self.tokenizer.padding_side = 'left'
        input_ids= []
        attention_mask= []
        input_ids_wo_label = []
        labels= []

        for instance in batch:
            label = instance['Instance']['label']
            instruction = self.get_instruction(instance)

            # add bos and eos
            task_input = instruction
            label = label + self.tokenizer.eos_token

            tokenized_input = self.tokenizer(task_input, add_special_tokens=False)["input_ids"]
            if len(tokenized_input)>self.max_source_length:
                tokenized_input=tokenized_input[:self.max_source_length]

            tokenized_label = self.tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(tokenized_label)>self.max_target_length:
                tokenized_label=tokenized_label[:self.max_target_length]

            # (input) for inference, (input + label) for training
            if instance['subset'] in ['dev', 'test']:
                input_ids.append(tokenized_input)
                input_ids_wo_label.append(tokenized_input)
                labels.append([self.label_pad_token_id]*len(tokenized_input))
            else:
                input_ids.append(tokenized_input+tokenized_label)
                input_ids_wo_label.append(tokenized_input)
                labels.append([self.label_pad_token_id]*len(tokenized_input)+tokenized_label)
        
        inputs_length=[len(i) for i in input_ids]
        inputs_length_wo_label = [len(i) for i in input_ids_wo_label]

        max_length=max(inputs_length)
        max_length_wo_label=max(inputs_length_wo_label)
        for i,(l,l_wo) in enumerate(zip(inputs_length, inputs_length_wo_label)):
            input_ids[i]=[self.tokenizer.pad_token_id]*(max_length-l) + input_ids[i]
            labels[i]=[self.label_pad_token_id]*(max_length-l) + labels[i]
            input_ids_wo_label[i] = [self.tokenizer.pad_token_id]*(max_length_wo_label-l_wo) + input_ids_wo_label[i]
            attention_mask.append([0]*(max_length-l) + [1]*l)

        input_ids=torch.tensor(input_ids)
        attention_mask=torch.tensor(attention_mask)
        labels=torch.tensor(labels)
        input_ids_wo_label=torch.tensor(input_ids_wo_label)
        model_inputs={
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'input_ids_wo_label': input_ids_wo_label,
        }
        return model_inputs

    def _save_samples(self, model_inputs, sources, labels):
        if not self.input_record_file:
            return

        loss_label = []
        if hasattr(model_inputs, 'loss_mask'):
            for loss, id in zip(model_inputs.loss_mask, model_inputs.input_ids):
                loss_label.append(self.tokenizer.decode((loss * id).view(-1).int()))

            with open(self.input_record_file, 'a+', encoding='utf-8') as f:
                for text, label, mask_label in zip(sources, labels, loss_label):
                    f.write(text+'\n')
                    f.write(label + '\n')
                    f.write(mask_label+'\n\n')
        else:
            with open(self.input_record_file, 'a+', encoding='utf-8') as f:
                for text, label in zip(sources, labels['input_ids']):
                    f.write(text + '\n')
                    f.write(self.tokenizer.decode(label, clean_up_tokenization_spaces=False) + '\n')