# Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Inference of a finetuned LLM on the task of predicting <in>, <out> or <other> special tokens from untagged sentences
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
import ipdb
import re
import pandas as pd
import csv
import numpy as np
from tqdm import tqdm
from math import sin, pi
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)


bnb4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

bnb8_config = BitsAndBytesConfig(
    load_in_8bit=True
)


logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Adaper model checkpoints."
            )
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class DataGenerationArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    prompt: str = field(
        default=None, 
        metadata={"help": "The prompt text file."}
    )
    data: str = field(
        default=None, 
        metadata={"help": "input data file."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_generation_length: int = field(
        default=512, 
        metadata={"help": "Max generation length."}
    )
    seed: int = field(
        default=1, 
        metadata={"help": "Random seed"}
    )
    max_input_length: int = field(
        default=2560, 
        metadata={"help": "Max input length."}
    )
    batch_size: int = field(
        default=1, 
        metadata={"help": "Batch size."}
    )
    output_path: str = field(
        default=None, 
        metadata={"help": "Place to store outputs."}
    )
    wp: bool = field(
        default=False, metadata={"help": "Use WPs for tempearture scaling"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataGenerationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set seed before initializing model.
    set_seed(data_args.seed)
    
    # Load the dataset 
    dataset = load_dataset('csv', data_files=data_args.data, sep='\t', index_col=None, quoting=csv.QUOTE_NONE, escapechar='\\')['train']
    # dataset = dataset.filter(lambda row: row['split']=='test')
    
    with open(data_args.prompt) as f:
        instructions = f.read()
    
    def wp_to_description(row):
        if ((row['win_prob']>=0) and (row['win_prob']<0.1)):
            row['game_state'] = row['opp'].title() + " are extremely likely to win."
        elif ((row['win_prob']>=0.1) and (row['win_prob']<0.25)):
            row['game_state'] = row['opp'].title() + " are likely to win."
        elif ((row['win_prob']>=0.25) and (row['win_prob']<0.48)):
            row['game_state'] = row['opp'].title() + " are slightly likely to win."
        elif ((row['win_prob']>=0.52) and (row['win_prob']<0.75)):
            row['game_state'] = row['team'].title() + " are slightly likely to win." 
        elif ((row['win_prob']>=0.75) and (row['win_prob']<0.9)):
            row['game_state'] = row['team'].title() + " are likely to win."
        elif ((row['win_prob']>=0.9) and (row['win_prob']<=1)):
            row['game_state'] = row['team'].title() + " are extremely likely to win."   
        else:
            row['game_state'] = "Both teams are equally likely to win." 
        return row
    
    dataset = dataset.map(wp_to_description)
    
    # Tokenize exactly like in Axolotl
    system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    
    def create_input_context(row):
        '''
        Takes an input row, and makes a long string with instruction, comment, in-group team, out-group team, as well as win probability.
        '''  
        input_prompt = (
            "COMMENT: " + row["tokenized_comment"] + "\n" +
            "IN-GROUP: " + row['team'].title() + "\n" +
            "OUT-GROUP: " + row['opp'].title() + "\n" 
        )
        
        if 'ling' in data_args.prompt:
            input_prompt += (
                "GAME STATE: " + row['game_state'] + "\n"
                "REF_EXPRESSIONS: "
            )
        elif 'wp' in data_args.prompt:
            input_prompt += (
                "WIN PROBABILITY: " + str(np.round(row['win_prob']*100, 1)) + "%\n" 
                "REF_EXPRESSIONS: "
            )
        else:
            input_prompt += (
                "REF_EXPRESSIONS: "
            )
            
        
        full_input = f"{system_prompt}\n\n### Instruction:\n{instructions}\n\n### Input:\n{input_prompt}\n\n### Response:\n"
        
        if data_args.wp:
            wp = sin(row['win_prob']*pi)
            if wp<1e-5:
                wp = 1e-5
            conf = np.mean([x/5 for x in eval(row['confs'])])
            return {'full_input': full_input, 'wp': wp, 'conf': conf}
            
        return {'full_input': full_input}
        
    dataset = dataset.map(create_input_context, load_from_cache_file=not data_args.overwrite_cache)
    
    ####### MODIFY TOKENIZER HERE TO ADD VOCAB ITEMS ########
    
    # Tokenizer config
    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, padding_side='left', **tokenizer_kwargs)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    ###### INITIALIZE AND MODIFY MODEL WITH NEW TOKENS ######
    if model_args.lora_path:
        quant_config = bnb4_config
    else:
        quant_config = bnb8_config
        
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        from_tf=bool(".ckpt" in model_args.model_path),
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )
    
    if model_args.lora_path:
        model = PeftModel.from_pretrained(model, model_args.lora_path)
    
    model = torch.compile(model)
    model.eval()
    tokenizer.model_max_length=data_args.max_input_length
    data_collator = DataCollatorWithPadding(tokenizer)
    
    def tokenize_function(inputs):
        tokenized_inputs = tokenizer(
            text=inputs['full_input'],
            padding=False,
            truncation=True,
            max_length=data_args.max_input_length,
            return_special_tokens_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
            is_split_into_words=False
        )
        if data_args.wp:
            tokenized_inputs['wp'] = inputs['wp']
            tokenized_inputs['conf'] = inputs['conf']
        return tokenized_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=list(dataset.features),
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    dataloader = DataLoader(tokenized_dataset, shuffle=False, batch_size=data_args.batch_size, collate_fn=data_collator)
    
    if not data_args.output_path:
        if model_args.lora_path:
            output_path = model_args.lora_path
        else:
            output_path = model_args.model_path
    else:
        output_path = data_args.output_path
    
    # regex pattern to get only input
    patt =  re.compile("\n:esnopseR ###\n\n(.*)\s:TNEMMOC\n:tupnI ###.*", re.DOTALL)
        
    with torch.no_grad():
        for batch in dataloader:
            model_input = {k: v.to('cuda') for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            if data_args.wp:
                wp = batch['wp'].detach().cpu().numpy().item()
                preds = model.generate(**model_input, max_new_tokens=data_args.max_generation_length, eos_token_id=terminators, do_sample=True, temperature=wp).detach().cpu().numpy()
            else:
                preds = model.generate(**model_input, max_new_tokens=data_args.max_generation_length, eos_token_id=terminators, do_sample=False).detach().cpu().numpy()
            
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            prompt_lengths = [len(x) for x in decoded_inputs]
            
            decoded_outputs = [decoded_preds[i][prompt_lengths[i]:] for i in range(len(prompt_lengths))]
            
            if data_args.wp:         
                filename = output_path + '/seed' + str(data_args.seed) + '_wp_'
            else:
                filename = output_path + '/seed' + str(data_args.seed) + '_'
                
            with open(filename + 'sample-output.txt', 'a') as f:
                for i, line in enumerate(decoded_outputs):
                    input_text = patt.search(decoded_inputs[i][::-1]).group(1)[::-1]
                    newline = input_text.strip() + "\n" + line.strip()
                    f.write(newline + "\n======\n")
            
            with open(filename + 'sample-sents.txt', 'a') as f: 
                decoded_targets = []
                for ind, s in enumerate(decoded_outputs):
                    if re.search(r'(.*)TARGET:\s(.*)', s):
                        decoded_targets.append(re.search(r'(.*)TARGET:\s(.*)', s).group(2).strip())
                    else:
                        decoded_targets.append("NONE" + re.search(r'(.*)TARGET:\s(.*)', decoded_inputs[ind]).group(2).strip())
                    
                for line in decoded_targets:
                    f.write(line+"\n")

if __name__ == "__main__":
    main()
