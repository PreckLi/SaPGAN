import torch
import numpy as np
from collections import defaultdict
from transformers import RobertaTokenizer
from example import Example
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=128, help='Batch size')
parser.add_argument('--device', type=str, default="cuda:0", help='Device')
args = parser.parse_args()

def get_ui_values(dataset, num_classes):
    tokenizer = RobertaTokenizer.from_pretrained("/home/likunhao/python_projects/roberta-base", trust_remote_code=True)
    token_freq = defaultdict(lambda: defaultdict(int))
    for i in range(len(dataset)):
        if type(dataset[i]) == Example:
            text, label = dataset[i].sentence, dataset[i].label
        elif type(dataset) == torch.utils.data.dataset.ConcatDataset:
            items = dataset[i].split("\t")
            text = items[3] + items[4]
            label = int(items[0])
        else:
            text, label = dataset[i]['sentence'], dataset[i]['label']
        for token in tokenizer(text)['input_ids']:
            token_freq[label][token] += 1

    ui_values = defaultdict(lambda: defaultdict(float))

    for label, freq_dict in token_freq.items():
        for token, freq in freq_dict.items():
            p_t_given_c = freq / sum(token_freq[label].values())
            
            for other_label in range(num_classes):
                if other_label == label:
                    continue
                p_t_given_other = token_freq[other_label].get(token, 1e-10) / sum(token_freq[other_label].values())
                ui_values[token][label] += np.log(p_t_given_c / p_t_given_other)
    return ui_values