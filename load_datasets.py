import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import RobertaTokenizer, BertTokenizer, Qwen2Tokenizer, Qwen2Config, RobertaConfig, BertConfig
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import RobertaForSequenceClassification, BertModel, Qwen2ForSequenceClassification
from collections import defaultdict
import pandas as pd
import random
import pickle
from example import Example

dataset_path = "/home/likunhao/python_projects/datasets/"
model_path = "/home/likunhao/python_projects/"

def read_data(filename):
    examples = []
    for line in open(filename):
        line = line.strip().split("\t")
        topic = line[0]
        age = line[1]
        gender = line[2]
        user = line[3]
        text = line[4]
        if topic != "None":
            meta = set()
            if age == "1":
                meta.add(0)
            if gender == "f":
                meta.add(1)
            topic = int(topic)
            examples.append(Example(text, topic, meta))
    return examples

def get_balanced_distribution(examples):
    signatures = defaultdict(list)
    for ex in examples:
        meta = ex.get_aux_labels()
        signatures[tuple(meta)].append(ex)
    min_num = 10**10
    subcorpora = list(signatures.values())
    for subcorpus in subcorpora:
        if len(subcorpus) < min_num:
            min_num = len(subcorpus)
    balanced_dataset = []
    for subcorpus in subcorpora:
        random.shuffle(subcorpus)
        balanced_dataset.extend(subcorpus[:min_num])
    random.shuffle(balanced_dataset)
    return balanced_dataset

class BlogDataset(Dataset):
    def __init__(self, max_length, tokenizer):
        self.max_length = max_length
        self.data, self.num_labels = self.load_data()
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def load_data(self):
        Data = {}
        file_path = "/home/likunhao/python_projects/datasets/blog_dataset.pkl"
        num_classes = 10
        with open(file_path, "rb") as file:
            examples = pickle.load(file)
        for i, item in enumerate(examples):
            Data[i] = item
        return Data, num_classes
    
    def collate_fn(self, batch_samples):
        batch_sentences = list()
        batch_labels = list()
        for sample in batch_samples:
            batch_sentences.append(sample.sentence)
            batch_labels.append(sample.label)
        X = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
        y = torch.tensor(batch_labels)
        return X, y

    def collate_fn_aia(self, batch_samples):
        batch_sentences = list()
        batch_labels = list()
        batch_metadata = list()
        for sample in batch_samples:
            batch_sentences.append(sample.sentence)
            batch_labels.append(sample.label)
            batch_metadata.append(sample.metadata)
        X = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
        y = torch.tensor(batch_labels)
        return X, y, batch_metadata


class FP_Dataset(Dataset):
    def __init__(self, max_length, tokenizer):
        self.max_length = max_length
        self.data, self.num_labels = self.load_data()
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def load_data(self):
        Data = {}
        ds_path = dataset_path + "financial_phrasebank"
        ch_path = dataset_path + "financial_phrasebank/data"
        d_name = "sentences_allagree"
        ds = load_dataset(path=ds_path, cache_dir=ch_path, name=d_name, trust_remote_code=True)
        unique_labels = set(ds['train']['label'])
        num_classes = len(unique_labels)
        for i, item in enumerate(ds['train']):
            Data[i] = item
        return Data, num_classes
    
    def collate_fn(self, batch_samples):
        batch_sentences = list()
        batch_labels = list()
        for sample in batch_samples:
            batch_sentences.append(sample['sentence'])
            batch_labels.append(sample['label'])
        X = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)
        y = torch.tensor(batch_labels)
        return X, y


class MRPC_Dataset(Dataset):
    def __init__(self, data_type, max_length, tokenizer):
        self.max_length = max_length
        self.data, self.num_labels = self.load_data(data_type)
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def load_data(self, data_type):
        Data = {}
        ds_path = dataset_path + "mrpc/"
        num_classes = 2
        with open (ds_path + f"msr_paraphrase_{data_type}.txt", "r") as f:
            lines = f.readlines()
        for i, item in enumerate(lines[1:]):
            Data[i] = item
        return Data, num_classes
    
    def collate_fn(self, batch_samples):
        batch_sentences1 = list()
        batch_sentences2 = list()
        batch_labels = list()
        for sample in batch_samples:
            items = sample.split("\t")
            batch_sentences1.append(items[3])
            batch_sentences2.append(items[4])
            batch_labels.append(int(items[0]))
        X = self.tokenizer(batch_sentences1, batch_sentences2, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)
        y = torch.tensor(batch_labels)
        return X, y

class Copanli_Dataset(Dataset):
    def __init__(self, max_length, tokenizer):
        self.max_length = max_length
        self.data, self.num_labels = self.load_data()
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def load_data(self):
        Data = {}
        ds_path = dataset_path + "copa_nli/data/full-00000-of-00001.parquet"
        num_classes = 2
        df = pd.read_parquet(ds_path)
        for i, item in enumerate(df.itertuples()):
            Data[i] = item
        return Data, num_classes
    
    def collate_fn(self, batch_samples):
        batch_sentences1 = list()
        batch_sentences2 = list()
        batch_labels = list()
        for sample in batch_samples:
            batch_sentences1.append(sample[1])
            batch_sentences2.append(sample[2])
            batch_labels.append(int(sample[3]))
        X = self.tokenizer(batch_sentences1, batch_sentences2, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)
        y = torch.tensor(batch_labels)
        return X, y


def load_dataloader(dataset_name, llm, BATCH, attack_type, type="same_random", random_seed=42):
    if dataset_name == "fp":
        num_labels = 3
    if dataset_name == "copanli":
        num_labels = 2
    if dataset_name == "blog":
        num_labels = 10
    if dataset_name == "mrpc":
        num_labels = 2

    if llm == "roberta":
        config = RobertaConfig.from_pretrained(model_path + "roberta-base/config.json")
        config.num_labels = num_labels
        llm_model = RobertaForSequenceClassification.from_pretrained(model_path + "roberta-base", config=config)
        tokenizer = RobertaTokenizer.from_pretrained(model_path + "roberta-base", trust_remote_code=True)
    elif llm == "roberta-large":
        config = RobertaConfig.from_pretrained(model_path + "roberta-large/config.json")
        config.num_labels = num_labels
        llm_model = RobertaForSequenceClassification.from_pretrained(model_path + "roberta-large", config=config)
        tokenizer = RobertaTokenizer.from_pretrained(model_path + "roberta-large", trust_remote_code=True)
    elif llm == "bert":
        config = BertConfig.from_pretrained(model_path + "bert-base-uncased/config.json")
        config.num_labels = num_labels
        llm_model = BertModel.from_pretrained(model_path + "bert-base-uncased", config=config)
        tokenizer = BertTokenizer.from_pretrained(model_path + "bert-base-uncased", trust_remote_code=True)
    elif llm == "qwen":
        config = Qwen2Config.from_pretrained(model_path + "Qwen2-0.5B-Instruct/config.json")
        config.num_labels = num_labels
        llm_model = Qwen2ForSequenceClassification.from_pretrained(model_path + "Qwen2-0.5B-Instruct", config=config)
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path + "Qwen2-0.5B-Instruct", trust_remote_code=True)
        llm_model.config.pad_token_id = tokenizer.pad_token_id
    max_position_embeddings = llm_model.config.max_position_embeddings
    padding_id = tokenizer.pad_token_id
    if dataset_name == "blog":
        ds = BlogDataset(max_position_embeddings, tokenizer)
        num_classes = ds.num_labels
        train_size = int(0.8 * len(ds))
        val_size = int(0.1 * len(ds))
        test_size = len(ds) - train_size - val_size
        if type == "random":
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, val_size, test_size])
        if type == "same_random":
            torch.manual_seed(random_seed)
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, val_size, test_size])
        if type == "normal":
            indices = list(range(len(ds)))
            train_indices = indices[val_size + test_size:]
            val_indices = indices[test_size:val_size + test_size]
            test_indices = indices[:test_size]
            train_dataset = Subset(ds, train_indices)
            val_dataset = Subset(ds, val_indices)
            test_dataset = Subset(ds, test_indices)
        if attack_type == "eia":
            train_loader = DataLoader(train_dataset, batch_size=BATCH, collate_fn=ds.collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=BATCH, collate_fn=ds.collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=BATCH, collate_fn=ds.collate_fn)
            minidataset = Subset(train_dataset, torch.randperm(len(train_dataset))[:int(0.2 * len(train_dataset))])
            miniloader = DataLoader(minidataset, batch_size=BATCH, collate_fn=ds.collate_fn)
        if attack_type == "aia":
            train_loader = DataLoader(train_dataset, batch_size=BATCH, collate_fn=ds.collate_fn_aia)
            val_loader = DataLoader(val_dataset, batch_size=BATCH, collate_fn=ds.collate_fn_aia)
            test_loader = DataLoader(test_dataset, batch_size=BATCH, collate_fn=ds.collate_fn_aia)
            minidataset = Subset(train_dataset, torch.randperm(len(train_dataset))[:int(0.2 * len(train_dataset))])
            miniloader = DataLoader(minidataset, batch_size=BATCH, collate_fn=ds.collate_fn_aia)
    if dataset_name == "mrpc":
        ds = None
        train_dataset = MRPC_Dataset('train', max_position_embeddings, tokenizer)
        test_dataset = MRPC_Dataset('test', max_position_embeddings, tokenizer)
        num_classes = train_dataset.num_labels
        new_train_dataset = Subset(train_dataset, indices = list(range(len(train_dataset))))
        new_test_dataset = Subset(test_dataset, indices = list(range(len(test_dataset))))
        train_loader = DataLoader(new_train_dataset, batch_size=BATCH, collate_fn=train_dataset.collate_fn)
        val_loader = DataLoader(new_test_dataset, batch_size=BATCH, collate_fn=test_dataset.collate_fn)
        test_loader = val_loader

        minidataset = Subset(new_train_dataset, torch.randperm(len(train_dataset))[:int(0.2 * len(train_dataset))])
        miniloader = DataLoader(minidataset, batch_size=BATCH, collate_fn=train_dataset.collate_fn)
    if dataset_name == "copanli" or dataset_name == "fp":
        if dataset_name == "copanli":
            ds = Copanli_Dataset(max_position_embeddings, tokenizer)
        if dataset_name == "fp":
            ds = FP_Dataset(max_position_embeddings, tokenizer)
        num_classes = ds.num_labels
        train_size = int(0.8 * len(ds))
        val_size = int(0.1 * len(ds))
        test_size = len(ds) - train_size - val_size
        if type == "random":
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, val_size, test_size])
        if type == "same_random":
            torch.manual_seed(random_seed)
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, val_size, test_size])
        if type == "normal":
            indices = list(range(len(ds)))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

            train_dataset = Subset(ds, train_indices)
            val_dataset = Subset(ds, val_indices)
            test_dataset = Subset(ds, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=BATCH, collate_fn=ds.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH, collate_fn=ds.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH, collate_fn=ds.collate_fn)

        minidataset = Subset(train_dataset, torch.randperm(len(train_dataset))[:int(0.2 * len(train_dataset))])
        miniloader = DataLoader(minidataset, batch_size=BATCH, collate_fn=ds.collate_fn)
    return train_loader, val_loader, test_loader, miniloader, llm_model, num_classes, max_position_embeddings, padding_id, ds
