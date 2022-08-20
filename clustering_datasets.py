import random
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


def batch_collate(batch):
    input_ids, attention_mask, label = torch.utils.data._utils.collate.default_collate(batch)
    max_length = attention_mask.sum(dim=1).max().item()
    attention_mask, input_ids = attention_mask[:, :max_length], input_ids[:, :max_length]
    return input_ids, attention_mask, label


class DatasetAutoEnc(Dataset):
    def __init__(self, text, id_=None):
        super().__init__()
        self.id_ = id_
        self.text = text

    def __getitem__(self, idx):
        return self.text['input_ids'][idx], self.text['attention_mask'][idx], self.id_[idx]

    def __len__(self):
        return len(self.text['attention_mask'])


class ClusteringDataset:
    def __init__(self, max_len=128, batch_size=100, model_name="bert-base-uncased"):
        self.max_len = max_len
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self, text, label):
        text = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
            padding='max_length',
            return_tensors='pt'
        )
        label = list(map(int, label))
        num_clusters = len(set(label))
        dataset = DatasetAutoEnc(text, label)
        shuffled_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=batch_collate)

        return shuffled_dataloader, num_clusters

    def init_trec(self):
        dataset = load_dataset("trec")
        text = dataset["train"]["text"]
        label = dataset["train"]["label-coarse"]
        return self.prepare_data(text, label)

    def init_dbpedia(self):
        dataset = load_dataset("dbpedia_14")
        text = dataset["train"]["content"]
        label = dataset["train"]["label"]
        random.seed(42)
        random.shuffle(text)
        random.seed(42)
        random.shuffle(label)
        text = text[:10000]
        label = label[:10000]
        return self.prepare_data(text, label)

    def init_tweets(self):
        df = pd.read_csv('datasets/tweet.csv')
        label = df['label'].tolist()
        text = df['text'].tolist()
        return self.prepare_data(text, label)

    def init_agnews(self):
        df = pd.read_csv('datasets/agnews.csv')
        label = df['label'].tolist()
        text = df['text'].tolist()
        return self.prepare_data(text, label)

    def init_short(self):
        df = pd.read_csv('datasets/short.csv')
        label = df['label'].tolist()
        text = df['text'].tolist()
        return self.prepare_data(text, label)

    def init_yelp(self):
        dataset = load_dataset("yelp_review_full")
        text = dataset["train"]["text"]
        label = dataset["train"]["label"]
        random.seed(42)
        random.shuffle(text)
        random.seed(42)
        random.shuffle(label)
        text = text[:10000]
        label = label[:10000]
        return self.prepare_data(text, label)
