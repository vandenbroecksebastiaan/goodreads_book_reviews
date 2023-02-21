import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import DistilBertTokenizer

import pandas as pd
import csv
from tqdm import tqdm

from config import config

def collate_fn(batch):
    # If it is a train batch
    if len(batch[0]) == 4:
        input_id_batch = torch.vstack([i[0] for i in batch])
        attention_mask_batch = torch.vstack([i[1] for i in batch])
        variables_batch = torch.vstack([i[2] for i in batch])
        target_batch = torch.vstack([i[3] for i in batch])
        return (input_id_batch, attention_mask_batch, variables_batch, target_batch)
    # Else it is a test batch and does not contain a target
    else:
        input_id_batch = torch.vstack([i[0] for i in batch])
        attention_mask_batch = torch.vstack([i[1] for i in batch])
        variables_batch = torch.vstack([i[2] for i in batch])
        return (input_id_batch, attention_mask_batch, variables_batch)

class ReviewDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()

        self.train = train
        self.load_data()
        self.preprocess_review_text()
        self.preprocess_date_added()
        self.preprocess_read_at()
        self.preprocess_started_at()
        self.preprocess_n_votes()
        self.preprocess_n_comments()

        if self.train:
            self.preprocess_rating()
            self.add_weights()

        self.tokenizer = DistilBertTokenizer.from_pretrained(config["model_name"])

    def load_data(self):
        if self.train: path = "data/goodreads_train.csv"
        else: path = "data/goodreads_test.csv"

        data = pd.read_csv(path)
        print("Data shape:", data.shape)

        self.user_id = data["user_id"]
        self.book_id = data["book_id"]
        self.review_id = data["review_id"]
        if self.train: self.rating = data["rating"].tolist()
        self.review_text = data["review_text"].tolist()
        self.date_added = data["date_added"]
        self.date_read_at = data["read_at"]
        self.date_started_at = data["started_at"]
        self.n_votes = data["n_votes"]
        self.n_comments = data["n_comments"]

    def preprocess_review_text(self):
        desc = "Preprocess review text"
        self.review_text = [i.replace("\n", "").replace("\t", "").lower()
                            for i in tqdm(self.review_text, desc=desc)]
        
    def preprocess_rating(self):
        # Scale the rating to a max of 1
        self.rating = torch.Tensor(self.rating).cuda()
        
    def add_weights(self):
        # Add weights based on continent
        rating_class, weights = self.rating.unique(return_counts=True)
        print(rating_class, weights)
        inv_weights = 1 / weights
        self.weights = inv_weights
        
    def preprocess_date_added(self):
        self.date_added = self.date_added.str.split()
        self.date_added = self.date_added.apply(lambda x: x[2] + " " + x[1]
                                                          + " " + x[5])
        self.date_added = pd.to_datetime(self.date_added,
                                         infer_datetime_format=True)
        self.date_added = (self.date_added - self.date_added.mean()) \
                               / self.date_added.std()
        self.date_added = self.date_added.tolist()

    def preprocess_read_at(self):
        self.date_read_at = self.date_read_at.str.split()
        self.date_read_at = self.date_read_at.apply(
            lambda x: x[2] + " " + x[1] + " " + x[5] if type(x) == list else x
        )
        self.date_read_at = pd.to_datetime(self.date_read_at, infer_datetime_format=True)
        self.date_read_at = (self.date_read_at - self.date_read_at.mean()) \
                               / self.date_read_at.std()
        self.date_read_at = self.date_read_at.fillna(value=0)
        self.date_read_at = self.date_read_at.tolist()

    def preprocess_started_at(self):
        self.date_started_at = self.date_started_at.str.split()
        self.date_started_at = self.date_started_at.apply(
            lambda x: x[2] + " " + x[1] + " " + x[5] if type(x) == list else x
        )
        self.date_started_at = pd.to_datetime(self.date_started_at)
        self.date_started_at = (self.date_started_at - self.date_started_at.mean()) \
                               / self.date_started_at.std()
        self.date_started_at = self.date_started_at.fillna(value=0)
        self.date_started_at = self.date_started_at.tolist()
        
    def preprocess_n_votes(self):
        self.n_votes = (self.n_votes - self.n_votes.mean()) \
                               / self.n_votes.std()
        self.n_votes = self.n_votes.tolist()
        
    def preprocess_n_comments(self):
        self.n_comments = (self.n_comments - self.n_comments.mean()) \
                               / self.n_comments.std()
        self.n_comments = self.n_comments.tolist()

    def __len__(self):
        return len(self.review_text)

    def __getitem__(self, index):
        x_data = self.tokenizer(self.review_text[index],
                                max_length=config["max_length"],
                                padding=True, truncation=True)

        input_id = x_data["input_ids"]
        attention_mask = x_data["attention_mask"]
        if self.train: target = self.rating[index].to(int)

        input_id = torch.Tensor(input_id).to(int).cuda()
        attention_mask = torch.Tensor(attention_mask).to(int).cuda()
        if self.train: target = torch.Tensor(torch.eye(n=6).cuda()[:, target])
        
        if input_id.shape[0] != config["max_length"]:
            input_id = torch.nn.functional.pad(
                input_id,
                (0, config["max_length"]-input_id.shape[0]),
                "constant", 0
            ).squeeze()
        if attention_mask.shape[0] != config["max_length"]:
            attention_mask = torch.nn.functional.pad(
                attention_mask,
                (0, config["max_length"]-attention_mask.shape[0]),
                "constant", 0
            ).squeeze()
            
        date_added = self.date_added[index]
        date_read_at = self.date_read_at[index]
        date_started_at = self.date_started_at[index]
        n_votes = self.n_votes[index]
        n_comments = self.n_comments[index]
        variables = torch.Tensor([date_added, date_read_at, date_started_at,
                                  n_votes, n_comments]).cuda()
        
        if self.train: return (input_id, attention_mask, variables, target)
        else: return (input_id, attention_mask, variables)
