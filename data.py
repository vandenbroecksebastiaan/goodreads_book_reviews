import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import DistilBertTokenizer

import pandas as pd
from tqdm import tqdm

from config import config

def collate_fn(batch):
    # If it is a train batch
    if len(batch[0]) == 3:
        input_id_batch = pad_sequence([i[0] for i in batch])
        if input_id_batch.shape[0] != config["max_length"]:
            print("*** EXTRA PADDED ***")
            input_id_batch = torch.nn.functional.pad(
                input_id_batch,
                (0, 0, 0, config["max_length"]-input_id_batch.shape[0]),
                "constant", 0
            )
        attention_mask_batch = pad_sequence([i[1] for i in batch])
        if attention_mask_batch.shape[0] != config["max_length"]:
            print("*** EXTRA PADDED ***")
            attention_mask_batch = torch.nn.functional.pad(
                attention_mask_batch,
                (0, 0, 0, config["max_length"]-attention_mask_batch.shape[0]),
                "constant", 0
            )
        target_batch = torch.vstack([i[2] for i in batch]).cuda()
        return (input_id_batch, attention_mask_batch, target_batch)
    # Else it is a test batch and does not contain a target
    else:
        input_id_batch = pad_sequence([i[0] for i in batch])
        if input_id_batch.shape[0] != config["max_length"]:
            print("*** EXTRA PADDED ***")
            input_id_batch = torch.nn.functional.pad(
                input_id_batch,
                (0, 0, 0, config["max_length"]-input_id_batch.shape[0]),
                "constant", 0
            ).squeeze()
        attention_mask_batch = pad_sequence([i[1] for i in batch])
        if attention_mask_batch.shape[0] != config["max_length"]:
            print("*** EXTRA PADDED ***")
            attention_mask_batch = torch.nn.functional.pad(
                attention_mask_batch,
                (0, 0, 0, config["max_length"]-attention_mask_batch.shape[0]),
                "constant", 0
            ).squeeze()
        return (input_id_batch, attention_mask_batch)
        

class ReviewDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        
        self.train = train
        self.load_data()
        self.preprocess_review_text()

        if self.train:
            self.preprocess_rating()
            self.add_weights()

        self.tokenizer = DistilBertTokenizer.from_pretrained(config["model_name"])
        
    def load_data(self):
        if self.train:
            data = pd.read_csv("data/goodreads_train.csv",
                               nrows=config["max_obs"])
        else:
            data = pd.read_csv("data/goodreads_test.csv")
        print("Data shape:", data.shape)

        self.user_id = data["user_id"].to_numpy()
        self.book_id = data["book_id"].to_numpy()
        self.review_id = data["review_id"].to_numpy()
        if self.train: self.rating = data["rating"].to_numpy()
        self.review_text = data["review_text"].to_numpy()
        self.date_added = data["date_added"].to_numpy()
        self.read_at = data["read_at"].to_numpy()
        self.started_at = data["started_at"].to_numpy()
        self.n_votes = data["n_votes"].to_numpy()
        self.n_comments = data["n_comments"].to_numpy()

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

    def __len__(self):
        return len(self.review_text)

    def __getitem__(self, index):
        x_data = self.tokenizer(self.review_text[index],
                                max_length=config["max_length"],
                                padding=True, truncation=True)

        input_id = x_data["input_ids"]
        attention_mask = x_data["attention_mask"]
        if self.train: target = int(self.rating[index])

        input_id = torch.Tensor(input_id).to(int).cuda()
        attention_mask = torch.Tensor(attention_mask).to(int).cuda()
        if self.train: target = torch.Tensor(torch.eye(n=6)[:, target])

        if self.train: return (input_id, attention_mask, target)
        else: return (input_id, attention_mask)
