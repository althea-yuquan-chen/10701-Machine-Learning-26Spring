import re, os, random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(10701)
random.seed(10701)
np.random.seed(10701)


class Yelp_Dataset(Dataset):
    def __init__(
        self, split: str, vocabulary_size: int = 10000, max_len=200, word_dict=None
    ):
        splits = {
            "train": "train-00000-of-00001.parquet",
            "test": "test-00000-of-00001.parquet",
        }
        pd_hf_prefix = "hf://datasets/fancyzhx/yelp_polarity/plain_text/"

        self.seed = 10701
        self.sample_frac = 0.15

        if split != "test" and split != "train":
            raise ValueError("Invalid split")

        pkl_file_name = f"yelp_{split}_sampled.pkl"
        if os.path.exists(pkl_file_name):
            df = pd.read_pickle(pkl_file_name)
        else:
            df = pd.read_parquet(pd_hf_prefix + splits[split])
            df = df.sample(
                frac=self.sample_frac, random_state=self.seed, ignore_index=True
            )
            df.to_pickle(pkl_file_name)

        self.df = df
        self.length = len(self.df)
        self.max_len = max_len
        self.vocab_size = vocabulary_size
        self.texts = []
        self.labels = []

        # Preprocess text
        self.df["text"] = self.df["text"].apply(self.preprocess_text)

        # Create a dictionary of words
        if word_dict is None:
            self.word_dict = build_vocab(self.df, vocabulary_size)
            print(f"Created vocabulary of size: {len(self.word_dict)}")
        else:
            self.word_dict = word_dict
            print(f"Using provided vocabulary with size: {len(self.word_dict)}")

        for i in range(self.length):
            review_words = self.df["text"][i].split()
            # Start with the 'start of sentence' token in each review
            indices = [self.word_dict.get("SOS", 1)]
            for word in review_words:
                idx = self.word_dict.get(word)
                if idx is not None:
                    indices.append(idx)

            label = self.df["label"][i]
            self.texts.append(torch.tensor(indices[:max_len], dtype=int))
            self.labels.append(label)

    def preprocess_text(self, s):
        s = s.lower().strip()
        s = re.sub(r"<br />", r" ", s)
        s = re.sub(r"(\W)(?=\1)", "", s)
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z0-9]+", r" ", s)
        return s.strip()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def yelp_collate(batch):
    texts, labels = zip(*batch)
    text_pad = pad_sequence(texts, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return text_pad, labels


def build_vocab(train_df, vocabulary_size):
    word_count = {}
    for review in train_df["text"]:
        for word in review.split():
            word_count[word] = word_count.get(word, 0) + 1

    # Include 'PAD' and 'SOS' tokens
    word_count = dict(
        [("PAD", 1), ("SOS", 1)]
        + sorted(word_count.items(), key=lambda x: x[1], reverse=True)[
            : vocabulary_size - 2
        ]
    )

    word_dict = {word: i for i, word in enumerate(word_count)}
    return word_dict


def get_dataloader(vocabulary_size, max_review_length, batch_size):
    train_dataset = Yelp_Dataset(
        "train", vocabulary_size=vocabulary_size, max_len=max_review_length
    )
    word_dict = train_dataset.word_dict
    test_dataset = Yelp_Dataset(
        "test",
        vocabulary_size=vocabulary_size,
        max_len=max_review_length,
        word_dict=word_dict,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=yelp_collate,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=yelp_collate,
        drop_last=True,
    )

    return train_loader, test_loader, word_dict
