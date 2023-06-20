import os, math
import pandas as pd
import torch as ch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from constants import SEPARATOR, DELETED_CHARS, REPLACED_CHARS, WRONG_ORDS


class BaseDataModule(pl.LightningDataModule):

    multilabel = None

    filters = (("text", "м"), ("text", "и"), ("text", "م"), ("text", "و"), ("text", "ج"), ("text", "VOTE TRUMP"),
        ("text", "点"), ("text", "击"), ("text", "查"), ("text", "看"), ("text", "σ"), ("text", "τ"), ("text", "ε"),
        ("text", "ρ"), ("text", "ن"), ("text", "спa"), ("text", "л"), ("text", "н"), ("text", "я"), ("text", "б"),
        ("text", "п"))

    def __init__(self, batch_size, seed: int = 0, train_ratio: float = 0.6, val_ratio: float = 0.2):
        super().__init__()
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

    def setup(self, stage: str = None):
        data = pd.concat([pd.read_csv(os.path.join(self.data_dir, filename), on_bad_lines="error", **self.load_args) for filename in self.filenames], ignore_index=True)
        data = self.filter_data(data, self.filters)
        data = data.reset_index(drop=True)

        full_ds = self.extract_dataset_from_df(data)
        del data

        num_train = math.ceil(len(full_ds) * self.train_ratio)
        num_val = math.ceil(len(full_ds) * self.val_ratio)
        num_test = len(full_ds) - num_train - num_val


        tmp_train_ds, tmp_val_ds, tmp_test_ds = random_split(full_ds, [num_train, num_val, num_test], generator=ch.Generator().manual_seed(self.seed))
        del full_ds

        self.train_ds, self.val_ds, self.test_ds = tmp_train_ds, tmp_val_ds, tmp_test_ds

        self.calculate_pos_weight()

    def calculate_pos_weight(self):
        labels = ch.tensor([s[1] for s in self.train_ds])
        label_freqs = ch.nn.functional.one_hot(labels, num_classes=len(set(self.class2idx.values()))).sum(0)

        label_neg_freqs = len(self.train_ds) - label_freqs

        self.pos_weights = label_neg_freqs / label_freqs

    def train_dataloader(self):

        num_workers = int(int(os.getenv("LSB_DJOB_NUMPROC", default=ch.get_num_threads())) / (ch.distributed.get_world_size() if ch.distributed.is_initialized() else 1))

        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=num_workers, shuffle=False, drop_last=True, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):

        num_workers = int(int(os.getenv("LSB_DJOB_NUMPROC", default=ch.get_num_threads())) / (ch.distributed.get_world_size() if ch.distributed.is_initialized() else 1))

        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=num_workers, shuffle=False, drop_last=True, generator=ch.Generator().manual_seed(self.seed) if not ch.distributed.is_initialized() else None,
                          persistent_workers=True, pin_memory=True)

    def test_dataloader(self):

        num_workers = int(int(os.getenv("LSB_DJOB_NUMPROC", default=ch.get_num_threads())) / (ch.distributed.get_world_size() if ch.distributed.is_initialized() else 1))

        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=num_workers, shuffle=True, drop_last=True, generator=ch.Generator().manual_seed(self.seed) if not ch.distributed.is_initialized() else None,
                          persistent_workers=True, pin_memory=True)

    def predict_dataloader(self):
        return None

    @staticmethod
    def filter_data(df, filters=None):
        if filters is None:
            filters = []
        for field, value in filters:
            if field in df:
                df = df[~df[field].str.contains(value)]
        return df

    @staticmethod
    def preprocessing(_str, lowercase=True):
        if lowercase:
            # Lower-case all text
            _str = _str.lower()
        # Remove some chars
        for char in DELETED_CHARS:
            _str = _str.replace(char, "")

        # Remove weird chars
        _str = "".join([c for c in _str if ord(c) not in WRONG_ORDS])

        # Replace some chars
        for char, repl_char in REPLACED_CHARS:
            _str = _str.replace(char, repl_char)

        # Final removal of double whitespaces
        while "  " in _str:
            _str = _str.replace("  ", " ")
        # Remove leading and trailing whitespaces
        _str = _str.strip()
        # Final removal of separator
        _str = _str.replace(SEPARATOR, "")

        return _str

    @property
    def num_train_samples(self):
        return len(self.train_ds)

    @property
    def num_val_samples(self):
        return len(self.val_ds)

    @property
    def num_test_samples(self):
        return len(self.test_ds)

    @property
    def num_total_samples(self):
        return len(self.train_ds) + len(self.val_ds) + len(self.test_ds)

