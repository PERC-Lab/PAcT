"""Module to represent the datasets"""

import math
import random
from abc import ABC, abstractmethod
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab

from tqdm import tqdm

from nltk import word_tokenize

from gensim.models import FastText


def get_end_idx(num_samples, ratio):
    return math.floor(num_samples * ratio)


class AbstractDataset(Dataset):

    def __init__(self):
        self.samples = []

    def get_raw_sample(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class AbstractCodeDataset(AbstractDataset, ABC):
    """Abstract class to represent Code dataset"""

    def __init__(self, split_ratio=(0.8, 0.1, 0.1), random=False) -> None:
        super().__init__()
        self.split_ratio = split_ratio

    def _split_samples(self):
        """Find and store the end index of
        train, valid, and test sets"""

        train_end_idx = get_end_idx(self.split_ratio[0], len(self.samples))
        val_end_idx = get_end_idx((self.split_ratio[0] + self.split_ratio[1]),
                                  len(self.samples))

        self.split_indices = {
            'train': (0, train_end_idx),
            'valid': (train_end_idx, val_end_idx),
            'test': (val_end_idx, len(self.samples))
        }

    def get_set(self, split_name: str):
        """Returns a map-style Dataset object containing samples
        of split_name ('train', 'valid', or 'test').

        Parameters
        ----------
        split_name : str
            Name of set. 'train' | 'valid' | 'test'

        Returns
        -------
        _RawCodeIterableDataset
            A wrapper dataset object
        """

        start_idx, end_idx = self.split_indices[split_name]
        dataset_samples = self.samples[start_idx:end_idx]
        return _RawCodeIterableDataset(dataset_samples,
                                       self.vocab,
                                       self.length,
                                       self.__getitem__)

    @abstractmethod
    def _init_dataset(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


class _RawCodeIterableDataset(AbstractDataset, Dataset):
    def __init__(self, samples, vocab, length, getter_fn):
        self.samples = samples
        self.vocab = vocab
        self.length = length
        self.getter_fn = getter_fn

    def __getitem__(self, index):
        return self.getter_fn(index)



class ASTSeqDataset(AbstractCodeDataset):

    def __init__(self,
                 filename,
                 split_ratio,
                 length,
                 labels,
                 randomize=False,
                 delimiter="###") -> None:
        super().__init__(split_ratio, randomize)
        self.filename = filename
        self.length = length
        self.labelset = labels
        self.randomize_samples = randomize
        self.vocab = None
        self.num_classes = len(set(labels.values()))
        self._init_dataset(delimiter)

    def _init_dataset(self, delimiter):
        counter = Counter()

        with open(self.filename, 'r') as f:
            data = f.readlines()

        status_bar = tqdm(range(len(data)),
                          desc='Samples',
                          ascii=' >>>>>>>>>=',
                          bar_format='{desc}: {percentage:3.0f}% [{bar}]'
                                     ' {n_fmt}/{total_fmt} {desc}',
                          ncols=65,
                          colour='green')

        for i, el in enumerate(data):
            line = el.split(delimiter)

            sample_id = line[0]
            ast_paths = line[1][:-1]  # Removing the additional space
            counter.update(word_tokenize(ast_paths))

            paths = ast_paths.split(',')

            if paths[0] == '':
                continue

            if len(paths) < self.length:
                padding = ['\0, \0, \0' for f in range(
                    self.length - len(paths))]
                paths.extend(padding)
            elif len(paths) >= self.length:
                paths = paths[:self.length-1]
                paths.extend(['\0, \0, \0'])

            label = line[-1].strip()
            self.samples.append((sample_id, paths, label))

            status_bar.update(1)

        self.vocab = Vocab(counter)

        if self.randomize_samples:
            random.shuffle(self.samples)

        self._split_samples()

    def __getitem__(self, index):
        _, src_code, label = self.samples[index]

        src_code_vocab = [[self.vocab[tok] for tok in path.split(' ')[:3]]
                          for path in src_code]
        src_code_tensor = torch.tensor(src_code_vocab, dtype=torch.int64)

        label = self.labelset[label]
        return (src_code_tensor, label)
