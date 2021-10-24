from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from config import vars
from zreader.utils.data import generate_subsequent_mask


class Collate(object):
    def __init__(self, min_noise: int, max_noise: int) -> None:
        assert 0 <= min_noise <= len(vars.ALPHABET), \
            f'min_noise should be between 0 and {len(vars.ALPHABET)} inclusive'
        assert min_noise <= max_noise <= len(vars.ALPHABET), \
            f'max_noise should be between {min_noise} and {len(vars.ALPHABET)} inclusive'

        self.min_noise = min_noise
        self.max_noise = max_noise

    def __str__(self) -> str:
        return f'<Collate(min_noise={self.min_noise}, max_noise={self.max_noise})>'

    def __call__(self, batch: List[str]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = [list(entry) for entry in batch]
        sizes = [len(entry) for entry in batch]
        batch_size, seq_len, token_size = len(batch), max(sizes), len(vars.ALPHABET)

        src = torch.zeros((batch_size, seq_len, token_size), dtype=torch.float)
        tgt_inp = torch.zeros((batch_size, seq_len, token_size), dtype=torch.float)
        tgt = list()
        padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)  # TODO fix bag

        for i in range(len(batch)):

            i_tgt = torch.full((seq_len,), fill_value=-1, dtype=torch.long)

            for j in range(len(batch[i])):
                num_repr = vars.ALPHABET2NUM[batch[i][j]]

                src[i, j, num_repr] = 1
                tgt_inp[i, j, num_repr] = 1
                i_tgt[j] = num_repr

                noise_size = np.random.randint(low=self.min_noise, high=self.max_noise + 1, size=1)[0]
                noise_indexes = np.random.randint(low=0, high=len(vars.ALPHABET), size=noise_size)

                src[i, j, noise_indexes] = 1

            tgt.append(i_tgt)
            padding_mask[i, sizes[i]:] = True

        empty_token = torch.zeros(batch_size, 1, token_size)
        tgt_inp = torch.cat([empty_token, tgt_inp[:, :-1, :]], dim=1)
        tgt = torch.stack(tgt)
        subsequent_mask = generate_subsequent_mask(seq_len)

        return src, tgt_inp, tgt, padding_mask, padding_mask, subsequent_mask


class WikiDataset(Dataset):

    def __init__(self, datafiles: List[Path], min_threshold: int, max_threshold: int, dataset_size: int) -> None:
        assert self.__exists(datafiles), 'datafiles do not exist'
        assert min_threshold > 0, 'min_threshold should be grater than 0'
        assert max_threshold >= min_threshold, f'max_threshold should be grater or equal than {min_threshold}'
        assert dataset_size > 0, 'dataset_size should be grater than 0'

        self.datafiles = datafiles
        self.n_files = range(len(self.datafiles))
        self.file_sizes = [file.stat().st_size for file in self.datafiles]
        self.distribution = self.__get_distribution()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.dataset_size = dataset_size

    def __getitem__(self, idx: int) -> str:
        np.random.seed(None)

        file_id = np.random.choice(self.n_files, p=self.distribution)
        shift = np.random.randint(low=0, high=self.file_sizes[file_id] - self.max_threshold + 1, size=1)[0]
        line_size = np.random.randint(low=self.min_threshold, high=self.max_threshold + 1, size=1)[0]

        with open(self.datafiles[file_id], mode="r") as f:
            f.seek(shift)

            return f.read(line_size)

    def __len__(self) -> int:
        return self.dataset_size

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'WikiDataset(min_threshold={self.min_threshold}, max_threshold={self.max_threshold},' \
               f' dataset_size={self.dataset_size})'

    @staticmethod
    def __exists(datafiles: List[Path]) -> bool:
        for file in datafiles:
            if not file.exists():
                print(f'{file} doesnt exist')
                return False

        return True if len(datafiles) else False

    def __get_distribution(self) -> np.ndarray:
        powered = np.array(self.file_sizes)

        return powered / np.sum(powered)

    def create_dataloader(self, batch_size: int,
                          min_noise: int,
                          max_noise: int,
                          num_workers: int = 0,
                          pin_memory: bool = True
                          ) -> DataLoader:

        assert batch_size > 0, 'batch_size should be grater than 0'
        assert num_workers >= 0, 'num_workers should be grater or equal than 0'

        return DataLoader(dataset=self, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          pin_memory=pin_memory, collate_fn=Collate(min_noise=min_noise, max_noise=max_noise))