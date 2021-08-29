from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import config


class Collate(object):
    alphabet_to_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10,
                       'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'v': 20,
                       'u': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}

    num_to_alphabet = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k',
                       11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'v',
                       21: 'u', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}

    def __init__(self, min_noise: int, max_noise: int) -> None:
        assert 0 <= min_noise <= len(self.alphabet_to_num), \
            f'min_noise should be between 0 and {len(self.alphabet_to_num)} inclusive'
        assert min_noise <= max_noise <= len(self.alphabet_to_num), \
            f'max_noise should be between {min_noise} and {len(self.alphabet_to_num)} inclusive'

        self.min_noise = min_noise
        self.max_noise = max_noise

    def __str__(self) -> str:
        return f'<Collate(min_noise={self.min_noise}, max_noise={self.max_noise})>'

    def __call__(self, batch: List[str]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = [list(entry) for entry in batch]
        sizes = [len(entry) for entry in batch]
        batch_size, seq_len, token_size = len(batch), max(sizes), len(self.alphabet_to_num)

        src = torch.zeros((batch_size, seq_len, token_size), dtype=torch.float)
        tgt_inp = torch.zeros((batch_size, seq_len, token_size), dtype=torch.float)
        tgt = list()
        padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)  # TODO fix bag

        for i in range(len(batch)):

            i_tgt = torch.full((seq_len,), fill_value=-1, dtype=torch.long)

            for j in range(len(batch[i])):
                num_repr = self.alphabet_to_num[batch[i][j]]

                src[i, j, num_repr] = 1
                tgt_inp[i, j, num_repr] = 1
                i_tgt[j] = num_repr

                noise_size = np.random.randint(low=self.min_noise, high=self.max_noise + 1, size=1)[0]
                noise_indexes = np.random.randint(low=0, high=len(self.alphabet_to_num), size=noise_size)

                src[i, j, noise_indexes] = 1

            tgt.append(i_tgt)
            padding_mask[i, sizes[i]:] = True

        empty_token = torch.zeros(batch_size, 1, token_size)
        tgt_inp = torch.cat([empty_token, tgt_inp[:, :-1, :]], dim=1)
        tgt = torch.stack(tgt)
        subsequent_mask = self.get_subsequent_mask(seq_len)

        return src, tgt_inp, tgt, padding_mask, padding_mask, subsequent_mask

    @staticmethod
    def get_subsequent_mask(size: int) -> Tensor:
        return torch.triu(torch.ones((size, size), dtype=torch.float), diagonal=1) == 1


class WikiDataset(Dataset):

    def __init__(self, datafiles: List[Path], min_threshold: int, max_threshold: int, dataset_size: int) -> None:
        assert self.__exists(datafiles)
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
        return f'<WikiDataset(min_threshold={self.min_threshold}, max_threshold={self.max_threshold},' \
               f' dataset_size={self.dataset_size})>'

    @staticmethod
    def __exists(datafiles: List[Path]) -> bool:
        for file in datafiles:
            if not file.exists():
                print(f'{file} doesnt exist')
                return False

        return True

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


if __name__ == '__main__':
    train_files = [Path(config.TRAIN_DATA, file) for file in config.TRAIN_DATA.iterdir()]
    dataset = WikiDataset(train_files, min_threshold=200, max_threshold=200, dataset_size=5)

    loader = dataset.create_dataloader(batch_size=3, min_noise=1, max_noise=8)

    for _src, _tgt_inp, _tgt, _src_pad_mask, _tgt_inp_pad_mask, _subsequent_mask in loader:
        print(f'| src: {_src.size()} '
              f'| tgt_inp: {_tgt_inp.size()} '
              f'| tgt: {_tgt.size()} '
              f'| src_pad_mask: {_src_pad_mask.size()} '
              f'| tgt_inp_pad_mask: {_tgt_inp_pad_mask.size()} '
              f'| tgt_inp_attn_mask: {_subsequent_mask.size()}')
