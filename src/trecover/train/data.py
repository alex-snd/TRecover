import os
import platform
from http.client import HTTPException
# from multiprocessing import Value
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from trecover.config import var, log


class BaseCollate(object):
    def __init__(self, min_noise: int, max_noise: int, device: Optional[torch.device] = None):
        assert 0 <= min_noise <= len(var.ALPHABET), \
            f'min_noise should be between 0 and {len(var.ALPHABET)} inclusive'
        assert min_noise <= max_noise <= len(var.ALPHABET), \
            f'max_noise should be between {min_noise} and {len(var.ALPHABET)} inclusive'

        self.min_noise = min_noise
        self.max_noise = max_noise
        # self._min_noise = Value('i', min_noise)
        # self._max_noise = Value('i', max_noise)
        self.device = device or torch.device("cpu")

    def __str__(self) -> str:
        return f'<Collate(min_noise={self.min_noise}, max_noise={self.max_noise})>'

    def __call__(self, batch: List[str]) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor]:
        raise NotImplementedError

    # @property
    # def min_noise(self) -> int:
    #     # return self._min_noise.value
    #     # log.project_console.print(f'Trying to get min_noise', justify='center')  # TODO
    #     with self._min_noise.get_lock():
    #         # log.project_console.print(f'Get min_noise end', justify='center')  # TODO
    #         return self._min_noise.value
    #
    # @min_noise.setter
    # def min_noise(self, value: int) -> None:
    #     with self._min_noise.get_lock():
    #         self._min_noise.value = value
    #
    # @property
    # def max_noise(self) -> int:
    #     # return self._max_noise.value
    #     # log.project_console.print(f'Trying to get max_noise', justify='center')  # TODO
    #     with self._max_noise.get_lock():
    #         # log.project_console.print(f'Get max_noise end', justify='center')  # TODO
    #         return self._max_noise.value
    #
    # @max_noise.setter
    # def max_noise(self, value: int) -> None:
    #     with self._max_noise.get_lock():
    #         self._max_noise.value = value

    def sync(self, verbose: bool = False) -> None:
        if verbose:
            log.project_console.print('BaseCollate:Unable to synchronize CollabCollate arguments', style='yellow',
                                      justify='right')  # TODO

    def generate_subsequent_mask(self, size: int) -> Tensor:
        return torch.triu(torch.ones((size, size), dtype=torch.float, device=self.device), diagonal=1) == 1


class StandardCollate(BaseCollate):
    def __init__(self, min_noise: int, max_noise: int, device: Optional[torch.device] = None):
        super(StandardCollate, self).__init__(min_noise=min_noise, max_noise=max_noise, device=device)

    def __call__(self, batch: List[str]) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor]:
        log.project_console.print(
            f'Batch generation with min_noise={self.min_noise}, max_noise={self.max_noise}, PID: {os.getpid()}',
            justify='center')  # TODO

        batch = [list(entry) for entry in batch]
        sizes = [len(entry) for entry in batch]
        batch_size, seq_len, token_size = len(batch), max(sizes), len(var.ALPHABET)

        src = torch.zeros((batch_size, seq_len, token_size), dtype=torch.float, device=self.device)
        tgt_inp = torch.zeros((batch_size, seq_len, token_size), dtype=torch.float, device=self.device)
        tgt = list()

        for i in range(len(batch)):

            i_tgt = torch.full((seq_len,), fill_value=-1, dtype=torch.long, device=self.device)

            for j in range(len(batch[i])):
                num_repr = var.ALPHABET2NUM[batch[i][j]]

                src[i, j, num_repr] = 1
                tgt_inp[i, j, num_repr] = 1
                i_tgt[j] = num_repr

                noise_size = np.random.randint(low=self.min_noise, high=self.max_noise + 1, size=1)[0]
                noise_indexes = np.random.randint(low=0, high=len(var.ALPHABET), size=noise_size)

                src[i, j, noise_indexes] = 1

            tgt.append(i_tgt)

        empty_token = torch.zeros((batch_size, 1, token_size), device=self.device)
        tgt_inp = torch.cat([empty_token, tgt_inp[:, :-1, :]], dim=1)
        tgt = torch.stack(tgt)
        subsequent_mask = self.generate_subsequent_mask(seq_len)

        if min(sizes) == seq_len:
            src_pad_mask = None
            tgt_inp_pad_mask = None

        else:
            src_pad_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=self.device)

            for i in range(len(batch)):
                src_pad_mask[i, sizes[i]:] = True

            empty_token_pad_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device)
            tgt_inp_pad_mask = torch.cat([empty_token_pad_mask, src_pad_mask[:, :-1]], dim=1)

        log.project_console.print(f'Batch generation end, PID: {os.getpid()}', justify='center')  # TODO

        return src, tgt_inp, tgt, src_pad_mask, tgt_inp_pad_mask, subsequent_mask


class CollabCollate(StandardCollate):
    def __init__(self, device: Optional[torch.device] = None):
        super(CollabCollate, self).__init__(min_noise=0, max_noise=0, device=device)
        # self.sync(verbose=False)  # TODO

    def sync(self, verbose: bool = False) -> None:
        try:
            remote_args: Dict = torch.hub.load('alex-snd/TRecover', 'collab_args', force_reload=True, verbose=False)
            min_noise = remote_args.get('min_noise', -1)
            max_noise = remote_args.get('max_noise', -1)

            assert 0 <= min_noise <= max_noise <= len(var.ALPHABET), 'Bad arguments'

            self.min_noise = 3  # min_noise   # TODO
            self.max_noise = 8  # max_noise

        except (HTTPException, AssertionError) as e:
            if verbose:
                log.project_console.print('CollabCollate: Unable to synchronize CollabCollate arguments -',
                                          style='yellow', justify='right')  # TODO
                log.project_console.print(e, style='yellow', justify='right')
        else:
            if verbose:
                log.project_console.print('CollabCollate arguments are synchronized', style='salmon1', justify='right')


class WikiDataset(Dataset):
    def __init__(self, datafiles: List[Path], min_threshold: int, max_threshold: int, dataset_size: int):
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
                          collate: Optional[BaseCollate] = StandardCollate(min_noise=0, max_noise=0),
                          num_workers: int = 0,
                          pin_memory: bool = True
                          ) -> DataLoader:
        assert batch_size > 0, 'batch_size should be grater than 0'
        assert num_workers >= 0, 'num_workers should be grater or equal than 0'

        if collate.device.type == 'cuda':
            if platform.system() == 'Windows' and num_workers > 0:
                log.project_console.print('WARNING: Dataloader does not support num_workers > 0 and GPU device '
                                          'on Windows. The training data will be transferred to the GPU just '
                                          'before it is fed into the model, but not during batch generation.',
                                          style='bold red')
                collate.device = torch.device('cpu')

            pin_memory = False

        return DataLoader(dataset=self,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          collate_fn=collate)
