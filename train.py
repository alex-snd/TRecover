import math
from datetime import datetime
from os import listdir, mkdir
from os.path import join, exists
from time import time

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import config
from data import WikiDataset, Collate
from scheduler import Scheduler, IdentityScheduler
from utils import set_seeds, get_model, visualize_columns, visualize_target


class Trainer(object):
    def __init__(self, model: torch.nn.Module, criterion: eval, optimizer, working_dir: str, scheduler=None,
                 device: torch.device = None, verbose: bool = True, log_interval: int = 1, console_width: int = None,
                 delimiter: str = ''):
        self.model = model
        self.criterion = criterion  # should return average on the batch
        self.optimizer = optimizer
        self.scheduler = scheduler or IdentityScheduler()
        self.device = device or torch.device("cpu")
        self.verbose = verbose
        self.log_interval = log_interval
        self.console_width = console_width
        self.delimiter = delimiter

        self.train_loss = 0.0
        self.val_loss = 0.0

        if self.console_width:
            self.n_columns_to_show = math.ceil(self.console_width / max(2 * len(delimiter), 1)) - len(delimiter)
        else:
            self.n_columns_to_show = None

        date = datetime.now()
        self.experiment_mark = f'{date.month:0>2}{date.day:0>2}_{date.hour:0>2}{date.minute:0>2}'

        self.experiment_folder = join(working_dir, self.experiment_mark)
        self.weights_folder = join(self.experiment_folder, 'weights')

        if not exists(self.experiment_folder):
            mkdir(self.experiment_folder)
        if not exists(self.weights_folder):
            mkdir(self.weights_folder)

        self.__log_file = open(join(self.experiment_folder, f'{self.experiment_mark}.log'), mode='w')

        self.__log_init_params()

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def save_parameters(self, weights_name: str) -> None:
        self.model.save_parameters(filename=join(self.weights_folder, f'{self.experiment_mark}_{weights_name}'))

    def __train_step(self, offset: int, train_loader: DataLoader, accumulation_step: int = 1) -> None:
        self.model.train()

        self.__log('*' * self.console_width)

        train_loss = 0.0

        self.optimizer.zero_grad()
        start_time = time()

        for batch_idx, (src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask) in enumerate(train_loader,
                                                                                                   start=1):
            src = src.to(self.device)
            tgt_inp = tgt_inp.to(self.device)
            tgt = tgt.to(self.device)
            src_pad_mask = src_pad_mask.to(self.device)
            tgt_pad_mask = tgt_pad_mask.to(self.device)
            tgt_attn_mask = tgt_attn_mask.to(self.device)

            tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
            tgt_out = tgt_out.reshape(-1, self.model.token_size)
            tgt = tgt.view(-1)

            loss = self.criterion(tgt_out, tgt)
            loss.backward()

            train_loss += loss.item()

            if batch_idx % accumulation_step == 0:
                self.scheduler.step()
                self.optimizer.step()

                self.optimizer.zero_grad()

                if self.verbose and (offset + batch_idx) % self.log_interval == 0:
                    accuracy = (torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)

                    self.__log(f'Train Batch:  {offset + batch_idx:^7} | '
                               f'Loss: {train_loss / accumulation_step:>10.6f} | Accuracy: {accuracy:>6.3f} | '
                               f'Elapsed: {time() - start_time:>7.3f}s | LR {round(self.lr, 6):>8}')

                    start_time = time()

    def __val_step(self, offset: int, val_loader: DataLoader) -> None:
        self.model.eval()

        self.__log('-' * self.console_width)

        test_loss = 0
        test_accuracy = 0
        start_time = time()

        with torch.no_grad():
            for batch_idx, (src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask) in enumerate(val_loader,
                                                                                                       start=1):
                src = src.to(self.device)
                tgt_inp = tgt_inp.to(self.device)
                tgt = tgt.to(self.device)
                src_pad_mask = src_pad_mask.to(self.device)
                tgt_pad_mask = tgt_pad_mask.to(self.device)
                tgt_attn_mask = tgt_attn_mask.to(self.device)

                tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
                tgt_out = tgt_out.reshape(-1, self.model.token_size)
                tgt = tgt.view(-1)

                loss = self.criterion(tgt_out, tgt).item()
                accuracy = (torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)

                test_loss += loss
                test_accuracy += accuracy

                if self.verbose:
                    self.__log(f'Test Batch:   {offset + batch_idx:^7} | Loss: {loss:>10.6f} | '
                               f'Accuracy: {accuracy:>6.3f} | Elapsed: {time() - start_time:>7.3f}s')

                    start_time = time()

            test_loss /= batch_idx
            test_accuracy /= batch_idx

            self.__log(f'Test Average:         | Loss: {test_loss:>10.6f} | Accuracy: {test_accuracy:>6.3f} |')

    def __vis_step(self, vis_loader: DataLoader) -> None:
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask) in enumerate(vis_loader,
                                                                                                       start=1):
                src = src.to(self.device)
                tgt_inp = tgt_inp.to(self.device)
                tgt = tgt.to(self.device)
                src_pad_mask = src_pad_mask.to(self.device)
                tgt_pad_mask = tgt_pad_mask.to(self.device)
                tgt_attn_mask = tgt_attn_mask.to(self.device)

                tgt_out = self.model(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
                tgt_out = tgt_out.reshape(-1, self.model.token_size)
                prediction = torch.argmax(tgt_out, dim=1).view_as(tgt)

                for i in range(src.size(0)):
                    self.__log('-' * self.console_width)
                    self.__log(visualize_columns(src[i, : self.n_columns_to_show], delimiter=self.delimiter))
                    self.__log('-' * self.console_width)
                    self.__log(visualize_target(prediction[i, : self.n_columns_to_show], delimiter=self.delimiter))
                    self.__log('-' * self.console_width)
                    self.__log(visualize_target(tgt[i, : self.n_columns_to_show], delimiter=self.delimiter))
                    self.__log('\n')

    def train(self, n_epochs: int, train_loader: DataLoader, val_loader: DataLoader, vis_loader: DataLoader,
              epoch_seek: int = 0, accumulation_step: int = 1, vis_interval: int = 1, saving_interval: int = 1):

        self.__log(f'Batch size: {train_loader.batch_size}')
        self.__log(f'Accumulation step: {accumulation_step}')

        if len(train_loader) % accumulation_step != 0:
            self.__log('WARNING: Train dataset size must be evenly divisible by batch_size * accumulation_step')

        try:
            for epoch_idx in range(epoch_seek + 1, epoch_seek + n_epochs + 1):
                offset = len(train_loader) * (epoch_idx - 1)

                self.__train_step(offset, train_loader, accumulation_step)

                self.__val_step(offset, val_loader)

                if epoch_idx % vis_interval == 0:
                    self.__vis_step(vis_loader)

                if epoch_idx % saving_interval == 0:
                    self.save_parameters(str(offset + len(train_loader)))

        except KeyboardInterrupt:
            print('Interrupted')

        finally:
            self.save_parameters('last_saving')

            self.close()

    def close(self) -> None:
        self.__log_file.close()

    def __log_init_params(self) -> None:
        self.__log(f'Date: {self.experiment_mark}')
        self.__log(f'Model: {self.model}')
        self.__log(f'Optimizer: {self.optimizer}'.replace('\n', ''))
        self.__log(f'Scheduler: {self.scheduler}')

    def __log(self, info: str) -> None:
        self.__log_file.write(f'{info}\n')

        if self.verbose:
            print(info)


def train() -> None:
    set_seeds(seed=2531)

    # ---------------------------------------------DATA PARAMETERS------------------------------------------------------
    train_files = [join(config.train_path, file) for file in listdir(config.train_path)]
    val_files = [join(config.val_path, file) for file in listdir(config.val_path)]
    vis_files = [join(config.vis_path, file) for file in listdir(config.vis_path)]
    min_threshold = 256
    max_threshold = 256
    train_dataset_size = 50
    test_dataset_size = 50
    vis_dataset_size = 5
    num_workers = 3
    min_noise = 0
    max_noise = 0
    # --------------------------------------------MODEL PARAMETERS------------------------------------------------------
    token_size = len(Collate.alphabet_to_num)
    pe_max_len = 1000
    num_layers = 6
    d_model = 512  # d_model % n_heads = 0
    n_heads = 16
    d_ff = 2048
    dropout = 0.1
    weights_name = ''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -----------------------------------------OPTIMIZATION PARAMETERS--------------------------------------------------
    criterion = CrossEntropyLoss(ignore_index=-1)
    lr = 0  # fictive with Scheduler
    step_seek = 0
    warmup = 4_000
    lr_step_size = 1
    # ------------------------------------------TRAIN LOOP PARAMETERS---------------------------------------------------
    n_epochs = 300
    epoch_seek = 0
    batch_size = 5
    accumulation_step = 1  # train_dataset_size % (batch_size * accumulation_step) == 0
    saving_interval = 1
    log_interval = 1
    vis_interval = 1
    verbose = True
    console_width = 94
    delimiter = ''
    # ------------------------------------------------------------------------------------------------------------------

    train_dataset = WikiDataset(datafiles=train_files, min_threshold=min_threshold, max_threshold=max_threshold,
                                dataset_size=train_dataset_size)
    val_dataset = WikiDataset(datafiles=val_files, min_threshold=min_threshold, max_threshold=max_threshold,
                              dataset_size=test_dataset_size)
    vis_dataset = WikiDataset(datafiles=vis_files, min_threshold=min_threshold, max_threshold=max_threshold,
                              dataset_size=vis_dataset_size)

    train_loader = train_dataset.create_dataloader(batch_size=batch_size, min_noise=min_noise, max_noise=max_noise,
                                                   num_workers=num_workers)
    val_loader = val_dataset.create_dataloader(batch_size=batch_size, min_noise=min_noise, max_noise=max_noise,
                                               num_workers=num_workers)
    vis_loader = vis_dataset.create_dataloader(batch_size=batch_size, min_noise=min_noise, max_noise=max_noise,
                                               num_workers=num_workers)

    z_reader = get_model(token_size, pe_max_len, num_layers, d_model, n_heads, d_ff, dropout, device,
                         weights=join(config.weights_path, weights_name))

    optimizer = torch.optim.Adam(z_reader.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    scheduler = Scheduler(optimizer, d_model, warmup, lr_step_size, seek=step_seek)

    trainer = Trainer(model=z_reader, criterion=criterion, optimizer=optimizer, working_dir=config.experiments_path,
                      scheduler=scheduler, device=device, verbose=verbose, log_interval=log_interval,
                      console_width=console_width, delimiter=delimiter)

    trainer.train(n_epochs, train_loader, val_loader, vis_loader, epoch_seek, accumulation_step,
                  vis_interval, saving_interval)


if __name__ == '__main__':
    train()
