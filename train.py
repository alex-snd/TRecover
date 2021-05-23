import math
from os import listdir
from os.path import join
from time import time

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import config
from data import WikiDataset, Collate
from model import ZReader
from scheduler import Scheduler
from utils import get_model, save_parameters, visualize_columns, visualize_target


def epoch_training(offset: int, z_reader: ZReader, train_loader: DataLoader, optimizer, scheduler: Scheduler,
                   accumulation_step: int, device: torch.device, log_interval: int, criterion: eval, console_width: int,
                   printable: bool = True) -> None:
    z_reader.train()

    if printable:
        print('*' * console_width)

    optimizer.zero_grad()
    start_time = time()

    for batch_idx, (src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask) in enumerate(train_loader, start=1):
        src = src.to(device)
        tgt_inp = tgt_inp.to(device)
        tgt = tgt.to(device)
        src_pad_mask = src_pad_mask.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)
        tgt_attn_mask = tgt_attn_mask.to(device)

        tgt_out = z_reader(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
        tgt_out = tgt_out.reshape(-1, z_reader.token_size)
        tgt = tgt.view(-1)

        loss = criterion(tgt_out, tgt)
        loss.backward()

        if batch_idx % accumulation_step == 0:
            scheduler.step()
            optimizer.step()

            optimizer.zero_grad()

            if printable and (offset + batch_idx) % log_interval == 0:
                accuracy = (torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)

                print(f'Train Batch:  {offset + batch_idx:^7} | Loss: {loss.item():>10.6f} | Accuracy: {accuracy:>6.3f}'
                      f' | Elapsed: {time() - start_time:>7.3f}s | LR {round(scheduler.lr, 6):>8}')

                start_time = time()


def epoch_testing(offset: int, z_reader: ZReader, test_loader: DataLoader, device: torch.device, criterion: eval,
                  console_width: int, printable: bool = True) -> None:
    z_reader.eval()

    if printable:
        print('-' * console_width)

    test_loss = 0
    test_accuracy = 0
    start_time = time()

    with torch.no_grad():
        for batch_idx, (src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask) in enumerate(test_loader,
                                                                                                   start=1):
            src = src.to(device)
            tgt_inp = tgt_inp.to(device)
            tgt = tgt.to(device)
            src_pad_mask = src_pad_mask.to(device)
            tgt_pad_mask = tgt_pad_mask.to(device)
            tgt_attn_mask = tgt_attn_mask.to(device)

            tgt_out = z_reader(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
            tgt_out = tgt_out.reshape(-1, z_reader.token_size)
            tgt = tgt.view(-1)

            loss = criterion(tgt_out, tgt).item()
            accuracy = (torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)

            test_loss += loss
            test_accuracy += accuracy

            if printable:
                print(f'Test Batch:   {offset + batch_idx:^7} | Loss: {loss:>10.6f} | Accuracy: {accuracy:>6.3f} | '
                      f'Elapsed: {time() - start_time:>7.3f}s')

                start_time = time()

        test_loss /= batch_idx
        test_accuracy /= batch_idx

        print(f'Test Average: {"":^7} | Loss: {test_loss:>10.6f} | Accuracy: {test_accuracy:>6.3f} |')


def epoch_visualization(z_reader: ZReader, vis_loader: DataLoader, device: torch.device, console_width: int) -> None:
    z_reader.eval()

    n_columns_to_show = math.ceil(console_width / 2) - 1

    with torch.no_grad():
        for batch_idx, (src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask) in enumerate(vis_loader,
                                                                                                   start=1):
            src = src.to(device)
            tgt_inp = tgt_inp.to(device)
            tgt = tgt.to(device)
            src_pad_mask = src_pad_mask.to(device)
            tgt_pad_mask = tgt_pad_mask.to(device)
            tgt_attn_mask = tgt_attn_mask.to(device)

            tgt_out = z_reader(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
            tgt_out = tgt_out.reshape(-1, z_reader.token_size)
            prediction = torch.argmax(tgt_out, dim=1).view_as(tgt)

            for i in range(src.size(0)):
                print('-' * console_width)
                visualize_columns(src[i, : n_columns_to_show])
                print('-' * console_width)
                visualize_target(prediction[i, : n_columns_to_show])
                print('-' * console_width)
                visualize_target(tgt[i, : n_columns_to_show])
                print('\n')


def train() -> None:
    torch.manual_seed(2531)

    # ---------------------------------------------DATA PARAMETERS------------------------------------------------------
    train_files = [join(config.train_path, file) for file in listdir(config.train_path)]
    test_files = [join(config.test_path, file) for file in listdir(config.test_path)]
    vis_files = [join(config.validation_path, file) for file in listdir(config.validation_path)]
    min_threshold = 256
    max_threshold = 257
    train_dataset_size = 5_000
    test_dataset_size = 50
    vis_dataset_size = 5
    num_workers = 3
    min_noise = 0
    max_noise = 2
    # --------------------------------------------MODEL PARAMETERS------------------------------------------------------
    token_size = len(Collate.alphabet_to_num)
    pe_max_len = 1000
    num_layers = 6
    d_model = 512  # d_model % n_heads = 0
    n_heads = 16
    d_ff = 2048
    dropout = 0.1
    pre_trained = False
    weights_name = ''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -----------------------------------------OPTIMIZATION PARAMETERS--------------------------------------------------
    criterion = CrossEntropyLoss(ignore_index=-1)
    lr = 0  # fictive
    step_seek = 0
    warmup = 4_000
    lr_step_size = 2
    # ------------------------------------------TRAIN LOOP PARAMETERS---------------------------------------------------
    n_epochs = 300
    epoch_seek = 0
    batch_size = 5
    accumulation_step = 20  # train_dataset_size % (batch_size * accumulation_step) == 0
    saving_interval = 1
    log_interval = 1
    vis_interval = 1
    train_printable = True
    test_printable = True
    console_width = 94
    # ------------------------------------------------------------------------------------------------------------------

    train_dataset = WikiDataset(filenames=train_files, min_threshold=min_threshold, max_threshold=max_threshold,
                                dataset_size=train_dataset_size)
    test_dataset = WikiDataset(filenames=test_files, min_threshold=min_threshold, max_threshold=max_threshold,
                               dataset_size=test_dataset_size)
    vis_dataset = WikiDataset(filenames=vis_files, min_threshold=min_threshold, max_threshold=max_threshold,
                              dataset_size=vis_dataset_size)

    collate_fn = Collate(min_noise=min_noise, max_noise=max_noise)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True, collate_fn=collate_fn)
    vis_loader = DataLoader(dataset=vis_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, collate_fn=collate_fn)

    z_reader = get_model(token_size, pe_max_len, num_layers, d_model, n_heads, d_ff, dropout, device)

    if pre_trained:
        z_reader.load_parameters(join(config.weights_path, weights_name), device=device)

    optimizer = torch.optim.Adam(z_reader.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    scheduler = Scheduler(optimizer, d_model, warmup, lr_step_size)
    scheduler.seek(step_seek)

    try:
        for epoch_idx in range(epoch_seek + 1, epoch_seek + n_epochs + 1):
            offset = math.ceil((epoch_idx - 1) * train_dataset_size / batch_size)

            epoch_training(offset, z_reader, train_loader, optimizer, scheduler, accumulation_step, device,
                           log_interval, criterion, console_width, printable=train_printable)

            epoch_testing(offset, z_reader, test_loader, device, criterion, console_width, printable=test_printable)

            if epoch_idx % vis_interval == 0:
                epoch_visualization(z_reader, vis_loader, device, console_width)

            if epoch_idx % saving_interval == 0:
                save_parameters(offset + math.ceil(train_dataset_size / batch_size), z_reader)

            scheduler.step()

    except KeyboardInterrupt:
        print('Interrupted')

        if input('Save model? ') == 'y':
            save_parameters(-1, z_reader)


if __name__ == '__main__':
    train()
