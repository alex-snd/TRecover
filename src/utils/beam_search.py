import asyncio
from typing import Optional, List, Tuple, Callable, Awaitable

import celery
import torch
import torch.nn.functional as F
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from torch import Tensor

from src.ml.model import ZReader
from src.utils.visualization import visualize_target


# ----------------------------------------------Synchronous Beam Search-------------------------------------------------


def beam_step(candidates: List[Tuple[Tensor, float]],
              encoded_src: Tensor,
              z_reader: ZReader,
              width: int, device: torch.device
              ) -> List[Tuple[Tensor, float]]:
    step_candidates = list()

    for tgt_inp, score in candidates:
        prediction = z_reader.predict(tgt_inp, encoded_src, tgt_attn_mask=None, tgt_pad_mask=None, src_pad_mask=None)
        probabilities = F.log_softmax(prediction[:, -1], dim=-1).squeeze()  # first is batch

        values, indices = probabilities.topk(k=width, dim=-1)
        for prob, pos in zip(values, indices):
            new_token = torch.zeros(1, 1, z_reader.token_size, device=device)
            new_token[0, 0, pos] = 1

            step_candidates.append((torch.cat([tgt_inp, new_token], dim=1), score + float(prob)))

    return sorted(step_candidates, key=lambda candidate: -candidate[1])[:width]


def celery_task_loop(task: celery.Task
                     ) -> Callable[[Tensor, ZReader, int, torch.device], List[Tuple[Tensor, float]]]:
    def inner_loop(encoded_src: Tensor,
                   z_reader: ZReader,
                   width: int,
                   device: torch.device
                   ) -> List[Tuple[Tensor, float]]:
        candidates = [(torch.zeros(1, 1, z_reader.token_size, device=device), 0)]

        for progress in range(encoded_src.shape[0]):
            candidates = beam_step(candidates, encoded_src, z_reader, width, device)
            task.update_state(meta={'progress': progress + 1})

        return candidates

    return inner_loop


def cli_interactive_loop(label: str = 'Processing'
                         ) -> Callable[[Tensor, ZReader, int, torch.device], List[Tuple[Tensor, float]]]:
    def inner_loop(encoded_src: Tensor,
                   z_reader: ZReader,
                   width: int,
                   device: torch.device
                   ) -> List[Tuple[Tensor, float]]:
        candidates = [(torch.zeros(1, 1, z_reader.token_size, device=device), 0)]

        with Progress(
                TextColumn('{task.description}', style='bright_blue'),
                BarColumn(complete_style='bright_blue'),
                TextColumn('{task.percentage:>3.0f}%', style='bright_blue'),
                TextColumn('Remaining', style='bright_blue'),
                TimeRemainingColumn(),
                TextColumn('Elapsed', style='bright_blue'),
                TimeElapsedColumn(),
                transient=True,
        ) as progress:
            beam_progress = progress.add_task(label, total=encoded_src.shape[0])

            for _ in range(encoded_src.shape[0]):
                candidates = beam_step(candidates, encoded_src, z_reader, width, device)
                progress.update(beam_progress, advance=1)

        return candidates

    return inner_loop


def standard_loop(encoded_src: Tensor,
                  z_reader: ZReader,
                  width: int,
                  device: torch.device
                  ) -> List[Tuple[Tensor, float]]:
    candidates = [(torch.zeros(1, 1, z_reader.token_size, device=device), 0)]

    for _ in range(encoded_src.shape[0]):
        candidates = beam_step(candidates, encoded_src, z_reader, width, device)

    return candidates


def beam_search(src: Tensor,
                z_reader: ZReader,
                width: int,
                device: torch.device,
                beam_loop: Callable[[Tensor, ZReader, int, torch.device], List[Tuple[Tensor, float]]] = standard_loop
                ) -> List[Tuple[Tensor, float]]:
    src = src.unsqueeze(dim=0)

    encoded_src = z_reader.encode(src, src_pad_mask=None)

    candidates = beam_loop(encoded_src, z_reader, width, device)

    return [(torch.argmax(tgt.squeeze(), dim=-1)[1:], prob) for tgt, prob in candidates]  # first token is empty_token


# ---------------------------------------------Asynchronous Beam Search-------------------------------------------------


async def async_beam_step(candidates: List[Tuple[Tensor, float]],
                          encoded_src: Tensor,
                          z_reader: ZReader,
                          width: int, device: torch.device
                          ) -> List[Tuple[Tensor, float]]:
    async def candidate_step(tgt_inp: Tensor, score: float) -> None:
        prediction = z_reader.predict(tgt_inp, encoded_src, tgt_attn_mask=None, tgt_pad_mask=None, src_pad_mask=None)
        probabilities = F.log_softmax(prediction[:, -1], dim=-1).squeeze()  # first is batch

        values, indices = probabilities.topk(k=width, dim=-1)
        for prob, pos in zip(values, indices):
            new_token = torch.zeros(1, 1, z_reader.token_size, device=device)
            new_token[0, 0, pos] = 1

            step_candidates.append((torch.cat([tgt_inp, new_token], dim=1), score + float(prob)))

    step_candidates = list()

    for candidate_tgt_inp, candidate_score in candidates:
        await candidate_step(candidate_tgt_inp, candidate_score)

    return sorted(step_candidates, key=lambda candidate: -candidate[1])[:width]


def api_interactive_loop(queue: asyncio.Queue,
                         delimiter: str = ''
                         ) -> Callable[[Tensor, ZReader, int, torch.device], Awaitable]:
    async def async_inner_loop(encoded_src: Tensor,
                               z_reader: ZReader,
                               width: int,
                               device: torch.device
                               ) -> None:
        candidates = [(torch.zeros(1, 1, z_reader.token_size, device=device), 0)]

        for _ in range(encoded_src.shape[0]):
            candidates = await async_beam_step(candidates, encoded_src, z_reader, width, device)

            chains = [(torch.argmax(tgt.squeeze(), dim=-1)[1:], prob) for tgt, prob in candidates]
            chains = [(visualize_target(tgt, delimiter), prob) for tgt, prob in chains]

            await queue.put(chains)

        await queue.put(None)

    return async_inner_loop


async def standard_async_loop(encoded_src: Tensor,
                              z_reader: ZReader,
                              width: int,
                              device: torch.device
                              ) -> List[Tuple[Tensor, float]]:
    candidates = [(torch.zeros(1, 1, z_reader.token_size, device=device), 0)]

    for _ in range(encoded_src.shape[0]):
        candidates = await async_beam_step(candidates, encoded_src, z_reader, width, device)

    return candidates


async def async_beam_search(src: Tensor,
                            z_reader: ZReader,
                            width: int,
                            device: torch.device,
                            beam_loop: Callable[[Tensor, ZReader, int, torch.device],
                                                Awaitable[Optional[List[Tuple[Tensor, float]]]]] = standard_async_loop
                            ) -> Optional[List[Tuple[Tensor, float]]]:
    src = src.unsqueeze(dim=0)

    encoded_src = z_reader.encode(src, src_pad_mask=None)

    candidates = await beam_loop(encoded_src, z_reader, width, device)

    return [(torch.argmax(tgt.squeeze(), dim=-1)[1:], prob) for tgt, prob in candidates] if candidates else None
