import asyncio
from typing import Optional, List, Tuple, Callable, Awaitable

import celery
import torch
import torch.nn.functional as F
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from torch import Tensor

from config import log
from zreader.model import ZReader
from zreader.utils.visualization import visualize_target


# ------------------------------------------Beam Search Auxiliary Functions---------------------------------------------


def get_steps_params(src: Tensor) -> Tuple[Tensor, List[int]]:
    """
    Get probability masks and beam widths required for each step of beam search algorithm.

    Parameters
    ----------
    src: Tensor[BATCH_SIZE, SEQUENCE_LEN, TOKEN_SIZE]
        Keyless reading columns that are passed to the ZReader encoder.

    Returns
    -------
    steps_mask: Tensor[BATCH_SIZE, SEQUENCE_LEN, TOKEN_SIZE]  TODO investigate
        Probability masks that consists of zeros in the places that correspond to the letters allowed
        for selection in the column (src[i]) and values equal to minus infinity in all others.
    steps_width: List[int]
        The beam width for each step of the algorithm.

    """

    return (
        torch.full_like(src, fill_value=float('-inf')).masked_fill(src == 1, value=0.0),
        src.sum(dim=-1).int().tolist()
    )


# ----------------------------------------------Synchronous Beam Search-------------------------------------------------


def beam_step(candidates: List[Tuple[Tensor, float]],
              step_mask: Tensor,
              step_width: int,
              encoded_src: Tensor,
              z_reader: ZReader,
              beam_width: int,
              device: torch.device
              ) -> List[Tuple[Tensor, float]]:
    """
    Implementation of the beam search algorithm step.

    Parameters
    ----------
    candidates: List[Tuple[Tensor[1, STEP_NUMBER, TOKEN_SIZE], float]]
        List of candidates from the previous step.
    step_mask: Tensor[1, TOKEN_SIZE] TODO investigate
        Column's mask that consists of zeros in the places that correspond to the letters allowed
        for selection in the column and values equal to minus infinity in all others.
        Required so that only the letters in the column are selected as a candidates.
    step_width: int
        Number of candidates that are contained in the step column.
    encoded_src: Tensor TODO investigate
        Columns for keyless reading that were encoded by ZReader encoder.
    z_reader: ZReader
        Trained model for keyless reading.
    beam_width: int
        Number of candidates that can be selected at the current step.
    device: torch.device
        Device on which to allocate the candidate chains.

    Returns
    -------
    step_candidates: List[Tuple[Tensor[1, STEP_NUMBER + 1, TOKEN_SIZE], float]]
        List of candidates of size "beam_width" for the current step
        sorted in descending order of their probabilities.

    Notes
    -----
    For each chain candidate from the previous step:
    *       Probability distribution is calculated using trained ZReader model to
            select the next symbol from the current column,taking into account the "step_mask".
    *       The most probable symbols are selected from the calculated probability distribution,
            the number of which is set by the "step_width" and "beam_width" parameters.
    *       For each selected symbol, a new candidate chain with updated probability
            is constructed and placed in the "step_candidates" list.

    All candidates are sorted in descending order of probabilities and the most probable ones
    are selected from them, the number of which is set by the "beam_width" parameter.

    """

    step_candidates = list()

    for chain, score in candidates:
        prediction = z_reader.predict(chain, encoded_src, tgt_attn_mask=None, tgt_pad_mask=None, src_pad_mask=None)
        probabilities = F.log_softmax(prediction[0, -1], dim=-1) + step_mask

        values, indices = probabilities.topk(k=min(beam_width, step_width))
        for prob, pos in zip(values, indices):
            new_token = torch.zeros(1, 1, z_reader.token_size, device=device)
            new_token[0, 0, pos] = 1

            step_candidates.append((torch.cat([chain, new_token], dim=1), score + float(prob)))

    return sorted(step_candidates, key=lambda candidate: -candidate[1])[:beam_width]


def celery_task_loop(task: celery.Task
                     ) -> Callable[[Tensor, Tensor, ZReader, int, torch.device], List[Tuple[Tensor, float]]]:
    """
    Get a beam search algorithm loop function, which is implemented for the Celery task.

    Parameters
    ----------
    task: celery.Task
        Celery task base class.

    Returns
    -------
    inner_loop: Callable[[Tensor, Tensor, ZReader, int, torch.device], List[Tuple[Tensor, float]]]
        Beam search algorithm loop function for the Celery task.

    """

    def inner_loop(src: Tensor,
                   encoded_src: Tensor,
                   z_reader: ZReader,
                   width: int,
                   device: torch.device
                   ) -> List[Tuple[Tensor, float]]:
        """
        Beam search algorithm loop implementation for the Celery task.

        Parameters
        ----------
        src: Tensor[BATCH_SIZE, SEQUENCE_LEN, TOKEN_SIZE]
            Keyless reading columns that are passed to the ZReader encoder.
        encoded_src: Tensor TODO investigate
            Keyless reading columns that were encoded by ZReader encoder.
        z_reader: ZReader
            Trained model for keyless reading.
        width: int
            Number of candidates that can be selected at the each step.
        device: torch.device
            Device on which to allocate the candidate chains.

        Returns
        -------
        candidates: List[Tuple[Tensor[1, SEQUENCE_LEN + 1, TOKEN_SIZE], float]]
            List of chains sorted in descending order of probabilities.
            The number of candidates is set by the "width" parameter.

        Notes
        -----
        Probability masks and beam widths values required for each step of the
        beam search algorithm are calculated by the "get_steps_params" function using
        keyless reading columns ("src") that are passed to the ZReader encoder.

        An empty chain (zero token of shape [1, 1, TOKEN_SIZE]) with zero probability
        is used as the first candidate for the algorithm.

        The progress of the Celery task is updated at each step of the algorithm.

        """

        step_masks, step_widths = get_steps_params(src)
        candidates = [(torch.zeros(1, 1, z_reader.token_size, device=device), 0)]

        for i in range(encoded_src.shape[0]):
            candidates = beam_step(candidates, step_masks[i], step_widths[i], encoded_src, z_reader, width, device)
            task.update_state(meta={'progress': i + 1}, state='PREDICT')

        return candidates

    return inner_loop


def cli_interactive_loop(label: str = 'Processing'
                         ) -> Callable[[Tensor, Tensor, ZReader, int, torch.device], List[Tuple[Tensor, float]]]:
    """
    Get a beam search algorithm loop function, which is implemented for the cli interface.

    Parameters
    ----------
    label: str
        Label for the cli progress bar.

    Returns
    -------
    inner_loop: Callable[[Tensor, Tensor, ZReader, int, torch.device], List[Tuple[Tensor, float]]]
        Beam search algorithm loop function for the cli interface.

    """

    def inner_loop(src: Tensor,
                   encoded_src: Tensor,
                   z_reader: ZReader,
                   width: int,
                   device: torch.device
                   ) -> List[Tuple[Tensor, float]]:
        """
        Beam search algorithm loop implementation for the cli interface.

        Parameters
        ----------
        src: Tensor[BATCH_SIZE, SEQUENCE_LEN, TOKEN_SIZE]
            Keyless reading columns that are passed to the ZReader encoder.
        encoded_src: Tensor TODO investigate
            Keyless reading columns that were encoded by ZReader encoder.
        z_reader: ZReader
            Trained model for keyless reading.
        width: int
            Number of candidates that can be selected at the each step.
        device: torch.device
            Device on which to allocate the candidate chains.

        Returns
        -------
        candidates: List[Tuple[Tensor[1, SEQUENCE_LEN + 1, TOKEN_SIZE], float]]
            List of chains sorted in descending order of probabilities.
            The number of candidates is set by the "width" parameter.

        Notes
        -----
        Probability masks and beam widths values required for each step of the
        beam search algorithm are calculated by the "get_steps_params" function using
        keyless reading columns ("src") that are passed to the ZReader encoder.

        An empty chain (zero token of shape [1, 1, TOKEN_SIZE]) with zero probability
        is used as the first candidate for the algorithm.

        At each step of the algorithm, the progress bar of the task
        is updated and displayed in the console.

        """

        step_masks, step_widths = get_steps_params(src)
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
                console=log.project_console
        ) as progress:
            beam_progress = progress.add_task(label, total=encoded_src.shape[0])

            for i in range(encoded_src.shape[0]):
                candidates = beam_step(candidates, step_masks[i], step_widths[i], encoded_src, z_reader, width, device)
                progress.update(beam_progress, advance=1)

        return candidates

    return inner_loop


def standard_loop(src: Tensor,
                  encoded_src: Tensor,
                  z_reader: ZReader,
                  width: int,
                  device: torch.device
                  ) -> List[Tuple[Tensor, float]]:
    """
    Base implementation of the beam search algorithm loop.

    Parameters
    ----------
    src: Tensor[BATCH_SIZE, SEQUENCE_LEN, TOKEN_SIZE]
        Keyless reading columns that are passed to the ZReader encoder.
    encoded_src: Tensor TODO investigate
        Keyless reading columns that were encoded by ZReader encoder.
    z_reader: ZReader
        Trained model for keyless reading.
    width: int
        Number of candidates that can be selected at the each step.
    device: torch.device
        Device on which to allocate the candidate chains.

    Returns
    -------
    candidates: List[Tuple[Tensor[1, SEQUENCE_LEN + 1, TOKEN_SIZE], float]]
        List of chains sorted in descending order of probabilities.
        The number of candidates is set by the "width" parameter.

    Notes
    -----
    Probability masks and beam widths values required for each step of the
    beam search algorithm are calculated by the "get_steps_params" function using
    keyless reading columns ("src") that are passed to the ZReader encoder.

    An empty chain (zero token of shape [1, 1, TOKEN_SIZE]) with zero probability
    is used as the first candidate for the algorithm.

    """

    step_masks, step_widths = get_steps_params(src)
    candidates = [(torch.zeros(1, 1, z_reader.token_size, device=device), 0)]

    for i in range(encoded_src.shape[0]):
        candidates = beam_step(candidates, step_masks[i], step_widths[i], encoded_src, z_reader, width, device)

    return candidates


def beam_search(src: Tensor,
                z_reader: ZReader,
                width: int,
                device: torch.device,
                beam_loop: Callable[[Tensor, Tensor, ZReader, int, torch.device],
                                    List[Tuple[Tensor, float]]] = standard_loop
                ) -> List[Tuple[Tensor, float]]:
    """
    Beam search algorithm implementation.

    Parameters
    ----------
    src: Tensor[BATCH_SIZE, SEQUENCE_LEN, TOKEN_SIZE]
        Keyless reading columns that are passed to the ZReader encoder.
    z_reader: ZReader
        Trained model for keyless reading.
    width: int
        Number of candidates that can be selected at the each step.
    device: torch.device
        Device on which to allocate the candidate chains.
    beam_loop: Callable[[Tensor, Tensor, ZReader, int, torch.device], List[Tuple[Tensor, float]]], default=standard_loop
        Beam search algorithm loop function.

    Returns
    -------
    candidates: List[Tuple[Tensor[SEQUENCE_LEN, 1], float]] TODO investigate
        List of chains sorted in descending order of probabilities.
        The number of candidates is set by the "width" parameter.

    Notes
    -----
    Initially, the keyless reading columns ("src") are encoded using ZReader encoder,
    then the encoded columns ("encoded_src") are used at each step of the algorithm.

    """

    encoded_src = z_reader.encode(src.unsqueeze(dim=0), src_pad_mask=None)

    candidates = beam_loop(src, encoded_src, z_reader, width, device)

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
