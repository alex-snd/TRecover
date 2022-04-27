import json
from pathlib import Path
from typing import Dict, Optional

import torch
from rich.prompt import Confirm

from trecover.config import log
from trecover.model import TRecover
from trecover.utils.train import ExperimentParams


def get_recent_weights_path(exp_dir: Path,
                            exp_mark: str,
                            weights_name: Optional[str] = None
                            ) -> Optional[Path]:
    """
    Get a model recent weights path.

    Parameters
    ----------
    exp_dir : Path
        Experiment directory path.
    exp_mark : str
        Experiment folder mark.
    weights_name : str, default=None
        Weights filename.

    Returns
    -------
    Optional[Path]:
        Recent weights path if it exists otherwise None object.

    """

    if weights_name:
        return weights_path if (weights_path := exp_dir / exp_mark / weights_name).exists() else None

    if (weights_path := exp_dir / exp_mark / 'weights').exists():
        recent_weights = None
        most_recent_timestamp = 0

        for weights in weights_path.iterdir():
            if timestamp := weights.stat().st_ctime > most_recent_timestamp:
                recent_weights = weights
                most_recent_timestamp = timestamp

        return recent_weights


def get_model(token_size: int,
              pe_max_len: int,
              num_layers: int,
              d_model: int,
              n_heads: int,
              d_ff: int,
              dropout: float,
              device: torch.device = torch.device('cpu'),
              weights: Optional[Path] = None,
              silently: bool = False
              ) -> TRecover:
    """
    Get a model with specified configuration.

    Parameters
    ----------
    token_size : int
        Token (column) size.
    pe_max_len : int
        Positional encoding max length.
    num_layers : int
        Number of encoder and decoder blocks
    d_model : int
        Model dimension - number of expected features in the encoder (decoder) input.
    n_heads : int
        Number of encoder and decoder attention heads.
    d_ff : int
        Dimension of the feedforward layer.
    dropout : float,
        Dropout range.
    device : torch.device, default=torch.device('cpu')
        Device on which to allocate the model.
    weights : Path, default=None
        Model weights path for initialization.
    silently : bool, default=False
        Initialize the model silently without any verbose information.

    Returns
    -------
    model : TRecover
        Initialized model.

    Raises
    ------
    SystemExit:
        If the weight's path is not provided and the cli 'stop' option is selected.

    """

    model = TRecover(token_size, pe_max_len, num_layers, d_model, n_heads, d_ff, dropout).to(device)

    if weights and weights.exists() and weights.is_file():
        model.load_parameters(weights, device=device)

        if not silently:
            log.project_console.print(f'The below model parameters have been loaded:\n{weights}',
                                      style='bright_green')
        return model

    if weights:
        log.project_console.print(f'Failed to load model parameters: {str(weights)}', style='bold red')
    else:
        log.project_console.print("Model parameters aren't specified", style='bright_blue')

    if silently or Confirm.ask(prompt='[bright_blue]Continue training from scratch?', default=True,
                               console=log.project_console):
        return model
    else:
        raise SystemExit


def load_params(model_params: Path) -> ExperimentParams:
    """
    Get experiment parameters container.

    Parameters
    ----------
    model_params : Path
        Path to serialized experiment parameters.

    Returns
    -------
    ExperimentParams:
        Experiment parameters container as a ExperimentParams object.

    """

    return ExperimentParams(json.load(model_params.open()))


def save_params(data: Dict, filepath: Path, sort=False) -> None:
    """
    Save experiment parameters on disk.

    Parameters
    ----------
    data : Dict
        Experiment parameters.
    filepath : Path
        File path for saving.
    sort : bool, default=False
        Perform parameters keys sorting.

    """

    with filepath.open('w') as f:
        json.dump(data, indent=2, fp=f, sort_keys=sort)
