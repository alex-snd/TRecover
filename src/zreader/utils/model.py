import json
from pathlib import Path
from typing import Dict, Optional

import torch
from rich.prompt import Confirm

from config import log
from zreader.model import ZReader
from zreader.utils.train import ExperimentParams


def get_recent_weights_path(exp_dir: Path,
                            exp_mark: Optional[str] = None,
                            weights_name: Optional[str] = None
                            ) -> Optional[Path]:
    if exp_mark:
        if weights_name:
            return exp_dir / exp_mark / weights_name

        weights_path = exp_dir / exp_mark / 'weights'

        if weights_path.exists():
            recent_weights = None
            most_recent_timestamp = 0

            for weights in weights_path.iterdir():
                if timestamp := weights.stat().st_ctime > most_recent_timestamp:
                    recent_weights = weights
                    most_recent_timestamp = timestamp

            return recent_weights

    return None


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
              ) -> ZReader:
    model = ZReader(token_size, pe_max_len, num_layers, d_model, n_heads, d_ff, dropout).to(device)

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
    # TODO load mlflow artifacts
    return ExperimentParams(json.load(model_params.open()))


def save_params(data: Dict, filepath: Path, sort=False) -> None:
    with filepath.open('w') as f:
        json.dump(data, indent=2, fp=f, sort_keys=sort)
