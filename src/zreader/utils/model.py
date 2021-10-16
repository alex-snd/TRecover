import json
from pathlib import Path
from typing import Dict, Union, Optional

import torch
from rich.prompt import Confirm

import config
from zreader.ml.model import ZReader


def get_model(token_size: int,
              pe_max_len: int,
              num_layers: int,
              d_model: int,
              n_heads: int,
              d_ff: int,
              dropout: float,
              device: torch.device = torch.device('cpu'),
              weights: Optional[Path] = None,
              prompt: bool = False
              ) -> ZReader:
    model = ZReader(token_size, pe_max_len, num_layers, d_model, n_heads, d_ff, dropout).to(device)

    if weights and weights.exists() and weights.is_file():
        model.load_parameters(weights, device=device)

        return model

    if weights:
        config.project_console.print(f'Failed to load model parameters: {str(weights)}', style='bold red')
    else:
        config.project_console.print("Model parameters aren't specified", style='bright_blue')

    if not prompt or Confirm.ask(prompt='[bright_blue]Continue training from scratch?', default=True,
                                 console=config.project_console):
        return model
    else:
        raise SystemExit()


# TODO for mlflow
def simplify_artifacts(artifacts: Dict) -> Dict:
    pass


def load_artifacts(model_artifacts: Path) -> Dict[str, Union[str, int, float]]:
    # TODO load mlflow artifacts
    return json.load(model_artifacts.open())


def save_artifacts(data: Dict, filepath: Path, sort=False) -> None:
    with open(filepath, 'w') as f:
        json.dumps(data, indent=2, fp=f, sort_keys=sort)
