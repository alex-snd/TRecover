from typing import Optional

import celery
import torch

import config
from src import utils
from src.model import ZReader


class ArtifactsTask(celery.Task):
    def __init__(self):
        super(ArtifactsTask, self).__init__()

        self.artifacts: Optional[dict] = None

    def __call__(self, *args, **kwargs):
        if not self.artifacts:
            self.artifacts = utils.load_artifacts(config.INFERENCE_DIR / 'artifacts.json')
            self.artifacts['cuda'] = config.CUDA and torch.cuda.is_available()

        return self.run(*args, **kwargs)


class PredictTask(celery.Task):
    def __init__(self):
        super(PredictTask, self).__init__()

        self.model: Optional[ZReader] = None
        self.device = torch.device(f'cuda' if config.CUDA and torch.cuda.is_available() else 'cpu')

    def __call__(self, *args, **kwargs):
        """
            Load model on first call (i.e. first task processed)
            Avoids the need to load model on each task request

        """

        if not self.model:
            artifacts = utils.load_artifacts(config.INFERENCE_DIR / 'artifacts.json')
            self.model = utils.get_model(artifacts['token_size'], artifacts['pe_max_len'], artifacts['num_layers'],
                                         artifacts['d_model'], artifacts['n_heads'], artifacts['d_ff'],
                                         artifacts['dropout'], self.device, weights=config.INFERENCE_DIR / 'weights')
            self.model.eval()

        return self.run(*args, **kwargs)
