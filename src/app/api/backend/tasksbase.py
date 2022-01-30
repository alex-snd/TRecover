from typing import Optional

import celery

from config import var


class ArtifactsTask(celery.Task):
    def __init__(self):
        super(ArtifactsTask, self).__init__()

        self.params: Optional[dict] = None

    def __call__(self, *args, **kwargs):
        """
        Load params config on first call (i.e. first task processed).
        Avoids the need to load params config on each task request.

        """

        if not self.params:
            import torch
            from zreader.utils.model import load_params

            self.params = load_params(var.INFERENCE_PARAMS_PATH)
            self.params.cuda = var.CUDA and torch.cuda.is_available()

        return self.run(*args, **kwargs)


class PredictTask(celery.Task):
    def __init__(self):

        super(PredictTask, self).__init__()

        self.device = None
        self.model = None

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed).
        Avoids the need to load model on each task request.

        """

        if not self.device:
            import torch

            self.device = torch.device(f'cuda' if var.CUDA and torch.cuda.is_available() else 'cpu')

        if not self.model:
            from zreader.utils.model import get_model, load_params

            self.update_state(state='LOADING')

            params = load_params(var.INFERENCE_PARAMS_PATH)
            self.model = get_model(params.token_size, params.pe_max_len, params.num_layers,
                                   params.d_model, params.n_heads, params.d_ff,
                                   params.dropout, self.device, weights=var.INFERENCE_WEIGHTS_PATH, silently=True)
            self.model.eval()

        self.update_state(state='PREDICT')

        return self.run(*args, **kwargs)
