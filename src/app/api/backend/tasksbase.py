from typing import Optional

import celery

from config import var


class ArtifactsTask(celery.Task):
    def __init__(self):
        super(ArtifactsTask, self).__init__()

        self.params: Optional[dict] = None

    def __call__(self, *args, **kwargs):
        if not self.params:
            import torch
            from zreader.utils.model import load_params

            self.params = load_params(var.INFERENCE_PARAMS_PATH)
            self.params.cuda = var.CUDA and torch.cuda.is_available()

        return self.run(*args, **kwargs)


class PredictTask(celery.Task):
    def __init__(self):
        import torch
        from zreader.model import ZReader

        super(PredictTask, self).__init__()

        self.model: Optional[ZReader] = None
        self.device = torch.device(f'cuda' if var.CUDA and torch.cuda.is_available() else 'cpu')

    def __call__(self, *args, **kwargs):
        """
            Load model on first call (i.e. first task processed)
            Avoids the need to load model on each task request

        """

        if not self.model:
            from zreader.utils.model import get_model, load_params

            params = load_params(var.INFERENCE_PARAMS_PATH)
            self.model = get_model(params.token_size, params.pe_max_len, params.num_layers,
                                   params.d_model, params.n_heads, params.d_ff,
                                   params.dropout, self.device, weights=var.INFERENCE_WEIGHTS_PATH)
            self.model.eval()

        return self.run(*args, **kwargs)
