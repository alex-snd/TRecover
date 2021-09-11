from typing import Optional, Tuple, List, Dict

import celery
import torch
from celery.exceptions import Ignore

import config
import utils
from model import ZReader


# -------------------------------------------------Celery Configuration-------------------------------------------------


class Celery(celery.Celery):

    def gen_task_name(self, name: str, module: str) -> str:
        if module.startswith('src.'):
            module = module[4:]

        return super().gen_task_name(name, module)


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


worker_app = Celery('ZReader',
                    broker=config.CELERY_BROKER,
                    backend=config.CELERY_BACKEND
                    )

worker_app.conf.update({
    'task_serializer': 'json',
    'result_serializer': 'json',
    'accept_content': ['json'],
    'worker_prefetch_multiplier': 1,
    'task_acks_late': True,
    'task_track_started': True,
    'task_reject_on_worker_lost': True,
})


# --------------------------------------------------Celery Tasks--------------------------------------------------------

@worker_app.task(bind=True, base=ArtifactsTask)
def get_artifacts(self: ArtifactsTask) -> Dict:
    return self.artifacts


@worker_app.task(bind=True, base=PredictTask)
def predict(self: PredictTask,
            data: List[str],
            beam_width: int,
            delimiter: str
            ) -> Tuple[List[str], List[Tuple[str, float]]]:
    assert len(data) <= self.model.pe_max_len, f'Number of columns must be less than {self.model.pe_max_len}.'

    src = utils.columns_to_tensor(data, self.device)

    chains = utils.beam_search(src, self.model, beam_width, self.device, beam_loop=utils.celery_task_loop(self))
    chains = [(utils.visualize_target(chain, delimiter=delimiter), prob) for (chain, prob) in chains]

    return data, chains


if __name__ == '__main__':
    pass
