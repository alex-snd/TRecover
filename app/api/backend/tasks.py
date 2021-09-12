from typing import Tuple, List, Dict

from app.api.backend.celeryapp import celery_app
from app.api.backend.tasksbase import ArtifactsTask, PredictTask
from ml import utils


@celery_app.task(bind=True, base=ArtifactsTask)
def get_artifacts(self: ArtifactsTask) -> Dict:
    return self.artifacts


@celery_app.task(bind=True, base=PredictTask)
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
