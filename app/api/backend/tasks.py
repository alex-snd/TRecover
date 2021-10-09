from typing import Tuple, List, Dict

from app.api.backend.celeryapp import celery_app
from app.api.backend.tasksbase import ArtifactsTask, PredictTask
from src.utils.beam_search import beam_search, celery_task_loop
from src.utils.data import columns_to_tensor
from src.utils.visualization import visualize_target


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

    src = columns_to_tensor(data, self.device)

    chains = beam_search(src, self.model, beam_width, self.device, beam_loop=celery_task_loop(self))
    chains = [(visualize_target(chain, delimiter=delimiter), prob) for (chain, prob) in chains]

    return data, chains
