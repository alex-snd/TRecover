from typing import Tuple, List, Dict

from trecover.app.api.backend.celeryapp import celery_app
from trecover.app.api.backend.tasksbase import ModelConfigTask, PredictTask


@celery_app.task(bind=True, base=ModelConfigTask)
def get_model_config(self: ModelConfigTask) -> Dict:
    """
    Celery task implementation that returns values of the model configuration.

    Parameters
    ----------
    self : ModelConfigTask
        Celery task base class.

    Returns
    -------
    self.config : Dict
        The values of the model configuration.

    """

    return self.config


@celery_app.task(bind=True, base=PredictTask)
def predict(self: PredictTask,
            columns: List[str],
            beam_width: int,
            delimiter: str
            ) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Celery task implementation that performs keyless reading.

    Parameters
    ----------
    self : ArtifactsTask
        Celery task base class.
    columns : List[str]
        Columns to keyless read.
    beam_width : int
        Width for beam search algorithm. Maximum value is alphabet size.
    delimiter : str
        Delimiter for columns visualization.

    Returns
    -------
    (columns, chains) : Tuple[List[str], List[Tuple[str, float]]]
        The columns and read chains.

    Raises
    ------
    AssertionError:
        If the number of columns is grater than self.model.pe_max_len.

    """

    from trecover.utils.beam_search import beam_search, celery_task_loop
    from trecover.utils.transform import columns_to_tensor, tensor_to_target
    from trecover.utils.visualization import visualize_target

    assert len(columns) <= self.model.pe_max_len, f'Number of columns must be less than {self.model.pe_max_len}.'

    src = columns_to_tensor(columns, self.device)

    chains = beam_search(src, self.model, beam_width, self.device, beam_loop=celery_task_loop(self))
    chains = [(visualize_target(tensor_to_target(chain), delimiter=delimiter), prob) for (chain, prob) in chains]

    return columns, chains
