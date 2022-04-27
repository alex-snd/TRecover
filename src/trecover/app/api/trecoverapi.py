from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import Dict, Callable

from celery.result import AsyncResult
from fastapi import FastAPI, Request, Path

from trecover.app.api.backend.celeryapp import celery_app
from trecover.app.api.backend.tasks import predict, get_model_config
from trecover.app.api.schemas import PredictPayload, PredictResponse, TaskResponse
from trecover.config import log

api = FastAPI(title='TRecoverAPI', description='API for TRecover model')


@api.on_event('startup')
def startup() -> None:
    """ API startup handler. """

    log.project_logger.info('FatAPI launched')


def construct_response(handler: Callable[..., Dict]) -> Callable[..., Dict]:
    """
    A decorator that wraps a request handler.

    Parameters
    ----------
    handler : Callable[..., Dict]
        Request processing function.

    Returns
    -------
    wrap : Callable[..., Dict]
        Decorated handler.

    """

    @wraps(handler)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        """
        A wrapper that constructs a JSON response for an endpoint's results.

        Parameters
        ----------
        request : Request
            Client request information.

        Returns
        -------
        response : Dict
            The result of the handler function.

        """

        response = handler(request, *args, **kwargs)

        response['method'] = request.method
        response['timestamp'] = datetime.now().isoformat()
        response['url'] = request.url._url

        return response

    return wrap


@api.get('/', tags=['General'])
@construct_response
def index(request: Request) -> Dict:
    """
    Healthcheck handler.

    Parameters
    ----------
    request : Request
        Client request information.

    Returns
    -------
    Dict:
        OK phrase as a Dict response.

    """

    return {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK
    }


@api.get('/config', tags=['Configuration'])
@construct_response
def config(request: Request) -> Dict:
    """
    Get the configuration of the model used for inference.

    Parameters
    ----------
    request : Request
        Client request information.

    Returns
    -------
    response : Dict
        Response containing the values of the model configuration in the 'config' field.

    """

    task = get_model_config.delay()
    model_config = task.get()

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        'config': model_config
    }

    return response


@api.get('/config/{param}', tags=['Configuration'])
@construct_response
def config_param(request: Request, param: str) -> Dict:
    """
    Get the specific configuration parameter of the model used for inference.

    Parameters
    ----------
    request : Request
        Client request information.
    param : str
        Parameter name.

    Returns
    -------
    response : Dict
        Response containing the value of the specific configuration parameter in
        the '<param>' field if it exists, otherwise 'Not found' value.

    """

    task = get_model_config.delay()
    model_config = task.get()

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        param: model_config.get(param, 'Not found')
    }

    return response


@api.post('/recover', tags=['Prediction'], response_model=TaskResponse)
@construct_response
def recover(request: Request, payload: PredictPayload) -> Dict:
    """
    Perform keyless reading.

    Parameters
    ----------
    request : Request
        Client request information.
    payload : PredictPayload
        Data for keyless reading.

    Returns
    -------
    response : TaskResponse
        Response containing the id of the celery task in the 'task_id' field.

    """

    task = predict.delay(payload.columns, payload.beam_width, payload.delimiter)

    response = {
        'message': HTTPStatus.ACCEPTED.phrase,
        'status_code': HTTPStatus.ACCEPTED,
        'task_id': task.id
    }

    return response


@api.get('/status/{task_id}', tags=['Prediction'], response_model=PredictResponse)
@construct_response
def status(request: Request,
           task_id: str = Path(...,
                               title='The ID of the task to get status',
                               regex=r'[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}')
           ) -> Dict:
    """
    Get a celery task status.

    Parameters
    ----------
    request : Request
        Client request information.
    task_id : str
        Celery task id.

    Returns
    -------
    response : PredictResponse
        Response containing the status of the celery task in the 'state' and 'progress'
        fields if it's still in process,
        error information in 'message' and 'status_code' fields if it's failed,
        otherwise the result of keyless reading in the 'chains' field.

    """

    task = AsyncResult(task_id, app=celery_app)

    if task.failed():
        response = {
            'message': str(task.info),
            'status_code': HTTPStatus.CONFLICT
        }

    elif task.ready():
        columns, chains = task.get()

        response = {
            'message': HTTPStatus.OK.phrase,
            'status_code': HTTPStatus.OK,
            'columns': columns,
            'chains': chains,
            'progress': len(columns)
        }
    else:
        info = task.info

        response = {
            'message': HTTPStatus.PROCESSING.phrase,
            'status_code': HTTPStatus.PROCESSING,
            'state': task.status,
            'progress': info.get('progress') if isinstance(info, dict) else None
        }

    return response


@api.delete('/{task_id}', tags=['Prediction'])
@construct_response
def delete_prediction(request: Request,
                      task_id: str = Path(...,
                                          title='The ID of the task to forget',
                                          regex=r'[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}')
                      ) -> Dict:
    """
    Delete task result in the celery backend database.

    Parameters
    ----------
    request : Request
        Client request information.
    task_id : str
        Task ID to delete its result from celery backend database.

    Returns
    -------
    response : Dict
        OK phrase.

    """

    task = AsyncResult(task_id, app=celery_app)

    if task.ready():
        task.forget()
    else:
        task.revoke(terminate=True)

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK
    }

    return response
