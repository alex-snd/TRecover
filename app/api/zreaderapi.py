from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import Dict, Callable

from celery.result import AsyncResult
from fastapi import FastAPI, Request, Path

import config
from app.api.backend.celeryapp import celery_app
from app.api.backend.tasks import predict, get_artifacts
from app.api.schemas import PredictPayload, PredictResponse, TaskResponse

api = FastAPI(title='ZreaderAPI', description='Description will be here')  # TODO write description


@api.on_event('startup')
def startup() -> None:
    config.project_logger.info('FatAPI launched')


def construct_response(handler: Callable[..., Dict]) -> Callable[..., Dict]:
    """ Construct a JSON response for an endpoint's results. """

    @wraps(handler)  # TODO figure out
    def wrap(request: Request, *args, **kwargs) -> Dict:
        response = handler(request, *args, **kwargs)

        response['method'] = request.method
        response['timestamp'] = datetime.now().isoformat()
        response['url'] = request.url._url

        return response

    return wrap


@api.get('/', tags=['General'])
@construct_response
def index(request: Request) -> Dict:
    return {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK
    }


@api.get('/params', tags=['Parameters'])
@construct_response
def all_parameters(request: Request) -> Dict:
    """ Get model parameter's values used for inference. """
    task = get_artifacts.delay()
    artifacts = task.get()

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        'artifacts': artifacts
    }

    return response


@api.get('/params/{param}', tags=['Parameters'])
@construct_response
def parameters(request: Request, param: str) -> Dict:
    """ Get a specific parameter's value used for inference. """

    task = get_artifacts.delay()
    artifacts = task.get()

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        param: artifacts.get(param, 'Not found')
    }

    return response


@api.post('/zread', tags=['Prediction'], response_model=TaskResponse)
@construct_response
def zread(request: Request, payload: PredictPayload) -> Dict:
    task = predict.delay(payload.data, payload.beam_width, payload.delimiter)

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
    task = AsyncResult(task_id, app=celery_app)

    if task.failed():
        response = {
            'message': str(task.info),
            'status_code': HTTPStatus.CONFLICT
        }

    elif task.ready():
        data, chains = task.get()

        response = {
            'message': HTTPStatus.OK.phrase,
            'status_code': HTTPStatus.OK,
            'data': data,
            'chains': chains,
            'progress': len(data)
        }
    else:
        info = task.info

        response = {
            'message': HTTPStatus.PROCESSING.phrase,
            'status_code': HTTPStatus.PROCESSING,
            'progress': info.get('progress') if isinstance(info, dict) else None
        }

    return response


@api.delete('/status/{task_id}', tags=['Prediction'])
@construct_response
def delete_prediction(request: Request,
                      task_id: str = Path(...,
                                          title='The ID of the task to get status',
                                          regex=r'[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}')
                      ) -> Dict:
    task = AsyncResult(task_id, app=celery_app)

    if task.ready():
        task.forget()
    else:
        task.revoke()

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK
    }

    return response
