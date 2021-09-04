from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import Dict, Callable

import uvicorn
from celery.result import AsyncResult
from fastapi import FastAPI, Request
from typer import Typer, Option

import config
import utils
from api_schemas import PredictPayload, PredictResponse, TaskResponse
from celery_worker import worker_app, predict

cli = Typer(name='ZreaderAPI', epilog='Description will be here')
api = FastAPI(title='ZreaderAPI', description='Description will be here')

context = {
    'artifacts': {},  # mlflow artifacts
    'tasks': set()  # list of celery tasks identifiers
}


@api.on_event('startup')
def startup() -> None:
    global context

    artifacts = utils.load_artifacts(config.INFERENCE_DIR / 'artifacts.json')
    context['artifacts'] = artifacts

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

    global context

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        'artifacts': context['artifacts']
    }

    return response


@api.get('/params/{param}', tags=['Parameters'])
@construct_response
def parameters(request: Request, param: str) -> Dict:
    """ Get a specific parameter's value used for inference. """

    global context

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        param: context['artifacts'].get(param, 'Not found')
    }

    return response


@api.post('/zread', tags=['Prediction'], response_model=TaskResponse)
@construct_response
def zread(request: Request, payload: PredictPayload) -> Dict:
    global context

    task = predict.delay(payload.data, payload.beam_width, payload.delimiter)

    context['tasks'].add(task.id)

    response = {
        'message': HTTPStatus.ACCEPTED.phrase,
        'status_code': HTTPStatus.ACCEPTED,
        'task_id': task.id
    }

    return response


@api.get('/status/{task_id}', tags=['Prediction'], response_model=PredictResponse)
@construct_response
def status(request: Request, task_id: str) -> Dict:
    global context

    if task_id in context['tasks']:
        task = AsyncResult(task_id, app=worker_app)

        if task.ready():
            data, chains = task.get()

            response = {
                'message': HTTPStatus.OK.phrase,
                'status_code': HTTPStatus.OK,
                'data': data,
                'chains': chains
            }
        else:
            response = {
                'message': HTTPStatus.PROCESSING.phrase,
                'status_code': HTTPStatus.PROCESSING,
                'progress': task.info.get('progress', None)
            }

    else:
        response = {
            'message': HTTPStatus.NOT_FOUND.phrase,
            'status_code': HTTPStatus.NOT_FOUND,
        }

    return response


@api.delete('/status/{task_id}', tags=['Prediction'])
@construct_response
def delete_prediction(request: Request, task_id: str) -> Dict:
    global context

    if task_id in context['tasks']:
        task = AsyncResult(task_id, app=worker_app)

        if task.ready():
            task.forget()
        else:
            task.revoke()

        context['tasks'].remove(task_id)

        response = {
            'message': HTTPStatus.OK.phrase,
            'status_code': HTTPStatus.OK
        }
    else:
        response = {
            'message': HTTPStatus.NOT_FOUND.phrase,
            'status_code': HTTPStatus.NOT_FOUND
        }

    return response


@cli.command()
def run(host: str = Option('localhost', help='Bind socket to this host'),
        port: int = Option(5001, help='Bind socket to this port'),
        log_level: str = Option('info', help='Log level'),
        reload: bool = Option(False, help='Enable auto-reload'),
        workers: int = Option(1, help='Number of worker processes')
        ) -> None:
    """ Run celery worker and uvicorn server """

    # TODO start celery worker here

    uvicorn.run('zreaderapi:api', host=host, port=port, log_level=log_level, reload=reload, workers=workers)


if __name__ == '__main__':
    cli()
