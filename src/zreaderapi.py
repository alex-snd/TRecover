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

cli = Typer(name='Zreader-api', epilog='Description will be here')
api = FastAPI(title='ZreaderAPI', description='Description will be here')  # TODO write description

artifacts = {}  # mlflow artifacts TODO remove


@api.on_event('startup')
def startup() -> None:
    global artifacts
    artifacts = utils.load_artifacts(config.INFERENCE_DIR / 'artifacts.json')

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

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        param: artifacts.get(param, 'Not found')
    }

    return response


@api.post('/zread', tags=['Prediction'], response_model=TaskResponse)
@construct_response
def zread(request: Request, payload: PredictPayload) -> Dict:
    if len(payload.data) > artifacts['pe_max_len']:  # TODO use .get or remove artifacts
        response = {
            'message': f'{HTTPStatus.REQUEST_ENTITY_TOO_LARGE.phrase}. Number of columns must be less than 1000.',
            'status_code': HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        }
    else:
        task = predict.delay(payload.data, payload.beam_width, payload.delimiter)

        response = {
            'message': HTTPStatus.ACCEPTED.phrase,
            'status_code': HTTPStatus.ACCEPTED,
            'task_id': task.id
        }

    return response


@api.get('/status/{task_id}', tags=['Prediction'], response_model=PredictResponse)
@construct_response
def status(request: Request, task_id: str) -> Dict:
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
        info = task.info

        response = {
            'message': HTTPStatus.PROCESSING.phrase,
            'status_code': HTTPStatus.PROCESSING,
            'progress': info.get('progress') if isinstance(info, dict) else None
        }

    return response


@api.delete('/status/{task_id}', tags=['Prediction'])
@construct_response
def delete_prediction(request: Request, task_id: str) -> Dict:
    task = AsyncResult(task_id, app=worker_app)

    if task.ready():
        task.forget()
    else:
        task.revoke()

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK
    }

    return response


@cli.command()
def run(host: str = Option(config.FASTAPI_HOST, help='Bind socket to this host'),
        port: int = Option(config.FASTAPI_PORT, help='Bind socket to this port'),
        log_level: str = Option('info', help='Log level'),
        reload: bool = Option(False, help='Enable auto-reload'),
        workers: int = Option(config.FASTAPI_WORKERS, help='Number of worker processes')
        ) -> None:
    """ Run uvicorn server """

    uvicorn.run('zreaderapi:api', host=host, port=port, log_level=log_level, reload=reload, workers=workers)


if __name__ == '__main__':
    cli()
