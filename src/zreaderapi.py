import asyncio
import uuid
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict, Callable, Awaitable

import torch
import uvicorn
from fastapi import FastAPI, Request
from typer import Typer, Option

import config
import utils
from api_schemas import PredictPayload, PredictResponse, InteractiveResponse, JobStatus
from model import ZReader

cli = Typer(name='ZreaderAPI', epilog='Description will be here')
api = FastAPI(title='ZreaderAPI', description='Description will be here', version='0.1')

model: ZReader  # declare model for further initialization on server startup
context = {
    'artifacts': {},  # mlflow artifacts
    'jobs': {}  # interactive api-calls
}
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')


@api.on_event('startup')
async def load_model() -> None:
    global model, context, device

    model_artifacts = config.INFERENCE_DIR / 'artifacts.json'
    weights_path = config.INFERENCE_DIR / 'weights'

    artifacts = utils.load_artifacts(model_artifacts)
    context['artifacts'] = artifacts

    model = utils.get_model(artifacts['token_size'], artifacts['pe_max_len'], artifacts['num_layers'],
                            artifacts['d_model'], artifacts['n_heads'], artifacts['d_ff'], artifacts['dropout'],
                            device, weights=Path(weights_path))
    model.eval()


def construct_response(handler: Callable[..., Awaitable[Dict]]) -> Callable[..., Awaitable[Dict]]:
    """ Construct a JSON response for an endpoint's results. """

    @wraps(handler)  # TODO figure out
    async def wrap(request: Request, *args, **kwargs) -> Dict:
        response = await handler(request, *args, **kwargs)

        response['method'] = request.method
        response['timestamp'] = datetime.now().isoformat()
        response['url'] = request.url._url

        return response

    return wrap


@api.get('/', tags=['General'])
@construct_response
async def index(request: Request) -> Dict:
    return {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK
    }


@api.get('/params', tags=['Parameters'])
@construct_response
async def all_parameters(request: Request) -> Dict:
    """ Get a specific parameter's value used for a run. """

    global context

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        'artifacts': context['artifacts']
    }

    return response


@api.get('/params/{param}', tags=['Parameters'])
@construct_response
async def parameters(request: Request, param: str) -> Dict:
    """ Get a specific parameter's value used for a run. """

    global context

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        param: context['artifacts'].get(param, 'Not found')
    }

    return response


# TODO noised data
@api.post('/zread', tags=['Prediction'], response_model=PredictResponse)
@construct_response
async def zread(request: Request, payload: PredictPayload) -> Dict:
    global model, device

    columns = utils.create_noisy_columns(payload.data, payload.min_noise, payload.max_noise)
    src = utils.columns_to_tensor(columns, device)

    chains = await utils.async_beam_search(src, model, payload.beam_width, device)
    chains = [(utils.visualize_target(tgt, payload.delimiter), prob) for tgt, prob in chains]

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        'columns': utils.visualize_columns(src, payload.delimiter),
        'chains': chains
    }

    return response


@api.post('/interactive_zread', tags=['Prediction'], response_model=InteractiveResponse)
@construct_response
async def interactive_zread(request: Request, payload: PredictPayload) -> Dict:
    global context

    identifier = str(uuid.uuid4())
    job_queue = asyncio.Queue()
    context['jobs'][identifier] = job_queue

    columns = utils.create_noisy_columns(payload.data, payload.min_noise, payload.max_noise)
    src = utils.columns_to_tensor(columns, device)

    asyncio.create_task(utils.async_beam_search(src, model, payload.beam_width, device,
                                                utils.api_interactive_loop(job_queue, payload.delimiter)))
    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        'identifier': identifier,
        'size': len(columns)
    }

    return response


@api.get('/status/{identifier}', tags=['Prediction'], response_model=JobStatus)
@construct_response
async def status(request: Request, identifier: str) -> Dict:
    global context

    if identifier in context['jobs']:
        job_status = await context['jobs'][identifier].get()

        if job_status is None:
            context['jobs'].pop(identifier)
            job_status = 'Job is completed'

    else:
        job_status = 'Not found'

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        'job_status': job_status
    }

    return response


@cli.command()
def run(host: str = Option('localhost', help='Bind socket to this host'),
        port: int = Option(5001, help='Bind socket to this port'),
        log_level: str = Option('info', help='Log level'),
        reload: bool = Option(False, help='Enable auto-reload'),
        workers: int = Option(1, help='Number of worker processes')
        ) -> None:
    """ Run uvicorn server """

    uvicorn.run('zreaderapi:api', host=host, port=port, log_level=log_level, reload=reload, workers=workers)


if __name__ == '__main__':
    cli()
