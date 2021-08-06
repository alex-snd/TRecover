from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Request
from typer import Typer, Option

import config
import utils
from api_schemas import PredictPayload, PredictResponse
from model import ZReader

cli = Typer(name='ZreaderAPI', epilog='Description will be here')
api = FastAPI(title='ZreaderAPI', description='Description will be here', version='0.1')

model: ZReader  # declare model for further initialization on server startup
artifacts: dict  # mlflow artifacts
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')


@api.on_event('startup')
async def load_model():
    global model, artifacts, device

    model_artifacts = config.INFERENCE_DIR / 'artifacts.json'
    weights_path = config.INFERENCE_DIR / 'weights'

    artifacts = utils.load_artifacts(model_artifacts)

    model = utils.get_model(artifacts['token_size'], artifacts['pe_max_len'], artifacts['num_layers'],
                            artifacts['d_model'], artifacts['n_heads'], artifacts['d_ff'], artifacts['dropout'],
                            device, weights=Path(weights_path))
    model.eval()


def construct_response(handler: eval) -> object:
    """ Construct a JSON response for an endpoint's results. """

    @wraps(handler)  # TODO figure out
    async def wrap(request: Request, *args, **kwargs):
        results = await handler(request, *args, **kwargs)

        response = {
            'message': results['message'],
            'method': request.method,
            'status_code': results['status_code'],
            'timestamp': datetime.now().isoformat(),
            'url': request.url._url,
        }

        if 'data' in results:
            response['data'] = results['data']

        return response

    return wrap


@api.get('/', tags=['General'])
@construct_response
async def index(request: Request) -> dict:
    return {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK
    }


@api.get('/params/{param}', tags=['Parameters'])
@construct_response
async def parameters(request: Request, param: str) -> dict:
    """ Get a specific parameter's value used for a run. """

    global artifacts

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        'data': {
            param: artifacts.get(param, 'Not found')
        }
    }

    return response


@api.post("/zread", tags=["Prediction"], response_model=PredictResponse)
@construct_response
async def predict(request: Request, payload: PredictPayload) -> dict:
    global model, device

    columns = utils.create_noisy_columns(payload.data, payload.min_noise, payload.max_noise)
    src = utils.columns_to_tensor(columns, device)

    chains = utils.beam_search(src, model, payload.beam_width, device)
    chains = [(utils.visualize_target(tgt, payload.delimiter), prob) for tgt, prob in chains]

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        'data': {
            'columns': utils.visualize_columns(src, payload.delimiter),
            'zread': chains[0],
            'chains': chains
        }
    }

    return response


@cli.command()
def run(host: str = Option('localhost', help='Bind socket to this host'),
        port: int = Option(5001, help='Bind socket to this port'),
        log_level: str = Option('info', help='Log level'),
        reload: bool = Option(False, help='Enable auto-reload'),
        workers: int = Option(1, help='Number of worker processes')) -> None:
    """ Run uvicorn server """

    uvicorn.run('zreaderapi:api', host=host, port=port, log_level=log_level, reload=reload, workers=workers)


if __name__ == '__main__':
    cli()
