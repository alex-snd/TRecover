import asyncio
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from typer import Typer, Option

import config
import utils
from model import ZReader

cli = Typer(name='ZreaderAPI')
api = FastAPI(title='ZreaderAPI', description='Description will be here', version='0.1')

model: ZReader  # declare model for further initialization on server startup


@api.get('/')
async def root() -> dict:
    return {'ZreaderAPI': 'OK'}


@api.get('/sleep/{sec}')
async def sleep(sec: int) -> dict:
    global model

    await asyncio.sleep(sec)

    return {'ZreaderAPI': f'Sleep for: {sec}', 'd_ff': model.d_ff}


@api.on_event("startup")
async def load_model():
    global model

    model_artifacts = config.INFERENCE_DIR / 'artifacts.json'
    weights_path = config.INFERENCE_DIR / 'weights'

    artifacts = utils.load_artifacts(model_artifacts)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    model = utils.get_model(artifacts['token_size'], artifacts['pe_max_len'], artifacts['num_layers'],
                            artifacts['d_model'], artifacts['n_heads'], artifacts['d_ff'], artifacts['dropout'],
                            device, weights=Path(weights_path))
    model.eval()


@cli.command()
def run(host: str = Option('localhost', help='Bind socket to this host'),
        port: int = Option(5000, help='Bind socket to this port'),
        log_level: str = Option('info', help='Log level'),
        reload: bool = Option(False, help='Enable auto-reload'),
        workers: int = Option(1, help='Number of worker processes')) -> None:
    """ Run uvicorn server """

    uvicorn.run('zreaderapi:api', host=host, port=port, log_level=log_level, reload=reload, workers=workers)


if __name__ == '__main__':
    cli()
