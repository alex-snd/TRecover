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

queue = asyncio.Queue()  # needed to hand over some params to uvicorn server
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
    global queue, model

    if queue.empty():
        model_artifacts = config.INFERENCE_DIR / 'artifacts.json'
        weights_path = config.INFERENCE_DIR / 'weights'
        cuda = True
        gpu_id = 0
    else:
        model_artifacts = await queue.get()
        weights_path = await queue.get()
        cuda = await queue.get()
        gpu_id = await queue.get()

    artifacts = utils.load_artifacts(model_artifacts)
    device = torch.device(f'cuda:{gpu_id}' if cuda and torch.cuda.is_available() else 'cpu')

    model = utils.get_model(artifacts['token_size'], artifacts['pe_max_len'], artifacts['num_layers'],
                            artifacts['d_model'], artifacts['n_heads'], artifacts['d_ff'], artifacts['dropout'],
                            device, weights=Path(weights_path))
    model.eval()


async def serve(model_artifacts: str, weights_path: str, cuda: bool, gpu_id: int,
                host: str, port: int, log_level: str, reload: bool, workers: int) -> None:
    if reload or workers > 1 or model_artifacts is None or weights_path is None:
        # support reload and workers options
        uvicorn.run('zreaderapi:api', host=host, port=port, log_level=log_level, reload=reload, workers=workers)

    else:
        global queue, api

        await queue.put(Path(model_artifacts))
        await queue.put(Path(weights_path))
        await queue.put(cuda)
        await queue.put(gpu_id)

        server_config = uvicorn.Config(app=api, host=host, port=port, log_level=log_level, reload=reload,
                                       workers=workers)

        # doesn't support reload and workers options
        await uvicorn.Server(server_config).serve()


# TODO separate this command
@cli.command()
def run(model_artifacts: str = Option(None, help='Path to model artifacts json file'),
        weights_path: str = Option(None, help='Path to model weights'),
        cuda: bool = Option(True, help='CUDA enabled'),
        gpu_id: int = Option(0, help='GPU id'),
        host: str = Option('localhost', help='Bind socket to this host'),
        port: int = Option(5000, help='Bind socket to this port'),
        log_level: str = Option('info', help='Log level'),
        reload: bool = Option(False, help='Enable auto-reload'),
        workers: int = Option(1, help='Number of worker processes')) -> None:
    """ Run uvicorn server """

    asyncio.run(
        serve(model_artifacts, weights_path, cuda, gpu_id, host, port, log_level, reload, workers))

    # uvicorn.run('zreaderapi:api', host=host, port=port, log_level=log_level, reload=reload, workers=workers)


if __name__ == '__main__':
    cli()
