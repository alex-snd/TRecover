import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent.parent.absolute()

CONFIG_DIR = BASE_DIR / 'config'
INFERENCE_DIR = BASE_DIR / 'inference'
EXAMPLES_DIR = BASE_DIR / 'examples'

load_dotenv(BASE_DIR / '.env')

CELERY_BROKER = os.getenv('CELERY_BROKER', default='pyamqp://guest@localhost')
CELERY_BACKEND = os.getenv('CELERY_BACKEND', default='redis://localhost:6379')
CELERY_WORKERS = int(os.getenv('CELERY_WORKERS', default=1))
FASTAPI_HOST = os.getenv('FASTAPI_HOST', default='localhost')
FASTAPI_PORT = int(os.getenv('FASTAPI_PORT', default=8001))
FASTAPI_WORKERS = int(os.getenv('FASTAPI_WORKERS', default=1))
FASTAPI_URL = f'http://{FASTAPI_HOST}:{FASTAPI_PORT}'
INFERENCE_PARAMS_PATH = Path(os.getenv('INFERENCE_PARAMS_PATH', default=INFERENCE_DIR / 'params.json'))
INFERENCE_WEIGHTS_PATH = Path(os.getenv('INFERENCE_WEIGHTS_PATH', default=INFERENCE_DIR / 'z_reader.pt'))
CUDA = False if os.getenv('CUDA', default='').lower() == 'false' else True
MAX_NOISE = int(os.getenv('MAX_NOISE', default=13))

ALPHABET = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'v',
            'u', 'w', 'x', 'y', 'z'}

ALPHABET2NUM = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10,
                'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'v': 20,
                'u': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}

NUM2ALPHABET = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k',
                11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'v',
                21: 'u', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
