import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent.parent.absolute()

CONFIG_DIR = BASE_DIR / 'src' / 'config'
INFERENCE_DIR = BASE_DIR / 'inference'
LOGS_DIR = BASE_DIR / 'logs'

LOGS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / '.env')

BROKER_PORT = int(os.getenv('BROKER_PORT', default=5672))  # TODO change docker ports
BROKER_UI_PORT = int(os.getenv('BROKER_UI_PORT', default=15672))
BACKEND_PORT = int(os.getenv('BACKEND_PORT', default=6379))
CELERY_BROKER = f"{os.getenv('CELERY_BROKER', default='pyamqp://guest@localhost')}:{BROKER_PORT}"
CELERY_BACKEND = f"{os.getenv('CELERY_BACKEND', default='redis://localhost')}:{BACKEND_PORT}"
CELERY_WORKERS = int(os.getenv('CELERY_WORKERS', default=1))

FASTAPI_HOST = os.getenv('FASTAPI_HOST', default='localhost')
FASTAPI_PORT = int(os.getenv('FASTAPI_PORT', default=8001))
FASTAPI_WORKERS = int(os.getenv('FASTAPI_WORKERS', default=1))
FASTAPI_URL = f'http://{FASTAPI_HOST}:{FASTAPI_PORT}'

STREAMLIT_HOST = os.getenv('STREAMLIT_HOST', default='localhost')
STREAMLIT_PORT = int(os.getenv('STREAMLIT_PORT', default=8000))

INFERENCE_PARAMS_PATH = Path(os.getenv('INFERENCE_PARAMS_PATH', default=INFERENCE_DIR / 'params.json'))
INFERENCE_WEIGHTS_PATH = Path(os.getenv('INFERENCE_WEIGHTS_PATH', default=INFERENCE_DIR / 'z_reader.pt'))
CUDA = False if os.getenv('CUDA', default='').lower() == 'false' else True
MAX_NOISE = int(os.getenv('MAX_NOISE', default=13))

DASHBOARD_PID = CONFIG_DIR / 'dashboard.pid'
API_PID = CONFIG_DIR / 'api.pid'
WORKER_PID = CONFIG_DIR / 'worker.pid'
BROKER_ID = 'zreader_broker'
BACKEND_ID = 'zreader_backend'

BROKER_IMAGE = os.getenv('BROKER_IMAGE', default='rabbitmq:3.9.8-management')
BACKEND_IMAGE = os.getenv('BACKEND_IMAGE', default='redis:6.2')

ALPHABET = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'v',
            'u', 'w', 'x', 'y', 'z'}

ALPHABET2NUM = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10,
                'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'v': 20,
                'u': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}

NUM2ALPHABET = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k',
                11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'v',
                21: 'u', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
