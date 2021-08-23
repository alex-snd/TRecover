from typing import Union, Optional, List, Dict, Tuple

from pydantic import BaseModel, validator

from data import Collate


class PredictPayload(BaseModel):
    data: str
    min_noise: int
    max_noise: int
    beam_width: int
    delimiter: str = ''

    @validator('data')
    def data_validator(cls, data: str) -> Optional[str]:
        if not data:
            ValueError(f'Data for zread must contain at least one character')

        return data

    @validator('min_noise')
    def min_noise_validator(cls, min_noise: int) -> Optional[int]:
        if not 0 <= min_noise <= len(Collate.num_to_alphabet):
            raise ValueError(f'Minimum noise range must be in between 0 and {len(Collate.num_to_alphabet)}')

        return min_noise

    @validator('max_noise')
    def max_noise_validator(cls, max_noise: int, values: Dict[str, Union[str, int]]) -> Optional[int]:
        if not 0 <= max_noise <= len(Collate.num_to_alphabet):
            raise ValueError(f'Maximum noise range must be in between 0 and {len(Collate.num_to_alphabet)}')

        if values['min_noise'] >= max_noise:
            raise ValueError('Maximum noise range must be grater than minimum noise range')

        return max_noise

    @validator('beam_width')
    def beam_width_validator(cls, beam_width: int) -> Optional[int]:
        if not 1 <= beam_width <= len(Collate.num_to_alphabet):
            raise ValueError(f'Beam width must be in between 1 and {len(Collate.num_to_alphabet)}')

        return beam_width

    class Config:
        """ PredictPayload example for API documentation"""

        schema_extra = {
            'example': {
                'data': 'As people around the country went into the streets to cheer the conviction, some businesses '
                        'in Portland boarded up their windows once again.',
                'min_noise': 2,
                'max_noise': 5,
                'beam_width': 5,
                'delimiter': ''
            }
        }


class BaseResponse(BaseModel):
    message: str
    method: str
    status_code: int
    timestamp: str
    url: str


class PredictResponse(BaseResponse):
    columns: str
    chains: List[Tuple[str, float]]

    class Config:
        """ PredictPayload example for API documentation"""

        schema_extra = {
            'example': {
                "message": "OK",
                "method": "POST",
                "status_code": 200,
                "timestamp": "2021-08-08T17:34:42.390414",
                "url": "http://localhost:5001/zread",
                "columns": "adheclceariagdfceclmfmahaecebfjbddefbcdabspeceaeloheckhhfafbkjebcebbdimesdeabcplfkfaddbiard"
                           "eabdahearagmdhjcfdbeagaic\nflononfphsohmslegoopmtcvppnocjtdehqmeeeefutorhckrtjmfojmicrionsi"
                           "frvpeoniugfkhlropljhitcjjvgldcgikmhtkinjotrnjcickjkk\nzsptxplwlwwinwthl vunxryw ptinvlqwtst"
                           "omjtwvpxxel uwnhwntrftqtpyomswth omwhklin xrtlsnuuoquoogtplmuivqyw rwsokrkqmnln\n   z wo r "
                           " utz ys   qz  y   s  ot w  rrow  q  fs w  x yv j w w tywzul  n owst   x sts  wy   quqts rur"
                           "   z v n qr rr \n      s    x        x          qz      rx  v  px       w q     y    s  x s"
                           "        x         x   u v w       u s  xx \n",
                "chains": [
                    [
                        "aspeoplearoundthecountrywentintothestreetstoreallthecontictionsofebusinessesinportlandboardedu"
                        "ptheirwindowsonceagain",
                        -6.104937998053401
                    ],
                    [
                        "aspeoplearoundthecountrywentintothestreetstoreallthecontictionsofsbusinessesinportlandboardedu"
                        "ptheirwindowsonceagain",
                        -6.2104413745624925
                    ],
                    [
                        "aspeoplearoundthecountrywentintothestreetstoreallthecontictionsofsbusinessesinportsandboardedu"
                        "ptheirwindowsonceagain",
                        -6.6121470403370495
                    ],
                    [
                        "aspeoplearoundthecountrywentintothestreetstoreallthecontictionsofebusinessesinportsandboardedu"
                        "ptheirwindowsonceagain",
                        -6.840312714927222
                    ],
                    [
                        "aspeoplearoundthecountrywentintothestreetstoreallthecontictionsofebusinessesinportlandboardedu"
                        "ptheirwindowconcergain",
                        -7.29570330515071
                    ]
                ]
            }
        }


class InteractiveResponse(BaseResponse):
    identifier: str
    size: int

    class Config:
        """ InteractiveResponse example for API documentation"""

        schema_extra = {
            'example': {
                "message": "OK",
                "method": "POST",
                "status_code": 200,
                "timestamp": "2021-08-08T17:33:27.512374",
                "url": "http://localhost:5001/interactive_zread",
                "identifier": "5de4506a-26a1-4a18-9ac1-5d88cf2d480e",
                "size": 116
            }
        }


class JobStatus(BaseResponse):
    job_status: Union[List[Tuple[str, float]], str]

    class Config:
        """ JobStatus example for API documentation"""

        schema_extra = {
            'example': {
                "message": "OK",
                "method": "GET",
                "status_code": 200,
                "timestamp": "2021-08-08T17:43:35.726135",
                "url": "http://localhost:5001/status/edc256a8-5c9d-4256-8c8e-acd3b69912ba",
                "job_status": [
                    [
                        "aspeoplearoundthecountry",
                        -1.9372545908522625
                    ],
                    [
                        "aspeoplearoundthecountry",
                        -2.5602809409524525
                    ],
                    [
                        "aspeoplearoundthecountry",
                        -2.9490952406658835
                    ],
                    [
                        "aspeoplearoundthecountry",
                        -3.245900469544722
                    ],
                    [
                        "aspeoplearoundthecoastof",
                        -3.591321390164012
                    ]
                ]
            }
        }
