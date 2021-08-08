from pydantic import BaseModel, validator
from data import Collate


class PredictPayload(BaseModel):
    data: str
    min_noise: int
    max_noise: int
    beam_width: int
    delimiter: str = ''

    @validator('data')
    def data_validator(cls, data: str) -> str or None:
        if not data:
            ValueError(f'Data for zread must contain at least one character')

        return data

    @validator('min_noise')
    def min_noise_validator(cls, min_noise: int) -> int or None:
        if not 0 <= min_noise <= len(Collate.num_to_alphabet):
            raise ValueError(f'Minimum noise range must be in between 0 and {len(Collate.num_to_alphabet)}')

        return min_noise

    @validator('max_noise')
    def max_noise_validator(cls, max_noise: int, values: dict) -> int or None:
        if not 0 <= max_noise <= len(Collate.num_to_alphabet):
            raise ValueError(f'Maximum noise range must be in between 0 and {len(Collate.num_to_alphabet)}')

        if values['min_noise'] >= max_noise:
            raise ValueError('Maximum noise range must be grater than minimum noise range')

        return max_noise

    @validator('beam_width')
    def beam_width_validator(cls, beam_width: int) -> int or None:
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


class PredictResponse(BaseModel):
    message: str
    method: str
    status_code: int
    timestamp: str
    url: str
    data: dict

    class Config:
        """ PredictPayload example for API documentation"""

        schema_extra = {
            'example': {
                "message": "OK",
                "method": "POST",
                "status_code": 200,
                "timestamp": "2021-08-06T09:14:43.832328",
                "url": "http://localhost:5001/zread",
                "data": {
                    "columns": "aecdoijeaacsndlheccaeeeuhcagifndhherdrccadtacdeelccecggaaalemhbkiebfiineldedibekdjganda"
                               "eafdedgcdfeaceindfjbjebaagacc\njgieqplgfbjuseruheoufppxidlnsntglifsfyeegeuldhfkrpdghiho"
                               "icohomsomiiksjrvmkkjxninrqleoebobliwlqpphmikklqilqmonceqtgig\nlsmivvpkmhowxitwmnrwntryl"
                               "enswoxhtphto kmmf oiwplwthhnomvtgtiqn strtrvvvyssqnzpnottmiruqqfpjxzuwtmrlrwrtzown oujz"
                               "xhrn\nq pkyyuz ryx oz n   yvu w  t   iu nwr  sts so  wz x qqnzzp pzv yvs uwy  vzrs  prx"
                               "vzxs v grv   zx  p   y t s s p  kvs\n  q   x         p    z         o  z t   v   r     "
                               "    w    z x   t z z        xvy   u    t                  x    v w\n",
                    "zread": [
                        "aspeoplearoundthecountrywentintothestreetstocheertheconvictionsstriksinessesinportlandboardedu"
                        "ptheirwindowsonceagain",
                        -5.332423858530319
                    ],
                    "chains": [
                        [
                            "aspeoplearoundthecountrywentintothestreetstocheertheconvictionsstriksinessesinportlandboar"
                            "deduptheirwindowsonceagain",
                            -5.332423858530319
                        ],
                        [
                            "aspeoplearoundthecountrywentintothestreetstocheertheconvictionsomebusinessesinportlandboar"
                            "deduptheirwindowsonceagain",
                            -5.484715415203709
                        ],
                        [
                            "lwpeoplearoundthecountrywentintothestreetstocheertheconvictionsstriksinessesinportlandboar"
                            "deduptheirwindowsonceagain",
                            -6.023200242937946
                        ],
                        [
                            "aspeoplearoundthecountrywentintothestreetstoowelltheconvictionsstriksinessesinportlandboar"
                            "deduptheirwindowsonceagain",
                            -6.484925642163489
                        ],
                        [
                            "aspeoplearoundthecountrywentintothestreetstocheertheconvictionsstriksinessesinportlandbea"
                            "rdeduptheirwindowsonceagain",
                            -7.280832245267902
                        ]
                    ]
                }
            }
        }


class InteractiveResponse(BaseModel):
    message: str
    method: str
    status_code: int
    timestamp: str
    url: str
    data: dict

    class Config:
        """ InteractiveResponse example for API documentation"""

        schema_extra = {
            'example': {
                "message": "OK",
                "method": "POST",
                "status_code": 200,
                "timestamp": "2021-08-06T09:14:43.832328",
                "url": "http://localhost:5001/zread",
                "data": {
                    "identifier": '8c82d3f0-f60d-474e-add1-22441aa006e4'
                }
            }
        }
