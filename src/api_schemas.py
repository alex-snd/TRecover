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
        """ Needed to declare examples of data the app can receive for API documentation"""

        schema_extra = {
            'example': {
                'texts': 'Hi! my name is Sasha.'
            }
        }
