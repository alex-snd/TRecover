from pydantic import BaseModel, validator
from data import Collate


class PredictPayload(BaseModel):
    data: str
    min_noise: int
    max_noise: int
    beam_width: int
    delimiter: str = ''

    @classmethod
    @validator('data')
    def data_validator(cls, value: str) -> str or None:
        if not value:
            ValueError(f'Data for zread must contain at least one character')

        return value

    @classmethod
    @validator('min_noise', 'max_noise')
    def noise_validator(cls, value: int) -> int or None:
        if not 0 <= value <= len(Collate.num_to_alphabet):
            raise ValueError(f'Noise range must be in between 0 and {len(Collate.num_to_alphabet)}')

        return value

    @classmethod
    @validator('min_noise', 'max_noise')
    def beam_width_validator(cls, value: int) -> int or None:
        if not 1 <= value <= len(Collate.num_to_alphabet):
            raise ValueError(f'Beam width must be in between 1 and {len(Collate.num_to_alphabet)}')

        return value

    class Config:
        """ Needed to declare examples of data the app can receive for API documentation"""

        schema_extra = {
            'example': {
                'texts': 'Hi! my name is Sasha.'
            }
        }
