from typing import Union, Optional, List, Dict, Tuple

from pydantic import BaseModel, validator

from data import Collate


class PredictPayload(BaseModel):
    data: List[str]
    beam_width: int
    delimiter: Optional[str] = ''

    @validator('data')
    def data_validator(cls, data: str) -> Optional[str]:
        if not data:
            ValueError(f'Data for zread must contain at least one character')

        return data

    @validator('beam_width')
    def beam_width_validator(cls, beam_width: int) -> Optional[int]:
        if not 1 <= beam_width <= len(Collate.num_to_alphabet):
            raise ValueError(f'Beam width must be in between 1 and {len(Collate.num_to_alphabet)}')

        return beam_width

    class Config:
        """ PredictPayload example for API documentation"""

        schema_extra = {
            'example': {
                'data': ['adin', 'scjz', 'pzxz', 'evft', 'odev', 'pkzc', 'lipss', 'eayh', 'aepqc', 'rvuqh', 'ozldx',
                         'ulgq', 'nabhr', 'dvvf', 'tmil', 'huiow', 'euzq', 'cdmh', 'ompzv', 'uguq', 'ntvqi', 'tctm',
                         'rwgi', 'yuonh', 'wkvr', 'efau', 'nrele', 'tbhxq', 'iiqd', 'nhwfy', 'tymio', 'onzpj', 'turho',
                         'hhgnd', 'ehmof', 'scpoi', 'tyqbn', 'roul', 'exoe', 'edwyc', 'thid', 'snqrp', 'tmmh', 'oejg',
                         'cjbaw', 'hgzmb', 'ezlwj', 'esxb', 'rria', 'taeo', 'hpnln', 'eeomf', 'cvxr', 'ofwy', 'nuon',
                         'vvex', 'ikbd', 'cmusb', 'tdnkb', 'iccd', 'ocay', 'nnxds', 'slbgd', 'oupap', 'mhip', 'ebtv',
                         'bipta', 'unazv', 'sccol', 'iovq', 'nxzev', 'ezaz', 'sakm', 'skls', 'eipvd', 'segz', 'inrlo',
                         'nzkug', 'puau', 'oolt', 'recs', 'tjot', 'lwds', 'assb', 'ntyzf', 'dpad', 'byep', 'owgol',
                         'aiwbb', 'ralk', 'dffos', 'ejfp', 'dwyug', 'uxucm', 'ppch', 'teznz', 'hpfbv', 'eaocf', 'iist',
                         'rdlyx', 'wseu', 'ivgg', 'nabi', 'dtuiw', 'oxzje', 'wderj', 'sawm', 'odpw', 'noqs', 'cimnf',
                         'eukjx', 'aszy', 'ghmiq', 'ajcsd', 'ipwgf', 'nhth', 'tpvfp', 'hnede', 'aclb', 'tteuq', 'nbvi',
                         'isgc', 'gejbz', 'hufnl', 'tggvn', 'aahfn', 'skkxl', 'mmwq', 'anqgw', 'lrvji', 'lker', 'gvqt',
                         'rdcd', 'obvt', 'ussht', 'piwb', 'ozbn', 'fyig', 'aczx', 'clak', 'tgnxi', 'iieyi', 'vqyqo',
                         'ijgp', 'shfth', 'tnbeg', 'sbuzx', 'wwnja', 'eguhn', 'aneai', 'rpsl', 'izut', 'njixa', 'grgh',
                         'bkypw', 'lliv', 'azeob', 'cojl', 'kmqnd', 'ahpq', 'pckw', 'phvpa', 'rgqv', 'opqby', 'aiqsp',
                         'clirf', 'hlrp', 'eztn', 'dgek', 'azubn', 'gnskd', 'rajo', 'okgv', 'ugcp', 'pilf', 'oxbx',
                         'fwqq', 'jsmf', 'okgw', 'uvjyg', 'rfio', 'nbtcy', 'aryp', 'ledgj', 'inex', 'sdqmm', 'tcmtw',
                         'soqh', 'tquu', 'hrfi', 'rsddp', 'ewfye', 'aumxj', 'tssui', 'ecgd', 'nunfp', 'isfka', 'ncvx',
                         'gqxi', 'typc', 'okcdt', 'sgcaq', 'mbcwd', 'angb', 'svyl', 'hjmk', 'tlcgd', 'hpka', 'efule',
                         'cbjou', 'agjjr', 'mziur', 'emoqh', 'rajte', 'ahoz', 'shaxe', 'okeyj', 'fgwnq', 'twojw',
                         'hcyfc', 'ojehd', 'soij', 'ekhpc', 'wbyq', 'hdfi', 'opyas', 'rzba', 'ediml', 'mvwys', 'avtmz',
                         'ilvj', 'nefy', 'eycng', 'dqdk', 'oihcj', 'nzuw', 'srrye', 'ccsi', 'emhnz', 'neosn', 'ewzki'],
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


class JobResponse(BaseResponse):
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


class PredictResponse(BaseResponse):
    chains: Union[List[Tuple[str, float]], str]

    class Config:
        """ PredictPayload example for API documentation"""

        schema_extra = {
            'example': {
                "message": "OK",
                "method": "POST",
                "status_code": 200,
                "timestamp": "2021-08-08T17:34:42.390414",
                "url": "http://localhost:5001/zread",
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
