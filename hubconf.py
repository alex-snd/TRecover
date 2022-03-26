""" Configuration for torch hub usage """

import json
from urllib.request import urlopen

import torch

from zreader.model import ZReader

dependencies = ['torch', 'zreader']

checkpoint_urls = {
    'latest': {'model': 'https://github.com/alex-snd/ZReader/releases/download/v0.1.0/model.pt',
               'config': 'https://github.com/alex-snd/ZReader/releases/download/v0.1.0/config.json'},
    'v0.1.0': {'model': 'https://github.com/alex-snd/ZReader/releases/download/v0.1.0/model.pt',
               'config': 'https://github.com/alex-snd/ZReader/releases/download/v0.1.0/config.json'},
}


def zreader(version: str = 'latest') -> ZReader:
    with urlopen(checkpoint_urls[version]['config']) as url:
        config = json.loads(url.read().decode())

    model = ZReader(token_size=config['token_size'], pe_max_len=config['pe_max_len'],
                    num_layers=config['num_layers'], d_model=config['d_model'], n_heads=config['n_heads'],
                    d_ff=config['d_ff'], dropout=config['dropout'])

    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_urls['model'], progress=False))

    return model
