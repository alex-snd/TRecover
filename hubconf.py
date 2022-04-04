""" Configuration for torch hub usage """

import torch

dependencies = ['torch', 'zreader']

checkpoint_urls = {
    'latest': {'model': 'https://github.com/alex-snd/ZReader/releases/download/v0.1.0/model.pt',
               'config': 'https://github.com/alex-snd/ZReader/releases/download/v0.1.0/config.json'},
    'v0.1.0': {'model': 'https://github.com/alex-snd/ZReader/releases/download/v0.1.0/model.pt',
               'config': 'https://github.com/alex-snd/ZReader/releases/download/v0.1.0/config.json'},
}


def zreader(device: torch.device = torch.device('cpu'), version: str = 'latest'):
    # TODO help docstring

    import json
    from urllib.request import urlopen
    from zreader.model import ZReader

    if version not in checkpoint_urls:
        version = 'latest'

    with urlopen(checkpoint_urls[version]['config']) as url:
        config = json.loads(url.read().decode())

    model = ZReader(token_size=config['token_size'], pe_max_len=config['pe_max_len'],
                    num_layers=config['num_layers'], d_model=config['d_model'], n_heads=config['n_heads'],
                    d_ff=config['d_ff'], dropout=config['dropout'])

    model.load_state_dict(torch.hub.load_state_dict_from_url(url=checkpoint_urls[version]['model'],
                                                             progress=False,
                                                             map_location=device))

    return model
