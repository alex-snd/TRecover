""" Configuration for torch hub usage """

import torch

dependencies = ['torch', 'zreader']


def zreader(device: torch.device = torch.device('cpu'), version: str = 'latest'):
    # TODO help docstring

    import json
    from urllib.request import urlopen
    from zreader.config import var
    from zreader.model import ZReader

    if version not in var.CHECKPOINT_URLS:
        version = 'latest'

    with urlopen(var.CHECKPOINT_URLS[version]['config']) as url:
        config = json.loads(url.read().decode())

    model = ZReader(token_size=config['token_size'], pe_max_len=config['pe_max_len'],
                    num_layers=config['num_layers'], d_model=config['d_model'], n_heads=config['n_heads'],
                    d_ff=config['d_ff'], dropout=config['dropout'])

    model.load_state_dict(torch.hub.load_state_dict_from_url(url=var.CHECKPOINT_URLS[version]['model'],
                                                             progress=False,
                                                             map_location=device))

    return model
