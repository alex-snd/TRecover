""" Configuration for torch hub usage """

import torch

dependencies = ['torch', 'trecover']


def trecover(device: torch.device = torch.device('cpu'), version: str = 'latest'):
    # TODO help docstring

    import json
    from urllib.request import urlopen
    from trecover.config import var
    from trecover.model import TRecover

    if version not in var.CHECKPOINT_URLS:
        version = 'latest'

    with urlopen(var.CHECKPOINT_URLS[version]['config']) as url:
        config = json.loads(url.read().decode())

    model = TRecover(token_size=config['token_size'], pe_max_len=config['pe_max_len'],
                     num_layers=config['num_layers'], d_model=config['d_model'], n_heads=config['n_heads'],
                     d_ff=config['d_ff'], dropout=config['dropout'])

    model.load_state_dict(torch.hub.load_state_dict_from_url(url=var.CHECKPOINT_URLS[version]['model'],
                                                             progress=False,
                                                             map_location=device))

    return model
