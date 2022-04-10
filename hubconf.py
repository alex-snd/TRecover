""" Configuration for torch hub usage """

from typing import Callable

import torch

dependencies = ['torch', 'trecover']


def add_available_versions_to_docstring(handler: Callable[..., Callable]) -> Callable:
    from trecover.config import var

    handler.__doc__ += '\n'.join([f'\t* {version}' for version in var.CHECKPOINT_URLS.keys()])

    return handler


@add_available_versions_to_docstring
def trecover(device: torch.device = torch.device('cpu'), version: str = 'latest'):
    """
    Load the TRecover model via torch.hub.

    Parameters
    ----------
    device : torch.device, default=torch.device('cpu')
        Device on which to allocate the model.
    version : str, default='latest'
        Model weights' version.

    Returns
    -------
    model : TRecover
        Model with specified weights' version.

    Examples
    --------
    Show the docstring with available versions for the TRecover model:
        >>> print(torch.hub.help(github='alex-snd/TRecover', model='trecover'))

    Load the TRecover model:
        >>> torch.hub.load('alex-snd/TRecover', model='trecover', device=torch.device('cpu'), version='latest')

    Available Versions
    ------------------
    """

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
