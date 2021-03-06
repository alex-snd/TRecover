from argparse import Namespace
from typing import List, Dict, Tuple, Optional

import hivemind
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import BytesWithPublicKey, SchemaValidator
from hivemind.dht.validation import RecordValidatorBase
from hivemind.utils.networking import choose_ip_address
from multiaddr import Multiaddr
from pydantic import BaseModel, StrictFloat, confloat, conint

from trecover.config.log import project_console


class LocalMetrics(BaseModel):
    loss: StrictFloat
    accuracy: StrictFloat
    lr: StrictFloat
    min_noise: conint(ge=0, le=26, strict=True)
    max_noise: conint(ge=0, le=26, strict=True)
    samples_per_second: confloat(ge=0.0, strict=True)
    samples_accumulated: conint(ge=0, strict=True)
    mini_steps: conint(ge=0, strict=True)
    step: conint(ge=0, strict=True)


class GlobalMetrics(BaseModel):
    loss: StrictFloat
    accuracy: confloat(ge=0, le=1)
    lr: StrictFloat
    min_noise: conint(ge=0, le=26, strict=True)
    max_noise: conint(ge=0, le=26, strict=True)
    samples_per_second: confloat(ge=0.0, strict=True)
    samples_accumulated: conint(ge=0, strict=True)
    alive_peers: conint(ge=0, strict=True)


class MetricSchema(BaseModel):
    metrics: Dict[BytesWithPublicKey, LocalMetrics]


class DHTManager:
    def __init__(self, args: Namespace, use_init_peers: bool = True):
        self.args = args
        self.use_init_peers = use_init_peers
        self.validators, self.local_public_key = self.make_validators()

        if args.initial_peers and self.use_init_peers:
            project_console.print(f'Found {len(args.initial_peers)} initial peers: ', style='bright_blue')
            project_console.print_json(data=args.initial_peers)

        self.dht = hivemind.DHT(
            start=True,
            initial_peers=args.initial_peers if self.use_init_peers else None,
            client_mode=args.client_mode,
            host_maddrs=args.host_maddrs,
            announce_maddrs=args.announce_maddrs,
            use_ipfs=args.use_ipfs,
            record_validators=self.validators,
            identity_path=args.identity_path,
        )

        self.visible_maddrs = self.dht.get_visible_maddrs()

        if args.client_mode:
            project_console.print(f'Created client mode peer with peer_id={self.dht.peer_id}', style='bright_blue')
        elif self.use_init_peers:
            if initial_peers := self.get_initial_peers(as_str=True):
                project_console.print(f'To connect other peers to this one over the Internet, use '
                                      f'--initial-peers {initial_peers}', style='bright_blue')

            project_console.print(f'Full list of visible multi addresses: ', style='bright_blue')
            project_console.print_json(data=[str(addr) for addr in self.visible_maddrs])

    def get_initial_peers(self, as_str: bool = False) -> Optional[List[str]]:
        if self.args.use_ipfs:
            peers = [f'/p2p/{addr}' for addr in {addr['p2p'] for addr in self.visible_maddrs}]

        elif available_ips := [Multiaddr(addr) for addr in self.visible_maddrs if 'ip4' in addr or 'ip6' in addr]:
            preferred_ip = choose_ip_address(available_ips)
            peers = [str(addr) for addr in self.visible_maddrs if preferred_ip in str(addr)]

        else:
            return None

        return ' '.join(peers) if as_str else peers or None

    def make_validators(self) -> Tuple[List[RecordValidatorBase], bytes]:
        signature_validator = RSASignatureValidator()
        validators = [SchemaValidator(MetricSchema, prefix=self.args.experiment_prefix), signature_validator]

        return validators, signature_validator.local_public_key
