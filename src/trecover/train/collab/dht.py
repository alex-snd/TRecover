import re
from argparse import Namespace
from typing import List, Dict, Tuple, Optional

import hivemind
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import BytesWithPublicKey, SchemaValidator
from hivemind.dht.validation import RecordValidatorBase
from hivemind.utils.networking import choose_ip_address
from multiaddr import Multiaddr
from pydantic import BaseModel, StrictFloat, StrictBool, confloat, conint
from speedtest import Speedtest, SpeedtestException

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


class OptimizerStatus(BaseModel):
    step: conint(ge=0, strict=True)
    client: StrictBool


class GlobalMetrics(BaseModel):
    loss: StrictFloat
    accuracy: confloat(ge=0, le=1)
    lr: StrictFloat
    min_noise: conint(ge=0, le=26, strict=True)
    max_noise: conint(ge=0, le=26, strict=True)
    samples_per_second: confloat(ge=0.0, strict=True)
    samples_accumulated: conint(ge=0, strict=True)
    alive_peers: conint(ge=0, strict=True)


class MetricsSchema(BaseModel):
    metrics: Dict[BytesWithPublicKey, LocalMetrics]


class StatusSchema(BaseModel):
    metrics: Dict[BytesWithPublicKey, OptimizerStatus]


class DHTManager:
    def __init__(self, args: Namespace, use_init_peers: bool = True):
        self.args = args
        self.use_init_peers = use_init_peers
        self.validators, self.local_public_key = self.make_validators()
        self._ip = None

        if not args.client_mode and self.ip and args.announce_maddrs is None:
            args.announce_maddrs = [
                re.sub(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', self.ip, host_maddr) for host_maddr in args.host_maddrs
            ]

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

    @property
    def ip(self) -> Optional[str]:
        if self._ip is None:
            try:
                self._ip = Speedtest().config['client']['ip']
            except SpeedtestException:
                return None

        return self._ip

    def make_validators(self) -> Tuple[List[RecordValidatorBase], bytes]:
        signature_validator = RSASignatureValidator()
        validators = [
            SchemaValidator(MetricsSchema, prefix=self.args.experiment_prefix),
            SchemaValidator(StatusSchema, prefix=self.args.experiment_prefix),
            signature_validator
        ]

        return validators, signature_validator.local_public_key
