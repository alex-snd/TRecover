from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch

from trecover.config import var, exp_var


def sync_base_args(args: Namespace) -> Namespace:
    if not args.sync_args:
        return args

    base_args = torch.hub.load('alex-snd/TRecover', 'collab_args', force_reload=True, verbose=False)

    if experiment_prefix := base_args.get('experiment_prefix'):
        args.experiment_prefix = experiment_prefix
    if target_batch_size := base_args.get('target_batch_size'):
        args.target_batch_size = target_batch_size
    if (min_noise := base_args.get('min_noise')) or min_noise == 0:
        args.min_noise = min_noise
    if (max_noise := base_args.get('max_noise')) or max_noise == 0:
        args.max_noise = max_noise
    if initial_peers := base_args.get('initial_peers'):
        if args.initial_peers:
            args.initial_peers.extend([peer_id for peer_id in initial_peers if peer_id not in args.initial_peers])
        else:
            args.initial_peers = initial_peers

    return args


def get_sync_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('Synchronization arguments', add_help=add_help)

    parser.add_argument('--sync-period', default=5, type=int,
                        help='Period (in collaborative steps) for arguments resynchronization')
    parser.add_argument('--sync-args', action='store_true',
                        help='Sync base collaborative arguments with torch.hub')

    return parser


def get_model_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('Model arguments', add_help=add_help)

    parser.add_argument('--token-size', default=len(var.ALPHABET), type=int,
                        help='Token size')
    parser.add_argument('--pe-max-len', default=512, type=int,
                        help='Positional encoding max length')
    parser.add_argument('--n-layers', default=8, type=int,
                        help='Number of encoder and decoder blocks')
    parser.add_argument('--d-model', default=768, type=int,
                        help='Model dimension - number of expected features in the encoder (decoder) input')
    parser.add_argument('--n-heads', default=16, type=int,
                        help='Number of encoder and decoder attention heads')
    parser.add_argument('--d-ff', default=768 * 4, type=int,
                        help='Dimension of the feedforward layer')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout range')

    return parser


def get_data_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('Data arguments', add_help=add_help)

    parser.add_argument('--train-files', default=exp_var.TRAIN_DATA, type=str,
                        help='Path to train files folder')
    parser.add_argument('--val-files', default=exp_var.VAL_DATA, type=str,
                        help='Path to validation files folder')
    parser.add_argument('--vis-files', default=exp_var.VIS_DATA, type=str,
                        help='Path to visualization files folder')
    parser.add_argument('--test-files', default=exp_var.VIS_DATA, type=str,
                        help='Path to test files folder')
    parser.add_argument('--min-threshold', default=512, type=int,
                        help='Min sentence lengths')
    parser.add_argument('--max-threshold', default=512, type=int,
                        help='Max sentence lengths')
    parser.add_argument('--train-dataset-size', default=1_000_000, type=int,
                        help='Train dataset size')
    parser.add_argument('--val-dataset-size', default=400, type=int,
                        help='Validation dataset size')
    parser.add_argument('--vis-dataset-size', default=5, type=int,
                        help='Visualization dataset size')
    parser.add_argument('--test-dataset-size', default=200, type=int,
                        help='Test dataset size')
    parser.add_argument('--min-noise', default=0, type=int,
                        help='Min noise range')
    parser.add_argument('--max-noise', default=1, type=int,
                        help='Max noise range')
    parser.add_argument('--n-workers', default=3, type=int,
                        help='Number of processes for dataloaders')

    return parser


def get_optimization_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('Optimization arguments', add_help=add_help)

    # CPULamb8Bit optimizer
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate value')
    parser.add_argument('--adam-beta1', default=0.9, type=float,
                        help='Coefficient for computing running averages of gradient and its square')
    parser.add_argument('--adam-beta2', default=0.96, type=float,
                        help='Coefficient for computing running averages of gradient and its square')
    parser.add_argument('--adam-epsilon', default=1e-6, type=float,
                        help='Term added to the denominator to improve numerical stability')
    parser.add_argument('--weight-decay', default=0.045, type=float,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--max-grad-norm', default=4.0, type=float,
                        help='Max norm of the gradients for clipping')
    parser.add_argument('--clamp-value', default=1e9, type=float,
                        help='Clamp weight_norm in (0,clamp_value). Set to a high value to avoid it (e.g 10e9)')

    # Scheduler
    parser.add_argument('--warmup-steps', default=1500, type=int,
                        help='Warmup steps value for learning rate scheduler')
    parser.add_argument('--total-steps', default=31250, type=int,
                        help='Total number of collaborative optimizer updates, used for learning rate schedule')
    parser.add_argument('--min-lr', default=1e-4, type=float,
                        help='Minimum value for learning rate schedule')

    # Collaborative optimizer
    parser.add_argument('--verbose', action='store_true',
                        help='Whether to show collaborative optimizer logs')
    parser.add_argument('--state-path', default=exp_var.COLLAB_STATE_PATH, type=Path,
                        help='Path to state backup file. Load this state upon init and when '
                             'recovering from NaN parameters')
    parser.add_argument('--backup-every-step', default=None, type=int,
                        help='Update collab state backup on disk once in this many global steps. '
                             'Default: do not update local state')
    parser.add_argument('--batch-size', default=None, type=int,
                        help='Batch size that fits into accelerator memory')
    parser.add_argument('--accumulate-batches', default=1, type=int,
                        help='Number of steps for gradients accumulation')
    parser.add_argument('--target-batch-size', default=512, type=int,
                        help='Perform optimizer step after all peers collectively accumulate this many samples')
    parser.add_argument('--matchmaking-time', default=30, type=float,
                        help='Averaging group will wait for stragglers for at most this many seconds')
    parser.add_argument('--allreduce-timeout', default=80, type=float,
                        help='Give up on a given all-reduce round after this many seconds')
    parser.add_argument('--averaging-timeout', default=200, type=float,
                        help='Give up on averaging step after this many seconds')
    parser.add_argument('--no-reuse-grad-buffers', action='store_true',
                        help="Whether or not to use model's grad buffers for accumulating gradients across local steps."
                             " This optimization reduces GPU memory consumption but may result in incorrect gradients "
                             "when using some advanced techniques (e.g. applying custom loss scaler)")

    # Collaborative averager
    parser.add_argument('--bandwidth', default=None, type=float,
                        help='Min(upload & download speed) in megabits/s, used to assign averaging tasks between peers')
    parser.add_argument('--min_vector_size', default=4_000_000, type=int,
                        help='Minimum slice of gradients assigned to one reducer, should be same across peers')

    return parser


def get_dht_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('DHT arguments', add_help=add_help)

    parser.add_argument('--experiment-prefix', default='trecover', type=str,
                        help='A unique experiment name, used as prefix for all DHT keys')
    parser.add_argument('--client-mode', action='store_true',
                        help='If True, runs training without incoming connections, in a firewall-compatible mode')
    parser.add_argument('--initial-peers', action='extend', type=str, nargs='+',
                        help='Multiaddrs of the peers that will welcome you into the existing collaboration')
    parser.add_argument('--use-ipfs', action='store_true',
                        help='Use IPFS to find initial_peers. If enabled, you only need to provide '
                             '/p2p/XXXX part of multiaddrs for the initial_peers (no need to specify '
                             'a particular IPv4/IPv6 address and port)')
    parser.add_argument('--host-maddrs', default=['/ip4/0.0.0.0/tcp/0', '/ip4/0.0.0.0/udp/0/quic'],
                        action='extend', type=str, nargs='+',
                        help='Multiaddrs to listen for external connections from other p2p instances. '
                             'To specify, for example, tcp port use /ip4/0.0.0.0/tcp/<port number>')
    parser.add_argument('--announce-maddrs', action='extend', type=str, nargs='+',
                        help='Visible multiaddrs the host announces for external connections from other p2p instances')
    parser.add_argument('--identity-path', type=Path,
                        help='Path to a pre-generated private key file. If defined, makes the peer ID deterministic')

    return parser


def get_wandb_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('W&B arguments', add_help=add_help)

    parser.add_argument('--wandb-project', default='TRecover', type=str,
                        help='Name of Weights & Biases project to report the training progress to')
    parser.add_argument('--wandb-id', default=None, type=str,
                        help='Id of the previous run to resume it')
    parser.add_argument('--wandb-key', default=None, type=str,
                        help='Weights & Biases credentials token to log in')
    parser.add_argument('--wandb-registry', default=exp_var.WANDB_REGISTRY_DIR, type=Path,
                        help='Default path for Weights & Biases logs and weights')
    parser.add_argument('--delay-in-steps', default=1, type=int,
                        help='The delay in displaying (reporting to W&B) the current status '
                             'of metrics (visualization) in such a number of steps')
    parser.add_argument('--delay-in-seconds', default=1300, type=float,
                        help='The delay in displaying (reporting to W&B) the current status '
                             'of metrics (visualization) for a maximum of such time in seconds')

    return parser


def get_tune_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('Tune arguments', add_help=add_help)

    parser.add_argument('--tune-mode', default='power', type=str, choices=['power', 'binsearch'],
                        help='Tuning algorithm type')
    parser.add_argument('--tune-batch-size-init', default=1, type=int,
                        help='Initial value for batch size tune algorithm')
    parser.add_argument('--tune-trials', default=25, type=int,
                        help='Number of tune algorithm trials')
    parser.add_argument('--tune-strategy', action='store_true',
                        help='Whether to use a collaborative strategy for batch size tuning')
    parser.add_argument('--scale-tuned-batch-size', action='store_true',
                        help='Whether to scale the tuned batch size')
    parser.add_argument('--batch-size-scale-factor', default=var.BATCH_SIZE_SCALE_FACTOR, type=float,
                        help='Factor for scaling tuned batch size')

    return parser


def get_auxiliary_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('Auxiliary arguments', add_help=add_help,
                            parents=[
                                get_sync_parser(add_help=False),
                                get_dht_parser(add_help=False),
                                get_model_parser(add_help=False),
                                get_data_parser(add_help=False),
                                get_optimization_parser(add_help=False)
                            ])

    parser.add_argument('--assist-refresh', default=5, type=float,
                        help='Period (in seconds) for trying to assist averaging')
    parser.add_argument('--as-active-peer', action='store_true',
                        help='Allow to share state with other peers otherwise only assist in averaging')

    return parser


def get_visualization_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('Visualization arguments',
                            add_help=add_help,
                            parents=[
                                get_wandb_parser(add_help=False),
                                get_auxiliary_parser(add_help=False)
                            ])

    parser.add_argument('--delimiter', default='', type=str,
                        help='Visualization columns delimiter')
    parser.add_argument('--visualize-every-step', default=None, type=int,
                        help='Perform visualization once in this many global steps.')
    parser.add_argument('--visualizer-refresh-period', default=10, type=float,
                        help='Period (in seconds) to check for visualization.')
    parser.add_argument('--assist-in-averaging', action='store_true',
                        help='If True, this peer will facilitate averaging for other (training) peers')

    return parser


def get_monitor_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('Monitor arguments', add_help=add_help, parents=[get_visualization_parser(add_help=False)])

    parser.add_argument('--refresh-period', default=10, type=float,
                        help='Period (in seconds) for fetching the metrics from DHT')
    parser.add_argument('--upload-state', action='store_true',
                        help='Whether to upload collab state to Weights & Biases. Default: do not upload.'
                             'Also you need to specify `--backup-every-step` argument')

    return parser


def get_train_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('Train loop arguments', add_help=add_help,
                            parents=[
                                get_sync_parser(add_help=False),
                                get_dht_parser(add_help=False),
                                get_model_parser(add_help=False),
                                get_data_parser(add_help=False),
                                get_optimization_parser(add_help=False),
                                get_tune_parser(add_help=False)
                            ])

    parser.add_argument('--pl-registry', default=exp_var.LIGHTNING_REGISTRY_DIR, type=Path,
                        help='Default path for pytorch-lightning logs and weights')
    parser.add_argument('--accelerator', default='auto', type=str, choices=['cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'auto'],
                        help='Train accelerator type')
    parser.add_argument('--n-epochs', default=10 ** 20, type=int,
                        help='Number of epochs for training')
    parser.add_argument('--statistics-expiration', default=600, type=float,
                        help='Statistics will be removed if not updated in this many seconds')

    return parser
