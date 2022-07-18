from argparse import ArgumentParser
from pathlib import Path

from trecover.config import var, exp_var


def get_model_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('Model arguments', add_help=add_help)

    parser.add_argument('--token-size', default=len(var.ALPHABET), type=int,
                        help='Token size')
    parser.add_argument('--pe-max-len', default=256, type=int,
                        help='Positional encoding max length')
    parser.add_argument('--n-layers', default=12, type=int,
                        help='Number of encoder and decoder blocks')
    parser.add_argument('--d-model', default=768, type=int,
                        help='Model dimension - number of expected features in the encoder (decoder) input')
    parser.add_argument('--n-heads', default=12, type=int,
                        help='Number of encoder and decoder attention heads')
    parser.add_argument('--d-ff', default=768, type=int,
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
    parser.add_argument('--min-threshold', default=256, type=int,
                        help='Min sentence lengths')
    parser.add_argument('--max-threshold', default=256, type=int,
                        help='Max sentence lengths')
    parser.add_argument('--train-dataset-size', default=2000, type=int,
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

    parser.add_argument('--lr', default=0.0025, type=float,
                        help='Learning rate value')
    parser.add_argument('--adam-beta1', default=0.9, type=float,
                        help='Coefficient for computing running averages of gradient and its square')
    parser.add_argument('--adam-beta2', default=0.96, type=float,
                        help='Coefficient for computing running averages of gradient and its square')
    parser.add_argument('--adam-epsilon', default=1e-6, type=float,
                        help='Term added to the denominator to improve numerical stability')
    parser.add_argument('--weight-decay', default=0.045, type=float,
                        help='Weight decay (L2 penalty)')

    # Scheduler
    parser.add_argument('--warmup-steps', default=3125, type=int,
                        help='Warmup steps value for learning rate scheduler')
    parser.add_argument('--total-steps', default=31250, type=int,
                        help='Total number of collaborative optimizer updates, used for learning rate schedule')

    # Collaborative optimizer
    parser.add_argument('--batch-size', default=None, type=int,
                        help='Batch size that fits into accelerator memory')
    parser.add_argument('--accumulate-batches', default=1, type=int,
                        help='Number of steps for gradients accumulation')
    parser.add_argument('--target-batch-size', default=2048, type=int,
                        help='Perform optimizer step after all peers collectively accumulate this many samples')
    parser.add_argument('--matchmaking-time', default=15, type=float,
                        help='Averaging group will wait for stragglers for at most this many seconds')
    parser.add_argument('--allreduce-timeout', default=60, type=float,
                        help='Give up on a given all-reduce round after this many seconds')
    parser.add_argument('--averaging-timeout', default=180, type=float,
                        help='Give up on averaging step after this many seconds')
    parser.add_argument('--no-reuse-grad-buffers', action='store_true',
                        help="Whether or not to use model's grad buffers for accumulating gradients across local steps."
                             " This optimization reduces GPU memory consumption but may result in incorrect gradients "
                             "when using some advanced techniques (e.g. applying custom loss scaler)")

    return parser


# TODO --no-args-sync
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


def get_monitor_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('DHT arguments', add_help=add_help,
                            parents=[
                                get_dht_parser(add_help=False),
                                get_model_parser(add_help=False),
                                get_data_parser(add_help=False),
                                get_optimization_parser(add_help=False)
                            ])

    parser.add_argument('--refresh-period', default=3, type=float,
                        help='Period (in seconds) for fetching the metrics from DHT')
    parser.add_argument('--wandb-project', default='TRecover', type=str,
                        help='Name of Weights & Biases project to report the training progress to')
    parser.add_argument('--wandb-id', default=None, type=str,
                        help='Id of the previous run to resume it')
    parser.add_argument('--wandb-key', default=None, type=str,
                        help='Weights & Biases credentials token to log in')
    parser.add_argument('--wandb-registry', default=exp_var.WANDB_REGISTRY_DIR, type=Path,
                        help='Default path for Weights & Biases logs and weights')
    parser.add_argument('--monitor-state-path', default=exp_var.MONITOR_STATE_PATH, type=Path,
                        help='Path to state backup file')
    parser.add_argument('--upload-every-step', default=None, type=int,
                        help='Upload to Weights & Biases and backup on disk training state '
                             'once in this many global steps. Default: do not upload')
    parser.add_argument('--verbose', action='store_true',
                        help='Whether to show collaborative optimizer logs')
    parser.add_argument('--assist-in-averaging', action='store_true',
                        help='If True, this peer will facilitate averaging for other (training) peers')
    parser.add_argument('--assist-refresh', default=1, type=float,
                        help='Period (in seconds) for trying to assist averaging')

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


def get_train_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('Train loop arguments', add_help=add_help,
                            parents=[
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
    parser.add_argument('--backup-every-step', default=None, type=int,
                        help='Update training state backup on disk once in this many global steps. '
                             'Default: do not update local state')
    parser.add_argument('--state-path', default=exp_var.TRAIN_STATE_PATH, type=Path,
                        help='Path to state backup file. Load this state upon init and when '
                             'recovering from NaN parameters')

    return parser


def get_auxiliary_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('Auxiliary arguments', add_help=add_help,
                            parents=[
                                get_dht_parser(add_help=False),
                                get_model_parser(add_help=False),
                                get_data_parser(add_help=False),
                                get_optimization_parser(add_help=False)
                            ])

    parser.add_argument('--verbose', action='store_true',
                        help='Whether to show collaborative optimizer logs')
    parser.add_argument('--assist-refresh', default=1, type=float,
                        help='Period (in seconds) for trying to assist averaging')

    return parser


def get_visualization_parser(add_help: bool = True) -> ArgumentParser:
    parser = ArgumentParser('Visualization arguments', add_help=add_help)

    parser.add_argument('--n-columns-to-show', default=96, type=int,
                        help='Number of visualization columns to show')
    parser.add_argument('--delimiter', default='', type=str,
                        help='Visualization columns delimiter')

    return parser
