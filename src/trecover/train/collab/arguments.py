from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch

from trecover.config import var, exp_var


@dataclass
class ModelArguments:
    """ Model configuration """

    token_size: int = len(var.ALPHABET)
    pe_max_len: int = 256
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 768
    dropout: float = 0.1


@dataclass
class DataArguments:
    """ Arguments for dataloaders """

    # seed: int = 2531
    train_files: Path = exp_var.TRAIN_DATA
    val_files: Path = exp_var.VAL_DATA
    vis_files: Path = exp_var.VIS_DATA
    test_files: Path = exp_var.VIS_DATA
    min_threshold: int = 256
    max_threshold: int = 256
    train_dataset_size: int = 10
    val_dataset_size: int = 10
    vis_dataset_size: int = 5
    test_dataset_size: int = 16
    min_noise: int = 0
    max_noise: int = 1


@dataclass
class PLTrainerArguments:
    """ Arguments for pytorch-lightning trainer, optimizer, scheduler"""

    # Pytorch-lightning Trainer
    default_root_dir: Path = var.EXPERIMENTS_DIR
    enable_progress_bar: bool = True
    accelerator: str = 'auto'
    dataloader_num_workers: int = 2
    auto_select_gpus: bool = True
    max_steps: int = 10 ** 20
    num_sanity_val_steps: int = 0
    batch_size: int = None
    accumulate_grad_batches: int = 1
    scale_batch_size_init_val: int = 1
    tune_max_trials: int = 25
    tune_strategy: bool = False
    scale_tuned_batch_size: bool = False

    @property
    def batch_size_per_step(self) -> Optional[int]:
        """ Compute the number of training sequences contributed by each .step() from this peer """

        if self.batch_size:
            total_batch_size_per_step = self.batch_size * self.accumulate_grad_batches
            if torch.cuda.device_count() > 0:
                total_batch_size_per_step *= torch.cuda.device_count()

            return total_batch_size_per_step

        return None

    # Visualization step
    n_columns_to_show: int = 96
    delimiter: str = ''

    # Optimizer
    learning_rate: float = 0.0025
    adam_beta1: float = 0.9
    adam_beta2: float = 0.96
    adam_epsilon: float = 1e-6
    weight_decay: float = 0.045

    # Scheduler
    total_steps: int = 31250  # total number of collaborative SGD updates, used for learning rate schedule
    warmup_steps: int = 3125


@dataclass
class CollaborativeArguments:
    """ Configuration for Collaborative Optimizer and its internals """

    target_batch_size: int = field(
        default=256,
        metadata={'help': 'Perform optimizer step after all peers collectively accumulate this many samples'},
    )
    matchmaking_time: float = field(
        default=15.0, metadata={'help': 'Averaging group will wait for stragglers for at most this many seconds'}
    )
    allreduce_timeout: float = field(
        default=60, metadata={'help': 'Give up on a given all-reduce round after this many seconds'}
    )
    averaging_timeout: float = field(
        default=180, metadata={'help': 'Give up on averaging step after this many seconds'}
    )
    reuse_grad_buffers: bool = field(default=True, metadata={
        'help': "Whether or not to use model's .grad buffers for accumulating gradients across local steps. This "
                "optimization reduces GPU memory consumption but may result in incorrect gradients when using some "
                "advanced techniques (e.g. applying custom loss scaler)"})


@dataclass
class BasePeerArguments:
    """Base arguments that are used for both trainers and for auxiliary peers such as training monitor"""

    experiment_prefix: str = field(
        default='trecover',
        metadata={'help': 'A unique experiment name, used as prefix for all DHT keys'}
    )
    cache_dir: Optional[str] = field(  # TODO change as checkpoint?
        default='./cache',
        metadata={'help': 'Path to the cache'}
    )
    client_mode: bool = field(
        default=False,
        metadata={'help': 'Of True, runs training without incoming connections, in a firewall-compatible mode'},
    )
    # initial_peers: List[str] = field(
    initial_peers: str = field(
        default_factory=list,
        metadata={
            'help': 'Multiaddrs of the peers that will welcome you into the existing collaboration. '
                    'Example: /ip4/203.0.113.1/tcp/31337/p2p/XXXX /ip4/203.0.113.2/udp/7777/quic/p2p/YYYY'
        },
    )
    use_ipfs: bool = field(
        default=False,
        metadata={
            'help': 'Use IPFS to find initial_peers. If enabled, you only need to provide /p2p/XXXX part of multiaddrs '
                    'for the initial_peers (no need to specify a particular IPv4/IPv6 address and port)'
        },
    )
    host_maddrs: List[str] = field(
        default_factory=lambda: ['/ip4/0.0.0.0/tcp/0'],
        metadata={
            'help': 'Multiaddrs to listen for external connections from other p2p instances. '
                    'Defaults to all IPv4 interfaces with TCP protocol: /ip4/0.0.0.0/tcp/0'
        },
    )
    announce_maddrs: List[str] = field(
        default_factory=list,
        metadata={'help': 'Visible multiaddrs the host announces for external connections from other p2p instances'},
    )
    identity_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Path to a pre-generated private key file. If defined, makes the peer ID deterministic. '
                    'May be generated using ``./p2p-keygen`` from ``go-libp2p-daemon``.'
        },
    )


@dataclass
class TrainingPeerArguments(BasePeerArguments):
    statistics_expiration: float = field(
        default=600,
        metadata={'help': 'Statistics will be removed if not updated in this many seconds'}
    )
    # backup_every_step: Optional[int] = field(
    #     default=None,
    #     metadata={'help': 'Update training state backup on disk once in this many global steps '
    #                       '(default = do not update local state)'}
    # )
    backup_every_step: int = None
    state_path: Path = field(
        default=exp_var.COLLAB_STATE_PATH,
        metadata={'help': 'Load this state upon init and when recovering from NaN parameters'})


@dataclass
class AuxiliaryPeerArguments(BasePeerArguments):
    """
    Arguments for run_aux_peer.py that is responsible for connecting peers to one another, tracking
    learning curves, assisting in all-reduce and uploading checkpoints to the hub
    """
    refresh_period: float = field(
        default=10,
        metadata={'help': 'Period (in seconds) for fetching the keys from DHT'}
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={'help': 'Name of Weights & Biases project to report the training progress to'}
    )
    save_checkpoint_step_interval: int = field(
        default=2,
        metadata={'help': 'Frequency (in steps) of fetching and saving state from peers'}
    )
    repo_url: Optional[str] = field(
        default=None,
        metadata={'help': 'URL of Hugging Face Hub repository to upload the model and optimizer states'}
    )
    local_path: Optional[str] = field(
        default='Repo',
        metadata={'help': 'Path to local repository to store the model and optimizer states'}
    )
    upload_interval: Optional[float] = field(
        default=None,
        metadata={'help': 'Frequency (in seconds) of uploading the model to Hub'}
    )
    store_checkpoints: bool = field(
        default=True,
        metadata={'help': 'If True, enables CheckpointHandler'}
    )
    assist_refresh: float = field(
        default=1.0,
        metadata={'help': 'Period (in seconds) for trying to assist averaging'}
    )
